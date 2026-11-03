import argparse
import json
import os
import threading
import time
import warnings

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer


os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=FutureWarning)

APP_STATE = {
    "model": None,
    "tokenizer": None,
    "model_name": None,
    "device": None,
    "dtype": None,
}

LOCK = threading.Lock()
app = FastAPI()


def dtype_from_name(name):
    name = name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def build_max_memory(gpu_memory, cpu_memory):
    if not gpu_memory and not cpu_memory:
        return None

    max_memory = {}

    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        for i in range(n):
            if gpu_memory:
                max_memory[i] = gpu_memory

    if cpu_memory:
        max_memory["cpu"] = cpu_memory

    return max_memory or None


def load_heavy_model(args):
    dtype = dtype_from_name(args.dtype)

    print()
    print("Loading answer model")
    print(f"model={args.model}")
    print(f"dtype={args.dtype}")
    print(f"quantization={args.quantization}")
    print(f"attn_implementation={args.attn_implementation}")
    print(f"local_files_only={args.local_files_only}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None

    if args.quantization == "4bit":
        try:
            from transformers import BitsAndBytesConfig
        except Exception as exc:
            raise RuntimeError(
                "4-bit quantization requested, but BitsAndBytesConfig could not be imported. "
                "Install bitsandbytes or use --quantization none."
            ) from exc

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    max_memory = build_max_memory(args.gpu_memory, args.cpu_memory)

    load_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": dtype,
        "device_map": args.device_map,
        "local_files_only": args.local_files_only,
    }

    if args.attn_implementation:
        load_kwargs["attn_implementation"] = args.attn_implementation

    if max_memory:
        load_kwargs["max_memory"] = max_memory

    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        **load_kwargs,
    )

    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    APP_STATE.update({
        "model": model,
        "tokenizer": tokenizer,
        "model_name": args.model,
        "device": device,
        "dtype": args.dtype,
    })

    print()
    print("Answer model loaded")
    print(f"device={device}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            name = torch.cuda.get_device_name(i)
            print(f"cuda:{i} {name} allocated={alloc:.2f}GiB reserved={reserved:.2f}GiB")
    print()


def build_messages(user_message, answer_context):
    system = (
        "You are a helpful assistant with a governed long-term memory system. "
        "Use retrieved memory context only when it is relevant to the user's current message. "
        "Do not mention internal memory IDs. "
        "Do not claim to remember information that is not in the provided memory context or the current message. "
        "If the memory context is irrelevant, ignore it."
    )

    user = (
        "Retrieved memory context:\n"
        f"{answer_context.strip() if answer_context else 'No retrieved memory context.'}\n\n"
        "Current user message:\n"
        f"{user_message.strip()}"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def generate_answer(
    user_message,
    answer_context,
    max_new_tokens=512,
    temperature=0.2,
    top_p=0.9,
    repetition_penalty=1.05,
):
    model = APP_STATE["model"]
    tokenizer = APP_STATE["tokenizer"]

    if model is None or tokenizer is None:
        raise RuntimeError("model not loaded")

    messages = build_messages(user_message, answer_context)

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        truncation=False,
    )

    first_device = next(model.parameters()).device
    inputs = {k: v.to(first_device) for k, v in inputs.items()}

    do_sample = temperature is not None and temperature > 0

    gen_kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": do_sample,
        "repetition_penalty": float(repetition_penalty),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if do_sample:
        gen_kwargs["temperature"] = float(temperature)
        gen_kwargs["top_p"] = float(top_p)

    t0 = time.perf_counter()

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            **gen_kwargs,
        )

    elapsed = time.perf_counter() - t0

    input_len = inputs["input_ids"].shape[-1]
    new_ids = output_ids[0][input_len:]

    answer = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    return {
        "answer": answer,
        "latency_sec": elapsed,
        "input_tokens": int(input_len),
        "output_tokens": int(new_ids.shape[-1]),
        "model": APP_STATE["model_name"],
    }


@app.get("/health")
def health():
    return {
        "ok": APP_STATE["model"] is not None,
        "model": APP_STATE["model_name"],
        "device": APP_STATE["device"],
        "dtype": APP_STATE["dtype"],
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


@app.post("/generate")
async def generate(request: Request):
    data = await request.json()

    user_message = data.get("user_message") or ""
    answer_context = data.get("answer_context") or ""

    if not user_message.strip():
        return JSONResponse({"ok": False, "error": "empty user_message"}, status_code=400)

    max_new_tokens = data.get("max_new_tokens", 512)
    temperature = data.get("temperature", 0.2)
    top_p = data.get("top_p", 0.9)
    repetition_penalty = data.get("repetition_penalty", 1.05)

    try:
        with LOCK:
            out = generate_answer(
                user_message=user_message,
                answer_context=answer_context,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
        out["ok"] = True
        return out
    except Exception as exc:
        return JSONResponse(
            {"ok": False, "error": repr(exc)},
            status_code=500,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--quantization", choices=["none", "4bit"], default="none")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--gpu_memory", default=None)
    parser.add_argument("--cpu_memory", default=None)
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8600)
    args = parser.parse_args()

    load_heavy_model(args)

    print("Answer model server")
    print(f"url=http://{args.host}:{args.port}")
    print()

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
