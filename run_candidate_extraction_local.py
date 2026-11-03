import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_done_ids(output_path):
    done = set()
    if not output_path.exists():
        return done

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                done.add(obj.get("task_id"))
            except Exception:
                continue

    return done


def extract_json_object(text):
    text = text.strip()

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    first = text.find("{")
    last = text.rfind("}")

    if first == -1 or last == -1 or last <= first:
        return None

    candidate = text[first:last + 1]

    try:
        return json.loads(candidate)
    except Exception:
        return None


def validate_extraction(obj):
    if not isinstance(obj, dict):
        return False, "output_not_dict"

    memories = obj.get("memories")
    if not isinstance(memories, list):
        return False, "missing_memories_list"

    allowed_memory_types = {
        "episodic",
        "semantic",
        "preference",
        "procedural",
        "sensitive",
        "authorization",
        "project_state",
    }

    allowed_sensitivity = {"low", "medium", "high"}
    allowed_stability = {"temporary", "session_relevant", "long_term", "unknown"}

    for idx, mem in enumerate(memories):
        if not isinstance(mem, dict):
            return False, f"memory_{idx}_not_dict"

        if not isinstance(mem.get("memory_text"), str) or not mem["memory_text"].strip():
            return False, f"memory_{idx}_missing_text"

        if mem.get("memory_type") not in allowed_memory_types:
            return False, f"memory_{idx}_invalid_type"

        if mem.get("sensitivity") not in allowed_sensitivity:
            return False, f"memory_{idx}_invalid_sensitivity"

        if mem.get("stability") not in allowed_stability:
            return False, f"memory_{idx}_invalid_stability"

        if not isinstance(mem.get("should_write"), bool):
            return False, f"memory_{idx}_invalid_should_write"

        if not isinstance(mem.get("reason"), str):
            return False, f"memory_{idx}_missing_reason"

        if not isinstance(mem.get("evidence_span"), str):
            return False, f"memory_{idx}_missing_evidence"

    return True, "ok"


def build_messages(prompt):
    return [
        {
            "role": "system",
            "content": (
                "You are a strict dataset annotation assistant. "
                "You must return valid JSON only. Do not include markdown. "
                "Do not invent facts. Follow the provided schema exactly."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]


def resolve_dtype(dtype_name):
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "auto":
        return "auto"
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def load_model_and_tokenizer(args):
    model_path = args.model
    local_path = BASE_DIR / model_path
    if local_path.exists():
        model_path = str(local_path)

    print(f"Loading tokenizer: {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = resolve_dtype(args.dtype)

    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": args.device_map,
        "trust_remote_code": True,
    }

    if args.attn_implementation != "auto":
        model_kwargs["attn_implementation"] = args.attn_implementation

    if args.quantization in {"8bit", "4bit"}:
        try:
            from transformers import BitsAndBytesConfig
        except Exception as e:
            raise RuntimeError(
                "bitsandbytes quantization was requested, but BitsAndBytesConfig could not be imported. "
                "Install with: pip install -U bitsandbytes"
            ) from e

        if args.quantization == "8bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        if args.quantization == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        model_kwargs.pop("torch_dtype", None)

    max_memory = {}
    if args.gpu_memory:
        max_memory[0] = args.gpu_memory
    if args.cpu_memory:
        max_memory["cpu"] = args.cpu_memory
    if max_memory:
        model_kwargs["max_memory"] = max_memory

    print(f"Loading model: {model_path}", flush=True)
    print(f"Model kwargs: {json.dumps({k: str(v) for k, v in model_kwargs.items()}, indent=2)}", flush=True)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs,
        )
    except Exception as e:
        if args.attn_implementation == "flash_attention_2":
            print("flash_attention_2 failed. Retry with --attn_implementation sdpa", flush=True)
        raise e

    model.eval()
    return model, tokenizer


def generate_one(model, tokenizer, prompt, max_new_tokens, temperature, max_input_tokens):
    messages = build_messages(prompt)

    if hasattr(tokenizer, "apply_chat_template"):
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        input_text = (
            "System: " + messages[0]["content"] + "\n\n"
            "User: " + messages[1]["content"] + "\n\n"
            "Assistant:"
        )

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    ).to(model.device)

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }

    if temperature > 0:
        generation_kwargs["temperature"] = temperature

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            **generation_kwargs,
        )

    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="raw/models/Qwen2.5-32B-Instruct",
        help="Hugging Face model ID or local path relative to memory_gov_data.",
    )
    parser.add_argument(
        "--input",
        default="processed/longmemeval_candidate_extraction_tasks.jsonl",
    )
    parser.add_argument(
        "--output",
        default="processed/longmemeval_candidate_extractions_raw_qwen32b.jsonl",
    )
    parser.add_argument(
        "--max_tasks",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1800,
    )
    parser.add_argument(
        "--max_input_tokens",
        type=int,
        default=32768,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--device_map",
        default="auto",
    )
    parser.add_argument(
        "--gpu_memory",
        default="76GiB",
        help="Maximum memory for GPU 0. Use lower values if sharing GPU.",
    )
    parser.add_argument(
        "--cpu_memory",
        default="96GiB",
    )
    parser.add_argument(
        "--quantization",
        default="none",
        choices=["none", "8bit", "4bit"],
    )
    parser.add_argument(
        "--attn_implementation",
        default="sdpa",
        choices=["auto", "sdpa", "flash_attention_2", "eager"],
    )

    args = parser.parse_args()

    input_path = BASE_DIR / args.input
    output_path = BASE_DIR / args.output

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    model, tokenizer = load_model_and_tokenizer(args)

    done_ids = load_done_ids(output_path)
    print(f"Already completed tasks: {len(done_ids)}", flush=True)

    total_seen = 0
    total_done = 0
    total_failed_parse = 0
    total_invalid = 0

    start_time = time.time()

    for task in read_jsonl(input_path):
        task_id = task.get("task_id")
        total_seen += 1

        if args.max_tasks is not None and total_done >= args.max_tasks:
            break

        if task_id in done_ids:
            continue

        prompt = task["prompt"]

        t0 = time.time()
        try:
            raw_output = generate_one(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                max_input_tokens=args.max_input_tokens,
            )
            error = None
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            raw_output = ""
            error = f"CUDA_OOM: {str(e)}"
        except Exception as e:
            raw_output = ""
            error = f"ERROR: {str(e)}"

        elapsed = time.time() - t0

        parsed = extract_json_object(raw_output) if raw_output else None
        is_valid, validation_error = validate_extraction(parsed)

        if error is not None:
            validation_error = error

        if parsed is None:
            total_failed_parse += 1
        elif not is_valid:
            total_invalid += 1

        row = {
            "task_id": task_id,
            "source_dataset": task.get("source_dataset"),
            "source_file": task.get("source_file"),
            "source_example_id": task.get("source_example_id"),
            "question_id": task.get("question_id"),
            "session_id": task.get("session_id"),
            "scenario_type": task.get("scenario_type"),
            "current_query": task.get("current_query"),
            "expected_answer": task.get("expected_answer"),
            "candidate_valid_from": task.get("candidate_valid_from"),
            "model": args.model,
            "quantization": args.quantization,
            "dtype": args.dtype,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "raw_output": raw_output,
            "parsed_output": parsed,
            "valid_json_schema": bool(is_valid),
            "validation_error": validation_error,
            "generation_seconds": elapsed,
        }

        append_jsonl(output_path, row)
        done_ids.add(task_id)
        total_done += 1

        if total_done % 5 == 0:
            runtime = time.time() - start_time
            print(
                f"done={total_done} seen={total_seen} parse_fail={total_failed_parse} invalid={total_invalid} runtime_sec={runtime:.1f}",
                flush=True,
            )

    print("Finished.", flush=True)
    print(f"New tasks completed: {total_done}", flush=True)
    print(f"Parse failures: {total_failed_parse}", flush=True)
    print(f"Invalid schema: {total_invalid}", flush=True)
    print(f"Output: {output_path}", flush=True)


if __name__ == "__main__":
    main()
