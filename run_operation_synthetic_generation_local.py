import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import subprocess
from collections import Counter, defaultdict
import re


def print_nvidia_smi(title):
    print("\n" + "=" * 100, flush=True)
    print(title, flush=True)
    print("=" * 100, flush=True)

    try:
        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        print(result.stdout, flush=True)
    except Exception as exc:
        print(f"Could not run nvidia-smi: {exc}", flush=True)


def print_model_placement_report(model):
    print("\n" + "=" * 100, flush=True)
    print("MODEL PLACEMENT REPORT", flush=True)
    print("=" * 100, flush=True)

    device_map = getattr(model, "hf_device_map", None)

    if device_map is None:
        print("model.hf_device_map not found.", flush=True)
    else:
        device_counter = Counter(str(device) for device in device_map.values())
        print("Device map summary:", flush=True)
        print(dict(device_counter), flush=True)

        offloaded = {
            module_name: str(device)
            for module_name, device in device_map.items()
            if str(device).lower() in {"cpu", "disk", "meta"}
        }

        if offloaded:
            print("\nCPU/DISK/META OFFLOADING DETECTED:", flush=True)
            for module_name, device in offloaded.items():
                print(f"{module_name} -> {device}", flush=True)
        else:
            print("\nNo CPU/disk/meta offloading detected in hf_device_map.", flush=True)

        print("\nFirst 80 device-map entries:", flush=True)
        for i, (module_name, device) in enumerate(device_map.items()):
            if i >= 80:
                break
            print(f"{module_name} -> {device}", flush=True)

    param_count_by_device = Counter()
    param_bytes_by_device = defaultdict(int)

    for _, param in model.named_parameters():
        device = str(param.device)
        param_count_by_device[device] += param.numel()
        try:
            param_bytes_by_device[device] += param.numel() * param.element_size()
        except Exception:
            pass

    total_params = sum(param_count_by_device.values())

    print("\nParameter summary by device:", flush=True)
    for device in sorted(param_count_by_device):
        n_params = param_count_by_device[device]
        pct = (100.0 * n_params / total_params) if total_params else 0.0
        gib = param_bytes_by_device[device] / (1024 ** 3)
        print(
            f"{device}: params={n_params} pct={pct:.4f}% estimated_param_gib={gib:.4f}",
            flush=True,
        )

    suspicious_devices = [
        device
        for device in param_count_by_device
        if device == "cpu" or device == "meta" or "disk" in device.lower()
    ]

    if suspicious_devices:
        print("\nCPU/META/DISK PARAMETER STORAGE DETECTED.", flush=True)
    else:
        print("\nNo CPU/meta/disk parameter storage detected.", flush=True)

    if torch.cuda.is_available():
        print("\nTorch CUDA memory:", flush=True)
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            max_allocated = torch.cuda.max_memory_allocated(i) / (1024 ** 3)
            name = torch.cuda.get_device_name(i)
            print(
                f"cuda:{i} {name}: allocated={allocated:.4f} GiB reserved={reserved:.4f} GiB max_allocated={max_allocated:.4f} GiB",
                flush=True,
            )
    else:
        print("\nTorch says CUDA is not available.", flush=True)

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_done_ids(path):
    done = set()
    path = Path(path)
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                done.add(row.get("task_id"))
            except Exception:
                continue
    return done


def dtype_from_name(name):
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    return torch.bfloat16


def extract_json_object(text):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        inner = fenced.group(1).strip()
        try:
            return json.loads(inner)
        except Exception:
            pass

    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object start found")

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    return json.loads(candidate)

    raise ValueError("no complete JSON object found")


def build_chat_prompt(tokenizer, task_prompt):
    messages = [
        {
            "role": "system",
            "content": "You generate strict JSON for supervised data construction. Return JSON only. Do not use markdown."
        },
        {
            "role": "user",
            "content": task_prompt
        }
    ]

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return (
        "System: You generate strict JSON for supervised data construction. Return JSON only. Do not use markdown.\n\n"
        "User:\n"
        + task_prompt
        + "\n\nAssistant:\n"
    )

def enforce_examples_per_task(task, examples_per_task):
    if examples_per_task is None:
        return task

    task["examples_requested"] = examples_per_task

    prompt = task.get("prompt")
    if isinstance(prompt, str):
        replacements = [
            (r"exactly\s+\d+\s+examples", f"exactly {examples_per_task} examples"),
            (r"exactly\s+five\s+examples", f"exactly {examples_per_task} examples"),
            (r"exactly\s+two\s+examples", f"exactly {examples_per_task} examples"),
            (r"generate\s+\d+\s+examples", f"generate {examples_per_task} examples"),
            (r"generate\s+five\s+examples", f"generate {examples_per_task} examples"),
            (r"generate\s+two\s+examples", f"generate {examples_per_task} examples"),
            (r"create\s+\d+\s+examples", f"create {examples_per_task} examples"),
            (r"create\s+five\s+examples", f"create {examples_per_task} examples"),
            (r"create\s+two\s+examples", f"create {examples_per_task} examples"),
            (r"produce\s+\d+\s+examples", f"produce {examples_per_task} examples"),
            (r"produce\s+five\s+examples", f"produce {examples_per_task} examples"),
            (r"produce\s+two\s+examples", f"produce {examples_per_task} examples"),
            (r"\"examples\"\s+array\s+must\s+contain\s+exactly\s+\d+", f"\"examples\" array must contain exactly {examples_per_task}"),
            (r"examples_requested\s*[:=]\s*\d+", f"examples_requested: {examples_per_task}"),
        ]

        for pattern, replacement in replacements:
            prompt = re.sub(pattern, replacement, prompt, flags=re.IGNORECASE)

        task["prompt"] = prompt

    return task

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=3000)
    parser.add_argument("--max_input_tokens", type=int, default=24576)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--quantization", default="none", choices=["none", "4bit", "8bit"])
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--gpu_memory", default="76GiB")
    parser.add_argument("--cpu_memory", default="96GiB")
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--examples_per_task", type=int, default=None)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    model_kwargs = {
        "torch_dtype": dtype_from_name(args.dtype),
        "device_map": "auto",
        "trust_remote_code": True,
        "attn_implementation": args.attn_implementation,
        "max_memory": {0: args.gpu_memory, "cpu": args.cpu_memory}
    }

    if args.quantization in {"4bit", "8bit"}:
        from transformers import BitsAndBytesConfig
        if args.quantization == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype_from_name(args.dtype),
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    print_nvidia_smi("NVIDIA-SMI BEFORE MODEL LOAD")

    print(f"Loading model: {args.model}", flush=True)
    print("Model kwargs:", json.dumps({k: str(v) for k, v in model_kwargs.items()}, indent=2), flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()
    print_model_placement_report(model)
    print_nvidia_smi("NVIDIA-SMI AFTER MODEL LOAD")

    done = load_done_ids(output_path)
    print(f"Already completed tasks: {len(done)}", flush=True)

    completed = 0
    parse_failures = 0
    invalid_schema = 0
    start_all = time.time()

    for task in read_jsonl(args.input):
        task_id = task.get("task_id")
        if task_id in done:
            continue

        if args.max_tasks is not None and completed >= args.max_tasks:
            break
        
        task = enforce_examples_per_task(task, args.examples_per_task)
        prompt = build_chat_prompt(tokenizer, task["prompt"])
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_input_tokens
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        start = time.time()
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature if args.do_sample else None,
                top_p=args.top_p if args.do_sample else None,
                pad_token_id=tokenizer.eos_token_id
            )

        new_tokens = generated[0][inputs["input_ids"].shape[-1]:]
        raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        seconds = time.time() - start

        parsed_output = None
        valid_json = False
        validation_error = "ok"

        try:
            parsed_output = extract_json_object(raw_output)
            valid_json = isinstance(parsed_output, dict) and isinstance(parsed_output.get("examples"), list)
            if not valid_json:
                validation_error = "parsed JSON does not contain examples list"
                invalid_schema += 1
        except Exception as e:
            validation_error = f"parse_failed: {type(e).__name__}: {e}"
            parse_failures += 1

        row = {
            "task_id": task_id,
            "task_type": task.get("task_type"),
            "target_label": task.get("target_label"),
            "examples_requested": task.get("examples_requested"),
            "seed_scenario": task.get("seed_scenario"),
            "anchor_count": task.get("anchor_count"),
            "anchors": task.get("anchors", []),
            "model": args.model,
            "quantization": args.quantization,
            "dtype": args.dtype,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "raw_output": raw_output,
            "parsed_output": parsed_output,
            "valid_json_schema": valid_json,
            "validation_error": validation_error,
            "generation_seconds": seconds
        }

        append_jsonl(output_path, row)
        completed += 1

        if completed % 10 == 0:
            print(
                f"done={completed} parse_fail={parse_failures} invalid={invalid_schema} runtime_sec={time.time() - start_all:.1f}",
                flush=True
            )

    print("Finished.", flush=True)
    print(f"New tasks completed: {completed}", flush=True)
    print(f"Parse failures: {parse_failures}", flush=True)
    print(f"Invalid schema: {invalid_schema}", flush=True)
    print(f"Output: {output_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
