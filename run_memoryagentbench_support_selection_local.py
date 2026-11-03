import argparse
import gc
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for",
    "from", "had", "has", "have", "he", "her", "him", "his", "how", "i", "in",
    "is", "it", "its", "me", "my", "of", "on", "or", "our", "she", "that",
    "the", "their", "them", "there", "they", "this", "to", "was", "were",
    "what", "when", "where", "which", "who", "whom", "whose", "why", "with",
    "you", "your", "fact", "facts", "provides", "indicates", "shows",
    "establishes", "associates", "associated", "while", "answer"
}


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def load_done_ids(output_path):
    done = set()
    if not output_path.exists():
        return done

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                task_id = row.get("task_id")
                if task_id:
                    done.add(task_id)
            except Exception:
                continue

    return done


def dtype_from_string(name):
    name = str(name).lower()
    if name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if name in {"float16", "fp16"}:
        return torch.float16
    if name in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def parse_max_memory(gpu_memory, cpu_memory):
    max_memory = {}
    if torch.cuda.is_available():
        max_memory[0] = gpu_memory
    if cpu_memory:
        max_memory["cpu"] = cpu_memory
    return max_memory


def load_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    model_kwargs = {
        "torch_dtype": dtype_from_string(args.dtype),
        "device_map": "auto",
        "trust_remote_code": True,
        "attn_implementation": args.attn_implementation,
        "max_memory": parse_max_memory(args.gpu_memory, args.cpu_memory),
    }

    if args.quantization in {"4bit", "8bit"}:
        from transformers import BitsAndBytesConfig

        if args.quantization == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype_from_string(args.dtype),
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )

    print(f"Loading tokenizer: {args.model}", flush=True)
    print(f"Loading model: {args.model}", flush=True)
    printable_kwargs = {
        k: str(v)
        for k, v in model_kwargs.items()
        if k != "quantization_config"
    }
    if "quantization_config" in model_kwargs:
        printable_kwargs["quantization_config"] = args.quantization
    print("Model kwargs:", json.dumps(printable_kwargs, indent=2), flush=True)

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def build_input_text(tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def get_input_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def generate_one(model, tokenizer, prompt, args):
    input_text = build_input_text(tokenizer, prompt)

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_input_tokens,
    )

    input_device = get_input_device()
    inputs = {k: v.to(input_device) for k, v in inputs.items()}

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[-1]
    generated_ids = output[0][input_len:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text.strip()


def extract_json_object(text):
    if not text:
        raise ValueError("empty model output")

    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned.strip(), flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned.strip()).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found")

    return json.loads(cleaned[start:end + 1])


def get_allowed_fact_ids(task):
    ids = []
    for fact in task.get("candidate_facts", []):
        if isinstance(fact, dict) and "fact_id" in fact:
            ids.append(str(fact["fact_id"]))
    return ids


def get_candidate_fact_map(task):
    mapping = {}
    for fact in task.get("candidate_facts", []):
        if isinstance(fact, dict) and "fact_id" in fact:
            mapping[str(fact["fact_id"])] = str(fact.get("text", ""))
    return mapping


def tokenize_for_repair(text):
    tokens = re.findall(r"[A-Za-z0-9]+", str(text).lower())
    return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]


def sentence_containing_id(reason, invalid_id):
    parts = re.split(r"(?<=[.!?])\s+", reason)
    for part in parts:
        if str(invalid_id) in part:
            return part
    return reason


def repair_invalid_id(invalid_id, reason, candidate_fact_map, already_used):
    segment = sentence_containing_id(reason, invalid_id)
    segment = re.sub(rf"\b[Ff]act\s+{re.escape(str(invalid_id))}\b", "", segment)
    seg_tokens = set(tokenize_for_repair(segment))

    if not seg_tokens:
        return None, 0.0

    best_id = None
    best_score = 0.0

    for fact_id, fact_text in candidate_fact_map.items():
        if fact_id in already_used:
            continue

        fact_tokens = set(tokenize_for_repair(fact_text))
        overlap = seg_tokens & fact_tokens
        score = float(len(overlap))

        if str(invalid_id).isdigit() and fact_id.isdigit():
            distance = abs(int(str(invalid_id)) - int(fact_id))
            if distance <= 3:
                score += 0.25

        if score > best_score:
            best_score = score
            best_id = fact_id

    if best_score >= 2.0:
        return best_id, best_score

    return None, best_score


def normalize_support_ids(parsed, task, repair_invalid_ids=True):
    allowed_ids = set(get_allowed_fact_ids(task))
    candidate_fact_map = get_candidate_fact_map(task)

    if not isinstance(parsed, dict):
        return None, False, "parsed output is not an object"

    if "support_fact_ids" not in parsed:
        return None, False, "missing support_fact_ids"

    if not isinstance(parsed["support_fact_ids"], list):
        return None, False, "support_fact_ids is not a list"

    reason = parsed.get("reason", "")
    if reason is None:
        reason = ""
    if not isinstance(reason, str):
        reason = str(reason)

    clean_ids = []
    removed_invalid_ids = []
    repaired_invalid_ids = {}
    repair_scores = {}

    for raw_id in parsed["support_fact_ids"]:
        fact_id = str(raw_id).strip()

        if not fact_id:
            continue

        if fact_id in allowed_ids:
            if fact_id not in clean_ids:
                clean_ids.append(fact_id)
            continue

        if repair_invalid_ids:
            repaired_id, score = repair_invalid_id(
                invalid_id=fact_id,
                reason=reason,
                candidate_fact_map=candidate_fact_map,
                already_used=set(clean_ids),
            )

            if repaired_id is not None and repaired_id in allowed_ids:
                if repaired_id not in clean_ids:
                    clean_ids.append(repaired_id)
                repaired_invalid_ids[fact_id] = repaired_id
                repair_scores[fact_id] = score
                continue

        removed_invalid_ids.append(fact_id)

    normalized = dict(parsed)
    normalized["support_fact_ids"] = clean_ids
    normalized["reason"] = reason

    if removed_invalid_ids:
        normalized["removed_invalid_support_fact_ids"] = removed_invalid_ids
    if repaired_invalid_ids:
        normalized["repaired_invalid_support_fact_ids"] = repaired_invalid_ids
        normalized["repair_scores"] = repair_scores

    return normalized, True, "ok"


def process_task(model, tokenizer, task, args):
    start_time = time.time()
    raw_output = generate_one(model, tokenizer, task["prompt"], args)
    generation_seconds = time.time() - start_time

    parsed = None
    normalized = None
    valid_schema = False
    validation_error = "ok"
    parse_error = None

    try:
        parsed = extract_json_object(raw_output)
        normalized, valid_schema, validation_error = normalize_support_ids(
            parsed=parsed,
            task=task,
            repair_invalid_ids=not args.disable_invalid_id_repair,
        )
    except Exception as exc:
        parse_error = str(exc)
        validation_error = parse_error
        valid_schema = False

    row = {
        "task_id": task.get("task_id"),
        "task_type": task.get("task_type"),
        "source_dataset": task.get("source_dataset"),
        "source_file": task.get("source_file"),
        "source_subset": task.get("source_subset"),
        "row_idx": task.get("row_idx"),
        "qa_idx": task.get("qa_idx"),
        "qa_pair_id": task.get("qa_pair_id"),
        "scenario_type": task.get("scenario_type"),
        "current_query": task.get("current_query"),
        "expected_answers": task.get("expected_answers"),
        "model": args.model,
        "quantization": args.quantization,
        "dtype": args.dtype,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "raw_output": raw_output,
        "raw_parsed_output": parsed,
        "parsed_output": normalized,
        "valid_json_schema": valid_schema,
        "validation_error": validation_error,
        "generation_seconds": generation_seconds,
    }

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_input_tokens", type=int, default=32768)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--quantization", default="none", choices=["none", "4bit", "8bit"])
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--gpu_memory", default="76GiB")
    parser.add_argument("--cpu_memory", default="96GiB")
    parser.add_argument("--progress_every", type=int, default=5)
    parser.add_argument("--disable_invalid_id_repair", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    model, tokenizer = load_model_and_tokenizer(args)

    done_ids = load_done_ids(output_path)
    print(f"Already completed tasks: {len(done_ids)}", flush=True)

    new_completed = 0
    seen = 0
    parse_failures = 0
    invalid_schema = 0

    for task in read_jsonl(input_path):
        seen += 1

        task_id = task.get("task_id")
        if task_id in done_ids:
            continue

        if args.max_tasks is not None and new_completed >= args.max_tasks:
            break

        row = process_task(model, tokenizer, task, args)
        append_jsonl(output_path, row)

        new_completed += 1

        if row.get("raw_parsed_output") is None:
            parse_failures += 1
        if not row.get("valid_json_schema"):
            invalid_schema += 1

        if new_completed % args.progress_every == 0:
            print(
                f"done={new_completed} seen={seen} "
                f"parse_fail={parse_failures} invalid={invalid_schema} "
                f"runtime_sec={round(sum_recent_runtime(output_path), 1)}",
                flush=True,
            )

        if new_completed % 25 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("Finished.", flush=True)
    print(f"New tasks completed: {new_completed}", flush=True)
    print(f"Parse failures: {parse_failures}", flush=True)
    print(f"Invalid schema: {invalid_schema}", flush=True)
    print(f"Output: {output_path.resolve()}", flush=True)


def sum_recent_runtime(output_path):
    total = 0.0
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    total += float(row.get("generation_seconds") or 0.0)
                except Exception:
                    continue
    except FileNotFoundError:
        return 0.0
    return total


if __name__ == "__main__":
    main()
