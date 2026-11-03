from datasets import load_dataset
from pathlib import Path

out_dir = Path("raw/memoryagentbench")
out_dir.mkdir(parents=True, exist_ok=True)

dataset_name = "ai-hyz/MemoryAgentBench"

splits = [
    "Accurate_Retrieval",
    "Test_Time_Learning",
    "Long_Range_Understanding",
    "Conflict_Resolution",
]

for split in splits:
    try:
        ds = load_dataset(dataset_name, split=split)
        path = out_dir / f"{split}.jsonl"
        ds.to_json(path)
        print(f"Saved {split}: {len(ds)} rows -> {path}")
    except Exception as e:
        print(f"Could not load split {split}: {e}")
