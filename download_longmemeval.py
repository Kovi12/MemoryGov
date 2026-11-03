from huggingface_hub import hf_hub_download
from pathlib import Path

out_dir = Path("raw/longmemeval")
out_dir.mkdir(parents=True, exist_ok=True)

repo_id = "xiaowu0162/longmemeval-cleaned"
files = [
    # "longmemeval_oracle.json",
    # "longmemeval_s_cleaned.json",
    "longmemeval_m_cleaned.json"
]

for file_name in files:
    path = hf_hub_download(
        repo_id=repo_id,
        filename=file_name,
        repo_type="dataset",
        local_dir=out_dir,
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded: {path}")
