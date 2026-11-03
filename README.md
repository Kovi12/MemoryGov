# MemoryGov

Long-term memory governance for language-model agents: deciding when information should be written, ignored, updated, merged, forgotten or quarantined, and which active memories should enter an answer context.

## Components

- `memory_governor_cli_v4_10.py`: current interactive lifecycle runtime with SQLite storage, lexical/embedding retrieval, policy overrides and target confirmation.
- `memory_governor_cli_v4_9.py`: preceding runtime retained for benchmark compatibility.
- `memory_governor_web_full.py` / `memory_governor_web_final.py`: web experiments using their matching v4.2 runtime.
- `demos/`: governed answer generation and structured memory-event decomposition.
- `operation_llm_train/`, `operation_transformer_train/`: learned operation governors.
- `hard_ood_benchmark/`, `longturn_benchmark/`, `external_baselines/`, `public_benchmarks/`: benchmark preparation, adapters and evaluation.

## Setup

Use Python 3.10+ and install `requirements.txt` in a dedicated environment. Install a CUDA-compatible PyTorch build for GPU experiments; quantized training additionally needs bitsandbytes. Configure model/data paths in the selected script and provide your own model access through environment variables such as `HF_TOKEN`.

The wider classifier/framework comparisons use optional CatBoost, LightGBM, XGBoost and Mem0 adapters; install `requirements-experiments.txt` for those experiments. Backend credentials and model artifacts are configured separately. Optional integrations are not required for the basic SQLite/lexical runtime.

The runtime expects a trained classifier at its configured model path. Generate the required model/data artifacts using the preparation and training scripts before starting the CLI. From this directory, use `python memory_governor_cli_v4_10.py --help` for runtime options. Web and benchmark scripts retain version-specific model/artifact requirements; use the matching runtime stated above.

## Evaluation

`results/operation_classifier_validation.json` records Qwen2.5-14B LoRA operation-classification results: 98.72% accuracy and 96.36% macro F1 on 1,091 validation examples. This evaluates the operation classifier, not end-to-end QA. Lifecycle, synthetic robustness and public QA experiments are separate evaluations.

FORGET deactivates a stored memory for retrieval; it is not a physical-erasure guarantee. SQLite stores and experiment logs can retain original content. The web demos are intended for local research use. Do not use real secrets or private conversations as demonstration data.

## Research and dependencies

Research project by Ovidiu Catalin Ghibea in the POLITEHNICA Bucharest context. LoCoMo, LongMemEval, MemoryAgentBench and EvoMemBench are external benchmarks with their own authors and terms. Framework-style local controls are not implementations of commercial memory products. Obtain datasets/models from their original providers; raw conversations, model weights and generated stores are not included.
