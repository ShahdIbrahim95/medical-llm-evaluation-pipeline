# Medical LLM Evaluation & Benchmarking Pipeline

An automated benchmarking and evaluation framework designed for assessing Large Language Models (LLMs) on medical question-answering benchmarks (e.g., HealthBench).

## Architecture & Features

- **Response Generation (`generate_responses.py`)**: Supports both remote inference APIs and high-throughput local serving via **vLLM** with concurrency management and automated resume functionality.
- **LLM-as-a-Judge Grading (`grade_responses.py`)**: Deterministic rubric evaluation pipeline using temperature-zero multi-model grading to quantify clinical reasoning stability.
- **HPC Orchestration (`slurm/`)**: Automated Slurm job submission scripts (`run_vllm.sbatch` and `run_api.sbatch`) tailored for NVIDIA A40 GPUs, featuring health-check polling loops and automatic server teardown hooks.

## Usage Overview

```bash
# Submit automated local vLLM generation job on Slurm
sbatch slurm/run_vllm.sbatch

# Run rubric-based grading across generated outputs
python grade_responses.py --data data/healthbench_subset_50.jsonl --limit 50
