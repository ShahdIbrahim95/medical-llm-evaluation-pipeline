import argparse
import os
from pathlib import Path

from src.benchmark_runner import BenchmarkRunner
from src.clients.api_client import MorpheusClient
from src.clients.vllm_client import VLLMClient
from src.config import (
    APIConfig,
    DATA_DIR,
    DEFAULT_MAX_WORKERS,
    ModelConfig,
    OUTPUT_DIR,
    add_model_argument,
    responses_path,
    resolve_model,
)
from src.healthbench.loader import HealthBenchLoader
from src.prompts.capo import apply_optimized_prompt, load_prompt_spec


def _safe_model_name(model_name: str) -> str:
    """
    Convert a model name into a safe string that can be used in file names.
    """
    return (
        model_name
        .replace("/", "_")
        .replace(":", "_")
        .replace(" ", "_")
        .replace(".", "_")
    )


def _build_prompt_transform(prompt_spec_path: str | None):
    if prompt_spec_path is None:
        return None

    spec = load_prompt_spec(prompt_spec_path)

    def transform(messages, example):
        del example
        return apply_optimized_prompt(
            messages,
            instruction_text=spec["instruction_text"],
            few_shots=spec.get("few_shots"),
        )

    return transform


def run_generate_responses(
    model="ministral",
    limit=5,
    data_path="data/healthbench_subset_50.jsonl",
    output_path=None,
    resume=True,
    technique=None,
    prompt_spec_path=None,
    max_workers=DEFAULT_MAX_WORKERS,
):
    """
    Generate model responses for HealthBench examples and save them to JSONL.
    """
    model_name = resolve_model(model)
    if output_path is None:
        output_path = responses_path(limit, model, technique=technique)
    api_config = APIConfig()
    model_config = ModelConfig(model_name=model_name, temperature=0.7)

    print(f"Running experiment with model: {model_name}")
    if technique:
        print(f"Technique: {technique}")
    if prompt_spec_path:
        print(f"Using optimized prompt: {prompt_spec_path}")

    client = MorpheusClient(
        api_config=api_config,
        model_config=model_config,
    )

    loader = HealthBenchLoader(path=data_path)
    examples = loader.load_examples(limit=limit)

    runner = BenchmarkRunner(
        model_client=client,
        examples=examples,
        max_workers=max_workers,
        prompt_transform=_build_prompt_transform(prompt_spec_path),
    )

    runner.run(output_path=output_path, resume=resume)


def generate_responses(
    model_name: str,
    output_path: str | None = None,
    dataset_path: str = f"{DATA_DIR}/healthbench_full.jsonl",
    limit: int | None = None,
    max_workers: int = 5,
    temperature: float = 0.7,
    backend: str = "morpheus",
    server_url: str | None = None,
    request_timeout: int = 300,
    max_tokens: int = 512,
    prompt_spec_path: str | None = None,
    resume: bool = True,
):
    """
    Generate model responses for HealthBench examples.

    Supports both remote Morpheus models and locally hosted vLLM models.
    Responses are written incrementally to a JSONL file, allowing long runs
    to resume safely if interrupted.
    """
    backend = backend.lower()

    if backend == "vllm":
        served_name = os.getenv("MODEL_ALIAS", model_name)
        model_config = ModelConfig(
            model_name=served_name,
            temperature=temperature,
        )
    else:
        model_config = ModelConfig(
            model_name=resolve_model(model_name),
            temperature=temperature,
        )

    print(f"Running response generation with model: {model_config.model_name}")
    print(f"Backend: {backend}")
    if prompt_spec_path:
        print(f"Using optimized prompt: {prompt_spec_path}")

    if backend == "morpheus":
        client = MorpheusClient(
            api_config=APIConfig(),
            model_config=model_config,
            timeout=request_timeout,
        )

    elif backend == "vllm":
        if server_url is None:
            raise ValueError("server_url is required when backend='vllm'")

        client = VLLMClient(
            base_url=server_url,
            model_config=model_config,
            timeout=request_timeout,
            max_tokens=max_tokens,
        )

    else:
        raise ValueError("backend must be either 'morpheus' or 'vllm'")

    loader = HealthBenchLoader(path=dataset_path)
    examples = loader.load_examples(limit=limit)

    print(f"Loaded {len(examples)} examples from {dataset_path}")

    if output_path is None:
        output_path = responses_path(len(examples), model_name)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"Responses will be saved to: {output_path}")
    print(f"Parallel workers: {max_workers}")

    runner = BenchmarkRunner(
        model_client=client,
        examples=examples,
        max_workers=max_workers,
        prompt_transform=_build_prompt_transform(prompt_spec_path),
    )

    runner.run(output_path=output_path, resume=resume)

    print(f"Response generation finished. Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate model responses for HealthBench examples.",
    )
    parser.add_argument(
        "--data",
        default=f"{DATA_DIR}/healthbench_full.jsonl",
        help=f"HealthBench JSONL dataset (default: {DATA_DIR}/healthbench_full.jsonl)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of examples to run (default: full dataset)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path (default: outputs/{count}_{model}_responses.jsonl)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Concurrent generation workers (default: {DEFAULT_MAX_WORKERS})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--backend",
        choices=["morpheus", "vllm"],
        default="morpheus",
        help="Inference backend (default: morpheus)",
    )
    parser.add_argument(
        "--server-url",
        default=None,
        help="vLLM OpenAI-compatible base URL (default: VLLM_SERVER_URL or LLM_API_BASE_URL)",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Optimized prompt JSONL from CAPO (best_prompt.jsonl)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh instead of resuming from existing output files",
    )
    add_model_argument(parser)
    args = parser.parse_args()

    server_url = (
        args.server_url
        or os.getenv("VLLM_SERVER_URL")
        or os.getenv("LLM_API_BASE_URL")
    )

    generate_responses(
        model_name=args.model,
        dataset_path=args.data,
        limit=args.limit,
        output_path=args.output,
        max_workers=args.workers,
        temperature=args.temperature,
        backend=args.backend,
        server_url=server_url,
        prompt_spec_path=args.prompt_file,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
