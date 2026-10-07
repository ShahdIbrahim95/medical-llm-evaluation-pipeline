import argparse
from pathlib import Path

from src.clients.factory import build_model_client, resolve_backend
from src.config import (
    DATA_DIR,
    DEFAULT_GRADER,
    DEFAULT_MAX_WORKERS,
    add_model_argument,
    model_alias,
    responses_path,
    resolve_model,
    scores_path,
)
from src.jsonl_io import load_jsonl
from src.healthbench.loader import HealthBenchLoader
from src.healthbench.grader import HealthBenchGrader
from src.evaluation_runner import EvaluationRunner

DEFAULT_DATA = f"{DATA_DIR}/healthbench_subset_50.jsonl"
DEFAULT_LIMIT = 5


def _load_responses(responses_file: str):
    records = load_jsonl(Path(responses_file))
    if not records:
        raise ValueError(f"No responses found in {responses_file}")

    rows = list(records.values())
    response_model = model_alias(rows[0]["model"])
    if any(model_alias(row["model"]) != response_model for row in rows):
        raise ValueError(
            f"All rows in {responses_file} must use the same response model"
        )

    return {
        prompt_id: row["response"]
        for prompt_id, row in records.items()
    }, response_model


def run_grade_responses(
    responses_file=None,
    output_path=None,
    data_path=DEFAULT_DATA,
    grader_model=DEFAULT_GRADER,
    limit=DEFAULT_LIMIT,
    response_model=None,
    max_workers=DEFAULT_MAX_WORKERS,
    resume=True,
    backend=None,
    server_url=None,
):
    grader_alias = model_alias(grader_model)
    grader_model_id = resolve_model(grader_model)
    backend = resolve_backend(grader_model, backend)

    if responses_file is None:
        source_model = response_model or grader_alias
        responses_file = responses_path(limit, source_model)

    responses_by_id, response_alias = _load_responses(responses_file)

    loader = HealthBenchLoader(path=data_path)
    examples = [
        ex for ex in loader.load_examples()
        if ex["prompt_id"] in responses_by_id
    ]
    if not examples:
        raise ValueError(
            f"No matching examples in {data_path} for prompts in {responses_file}"
        )

    responses = [responses_by_id[ex["prompt_id"]] for ex in examples]

    if output_path is None:
        output_path = scores_path(len(examples), response_alias, grader_alias)

    grader_client = build_model_client(
        grader_model,
        backend=backend,
        server_url=server_url,
        temperature=0.0,
        max_concurrent_requests=max_workers,
    )
    grader = HealthBenchGrader(grader_client)

    print(f"Responses: {responses_file}")
    print(
        f"Grading {len(examples)} examples "
        f"(response model: {response_alias}, grader: {grader_alias}, "
        f"backend: {backend}, {max_workers} concurrent rubric calls)"
    )

    output_file = EvaluationRunner(
        grader=grader,
        examples=examples,
        responses=responses,
        max_workers=max_workers,
        resume=resume,
    ).run(output_path=output_path)

    print(f"Saved scores: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Grade model responses against HealthBench rubrics.",
    )
    parser.add_argument(
        "--responses",
        default=None,
        help=(
            "JSONL from a benchmark run "
            f"(default: outputs/{{limit}}_{{response_model}}_responses.jsonl)"
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output JSONL path "
            "(default: outputs/{count}_{response_model}_{grader}_scores.jsonl)"
        ),
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA,
        help=f"HealthBench JSONL subset (default: {DEFAULT_DATA})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=(
            "Example count used to build default response path when "
            f"--responses is omitted (default: {DEFAULT_LIMIT})"
        ),
    )
    parser.add_argument(
        "--response-model",
        default=None,
        choices=["ministral", "gemma", "qwen"],
        help=(
            "Response model alias for default paths when --responses is "
            "omitted (default: same as --model)"
        ),
    )
    add_model_argument(parser, default=DEFAULT_GRADER)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=(
            f"Concurrent rubric API calls per example "
            f"(default: {DEFAULT_MAX_WORKERS})"
        ),
    )
    parser.add_argument(
        "--backend",
        choices=["morpheus", "vllm"],
        default=None,
        help="Inference backend for the grader (default: auto from model alias)",
    )
    parser.add_argument(
        "--server-url",
        default=None,
        help="vLLM OpenAI-compatible base URL (default: VLLM_SERVER_URL)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh instead of resuming from existing output files",
    )
    args = parser.parse_args()

    run_grade_responses(
        responses_file=args.responses,
        output_path=args.output,
        data_path=args.data,
        grader_model=args.model,
        limit=args.limit,
        response_model=args.response_model,
        max_workers=args.workers,
        resume=not args.no_resume,
        backend=args.backend,
        server_url=args.server_url,
    )


if __name__ == "__main__":
    main()
