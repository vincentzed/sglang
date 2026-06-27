#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FIXED_PROMPTS = [
    "Solve step by step: If 3x + 7 = 31, what is x?",
    "A store sells notebooks for $4 each. Mira buys 6 notebooks and pays with $30. How much change does she get?",
    "Write a Python function that returns the first n Fibonacci numbers.",
    "Explain why the sky appears blue in two concise paragraphs.",
    "A train travels 180 miles in 3 hours. What is its average speed in miles per hour?",
    "Translate to French: The meeting starts at nine tomorrow morning.",
    "Classify the sentiment of this sentence as positive, neutral, or negative: The package arrived late but the product works well.",
    "Find the next number in the sequence and explain: 2, 6, 12, 20, 30, ...",
    "Summarize the following in one sentence: Tree speculative decoding verifies multiple candidate continuations in one target-model pass.",
    "Given a right triangle with legs 5 and 12, compute the hypotenuse.",
]


def request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["content-type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def request_text(method: str, url: str, timeout: float = 120.0) -> str:
    req = urllib.request.Request(url, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def wait_health(base_url: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_error = ""
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{base_url}/health", timeout=5.0).read()
            return
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            time.sleep(5.0)
    raise TimeoutError(f"server did not become healthy in {timeout_s}s: {last_error}")


def compute_accept_length(meta_info: dict[str, Any]) -> float:
    completion_tokens = int(meta_info.get("completion_tokens") or 0)
    spec_verify_ct = int(meta_info.get("spec_verify_ct") or 0)
    if spec_verify_ct <= 0:
        return 1.0
    return completion_tokens / spec_verify_ct


def load_baseline(path: Path | None) -> dict[str, list[int]]:
    if path is None:
        return {}
    data = json.loads(path.read_text())
    prompts = data["prompts"]
    return {row["prompt"]: row["output_ids"] for row in prompts}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_completion_tokens = sum(row["completion_tokens"] for row in rows)
    total_e2e_latency = sum(row["e2e_latency"] for row in rows)
    accept_lengths = [row["accept_length"] for row in rows]
    tok_per_s = [row["tok_per_s"] for row in rows if row["tok_per_s"] is not None]
    return {
        "num_prompts": len(rows),
        "total_completion_tokens": total_completion_tokens,
        "total_e2e_latency": total_e2e_latency,
        "aggregate_tok_per_s": (
            total_completion_tokens / total_e2e_latency if total_e2e_latency else None
        ),
        "mean_per_prompt_tok_per_s": (
            sum(tok_per_s) / len(tok_per_s) if tok_per_s else None
        ),
        "mean_accept_length": (
            sum(accept_lengths) / len(accept_lengths) if accept_lengths else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--tree-width", required=True, type=int)
    parser.add_argument("--tree-budget", required=True, type=int)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--compare-to", type=Path)
    parser.add_argument("--attention-backend", default="fa3")
    parser.add_argument("--prompt-index", type=int)
    parser.add_argument("--flush-cache-before-run", action="store_true")
    parser.add_argument("--flush-cache-between-prompts", action="store_true")
    parser.add_argument("--health-timeout-s", type=float, default=1800.0)
    parser.add_argument("--request-timeout-s", type=float, default=600.0)
    args = parser.parse_args()

    wait_health(args.base_url, args.health_timeout_s)
    if args.flush_cache_before_run:
        request_text("POST", f"{args.base_url}/flush_cache?timeout=60", timeout=120.0)
    baseline = load_baseline(args.compare_to)

    rows = []
    mismatches = []
    prompt_items = list(enumerate(FIXED_PROMPTS))
    if args.prompt_index is not None:
        prompt_items = [
            item for item in prompt_items if item[0] == int(args.prompt_index)
        ]
        if not prompt_items:
            raise ValueError(f"prompt index out of range: {args.prompt_index}")
    for index, prompt in prompt_items:
        if args.flush_cache_between_prompts:
            request_text("POST", f"{args.base_url}/flush_cache?timeout=60", timeout=120.0)
        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": args.max_new_tokens,
            },
            "return_logprob": False,
            "stream": False,
        }
        started = time.perf_counter()
        try:
            response = request_json(
                "POST",
                f"{args.base_url}/generate",
                payload=payload,
                timeout=args.request_timeout_s,
            )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"request {index} failed: HTTP {exc.code}: {body}") from exc
        wall_latency = time.perf_counter() - started

        meta_info = response.get("meta_info") or {}
        completion_tokens = int(meta_info.get("completion_tokens") or 0)
        e2e_latency = float(meta_info.get("e2e_latency") or wall_latency)
        output_ids = response.get("output_ids") or []
        accept_length = compute_accept_length(meta_info)
        expected_output_ids = baseline.get(prompt)
        token_exact = expected_output_ids is None or output_ids == expected_output_ids
        if not token_exact:
            first_diff = next(
                (
                    pos
                    for pos, (actual, expected) in enumerate(
                        zip(output_ids, expected_output_ids)
                    )
                    if actual != expected
                ),
                min(len(output_ids), len(expected_output_ids)),
            )
            mismatches.append(
                {
                    "prompt_index": index,
                    "prompt": prompt,
                    "first_diff": first_diff,
                    "actual_len": len(output_ids),
                    "expected_len": len(expected_output_ids),
                    "actual_window": output_ids[max(0, first_diff - 5) : first_diff + 5],
                    "expected_window": expected_output_ids[
                        max(0, first_diff - 5) : first_diff + 5
                    ],
                }
            )

        rows.append(
            {
                "prompt_index": index,
                "prompt": prompt,
                "text": response.get("text"),
                "output_ids": output_ids,
                "completion_tokens": completion_tokens,
                "e2e_latency": e2e_latency,
                "wall_latency": wall_latency,
                "tok_per_s": (
                    completion_tokens / e2e_latency if e2e_latency else None
                ),
                "accept_length": accept_length,
                "spec_verify_ct": int(meta_info.get("spec_verify_ct") or 0),
                "meta_info": meta_info,
                "token_exact_vs_baseline": token_exact,
            }
        )

    try:
        server_info = request_json("GET", f"{args.base_url}/server_info", timeout=30.0)
    except Exception as exc:  # noqa: BLE001
        server_info = {"error": str(exc)}

    result = {
        "status": "ok" if not mismatches else "mismatch",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_name": args.run_name,
        "base_url": args.base_url,
        "models": {
            "target": args.target_model,
            "draft": args.draft_model,
        },
        "decode": {
            "dtype": "bfloat16",
            "temperature": 0,
            "max_new_tokens": args.max_new_tokens,
            "attention_backend": args.attention_backend,
            "tp_size": 1,
            "tree_width": args.tree_width,
            "tree_budget": args.tree_budget,
        },
        "summary": summarize(rows),
        "losslessness": {
            "compared_to": str(args.compare_to) if args.compare_to else None,
            "token_exact": not mismatches,
            "mismatches": mismatches,
        },
        "prompts": rows,
        "server_info": server_info,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    if mismatches:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
