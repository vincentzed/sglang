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


PROMPT_FORMATS = {
    "gsm8k": {
        "repo": "openai/gsm8k",
        "config": "main",
        "split": "test",
        "field": "question",
        "template": (
            "{q}\n"
            "Please reason step by step, and put your final answer within \\boxed{{}}."
        ),
    },
    "math500": {
        "repo": "HuggingFaceH4/MATH-500",
        "config": None,
        "split": "test",
        "field": "problem",
        "template": (
            "{q}\n"
            "Please reason step by step, and put your final answer within \\boxed{{}}."
        ),
    },
}


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


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


def parse_indices(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    ret: list[int] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            lo_raw, hi_raw = piece.split("-", 1)
            lo = int(lo_raw)
            hi = int(hi_raw)
            if hi < lo:
                raise ValueError(f"bad index range {piece!r}")
            ret.extend(range(lo, hi + 1))
        else:
            ret.append(int(piece))
    return ret


def load_prompt_records(
    *,
    dataset: str,
    num_samples: int,
    tokenizer_path: str,
    sample_order: str,
    seed: int,
    indices: list[int] | None,
) -> list[dict[str, Any]]:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    spec = PROMPT_FORMATS[dataset]
    if spec["config"] is None:
        ds = load_dataset(spec["repo"], split=spec["split"])
    else:
        ds = load_dataset(spec["repo"], spec["config"], split=spec["split"])

    if sample_order == "shuffle":
        ds = ds.shuffle(seed=seed)
    if indices is None:
        selected = list(range(min(num_samples, len(ds))))
    else:
        selected = indices
        bad = [idx for idx in selected if idx < 0 or idx >= len(ds)]
        if bad:
            raise ValueError(f"indices out of range for {dataset}: {bad[:5]}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    records = []
    for sample_index in selected:
        row = ds[sample_index]
        raw_prompt = spec["template"].format(q=row[spec["field"]])
        chat_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": raw_prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        records.append(
            {
                "dataset": dataset,
                "sample_index": int(sample_index),
                "raw_prompt": raw_prompt,
                "prompt": chat_prompt,
            }
        )
    return records


def flush_cache(base_url: str) -> None:
    request_text("POST", f"{base_url}/flush_cache?timeout=60", timeout=120.0)


def compute_accept_length(meta_info: dict[str, Any]) -> float:
    completion_tokens = int(meta_info.get("completion_tokens") or 0)
    spec_verify_ct = int(meta_info.get("spec_verify_ct") or 0)
    if spec_verify_ct <= 0:
        return 1.0
    return completion_tokens / spec_verify_ct


def extract_dflash_tree_node_stats(
    server_info: dict[str, Any],
) -> dict[str, int | float] | None:
    for state in server_info.get("internal_states") or []:
        if "dflash_tree_build_ct" not in state:
            continue
        return {
            "build_ct": int(state.get("dflash_tree_build_ct") or 0),
            "num_nodes_total": int(state.get("dflash_tree_num_nodes_total") or 0),
            "avg_num_nodes": float(state.get("avg_dflash_tree_num_nodes") or 0.0),
        }
    return None


def diff_dflash_tree_node_stats(
    before: dict[str, int | float] | None,
    after: dict[str, int | float] | None,
) -> dict[str, int | float] | None:
    if before is None or after is None:
        return None
    build_ct = int(after["build_ct"]) - int(before["build_ct"])
    num_nodes_total = int(after["num_nodes_total"]) - int(before["num_nodes_total"])
    return {
        "build_ct": build_ct,
        "num_nodes_total": num_nodes_total,
        "mean_num_nodes": num_nodes_total / build_ct if build_ct > 0 else None,
    }


def load_baseline(path: Path | None) -> dict[tuple[str, int], list[int]]:
    if path is None:
        return {}
    data = json.loads(path.read_text())
    baseline: dict[tuple[str, int], list[int]] = {}
    for row in data.get("prompts", []):
        baseline[(row["dataset"], int(row["sample_index"]))] = row["output_ids"]
    return baseline


def run_one(
    *,
    base_url: str,
    record: dict[str, Any],
    max_new_tokens: int,
    request_timeout_s: float,
) -> dict[str, Any]:
    payload = {
        "text": record["prompt"],
        "sampling_params": {
            "temperature": 0,
            "top_p": 1.0,
            "max_new_tokens": max_new_tokens,
        },
        "return_logprob": False,
        "stream": False,
    }
    started = time.perf_counter()
    try:
        response = request_json(
            "POST",
            f"{base_url}/generate",
            payload=payload,
            timeout=request_timeout_s,
        )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"{record['dataset']}[{record['sample_index']}] failed: "
            f"HTTP {exc.code}: {body}"
        ) from exc
    wall_latency = time.perf_counter() - started
    meta_info = response.get("meta_info") or {}
    output_ids = response.get("output_ids") or []
    completion_tokens = int(meta_info.get("completion_tokens") or len(output_ids))
    server_e2e_latency = meta_info.get("e2e_latency")
    server_e2e_latency = (
        float(server_e2e_latency) if server_e2e_latency is not None else None
    )
    return {
        **record,
        "text": response.get("text"),
        "output_ids": output_ids,
        "completion_tokens": completion_tokens,
        "spec_verify_ct": int(meta_info.get("spec_verify_ct") or 0),
        "accept_length": compute_accept_length(meta_info),
        "wall_latency_s": wall_latency,
        "server_e2e_latency_s": server_e2e_latency,
        "wall_tok_s": completion_tokens / wall_latency if wall_latency else None,
        "server_e2e_tok_s": (
            completion_tokens / server_e2e_latency
            if server_e2e_latency
            else None
        ),
        "meta_info": meta_info,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_completion_tokens = sum(int(row["completion_tokens"]) for row in rows)
    total_spec_verify_ct = sum(int(row["spec_verify_ct"]) for row in rows)
    total_wall_latency = sum(float(row["wall_latency_s"]) for row in rows)
    server_latencies = [
        float(row["server_e2e_latency_s"])
        for row in rows
        if row["server_e2e_latency_s"] is not None
    ]
    total_server_e2e_latency = sum(server_latencies) if server_latencies else None
    accept_length = (
        total_completion_tokens / total_spec_verify_ct
        if total_spec_verify_ct > 0
        else 1.0
    )
    return {
        "num_prompts": len(rows),
        "total_completion_tokens": total_completion_tokens,
        "total_spec_verify_ct": total_spec_verify_ct,
        "accept_length": accept_length,
        "total_wall_latency_s": total_wall_latency,
        "throughput_wall_tok_s": (
            total_completion_tokens / total_wall_latency
            if total_wall_latency
            else None
        ),
        "total_server_e2e_latency_s": total_server_e2e_latency,
        "throughput_server_e2e_tok_s": (
            total_completion_tokens / total_server_e2e_latency
            if total_server_e2e_latency
            else None
        ),
        "mean_per_prompt_accept_length": (
            sum(float(row["accept_length"]) for row in rows) / len(rows)
            if rows
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--mode", required=True, choices=["ar", "linear", "tree"])
    parser.add_argument("--dataset", required=True, choices=sorted(PROMPT_FORMATS))
    parser.add_argument("--num-samples", type=int, default=80)
    parser.add_argument("--indices", type=parse_indices)
    parser.add_argument("--sample-order", default="first", choices=["first", "shuffle"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--draft-model", default="JetSpec/jetspec-qwen3-8b")
    parser.add_argument("--tokenizer-path", default="Qwen/Qwen3-8B")
    parser.add_argument("--tree-width", type=int, default=1)
    parser.add_argument("--tree-budget", type=int)
    parser.add_argument("--tree-draft", default="accum_logp")
    parser.add_argument("--top2gap-beta", type=float)
    parser.add_argument("--top2gap-g0", type=float)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--warmup-samples", type=int, default=2)
    parser.add_argument("--compare-to", type=Path)
    parser.add_argument("--flush-cache-before-run", action="store_true")
    parser.add_argument("--flush-cache-between-prompts", action="store_true")
    parser.add_argument("--health-timeout-s", type=float, default=1800.0)
    parser.add_argument("--request-timeout-s", type=float, default=1800.0)
    args = parser.parse_args()

    wait_health(args.base_url, args.health_timeout_s)
    records = load_prompt_records(
        dataset=args.dataset,
        num_samples=args.num_samples,
        tokenizer_path=args.tokenizer_path,
        sample_order=args.sample_order,
        seed=args.seed,
        indices=args.indices,
    )
    if args.flush_cache_before_run:
        flush_cache(args.base_url)

    warmup_records = records[: max(0, min(args.warmup_samples, len(records)))]
    for record in warmup_records:
        run_one(
            base_url=args.base_url,
            record=record,
            max_new_tokens=args.max_new_tokens,
            request_timeout_s=args.request_timeout_s,
        )
    if args.flush_cache_before_run:
        flush_cache(args.base_url)

    try:
        server_info_before = request_json(
            "GET", f"{args.base_url}/server_info", timeout=30.0
        )
    except Exception as exc:  # noqa: BLE001
        server_info_before = {"error": str(exc)}

    baseline = load_baseline(args.compare_to)
    rows = []
    mismatches = []
    for index, record in enumerate(records):
        if args.flush_cache_between_prompts:
            flush_cache(args.base_url)
        row = run_one(
            base_url=args.base_url,
            record=record,
            max_new_tokens=args.max_new_tokens,
            request_timeout_s=args.request_timeout_s,
        )
        expected = baseline.get((row["dataset"], int(row["sample_index"])))
        token_exact = expected is None or row["output_ids"] == expected
        row["token_exact_vs_baseline"] = token_exact
        if not token_exact:
            first_diff = next(
                (
                    pos
                    for pos, (actual, exp) in enumerate(zip(row["output_ids"], expected))
                    if actual != exp
                ),
                min(len(row["output_ids"]), len(expected)),
            )
            mismatches.append(
                {
                    "row_index": index,
                    "dataset": row["dataset"],
                    "sample_index": row["sample_index"],
                    "first_diff": first_diff,
                    "actual_len": len(row["output_ids"]),
                    "expected_len": len(expected),
                    "actual_window": row["output_ids"][
                        max(0, first_diff - 5) : first_diff + 5
                    ],
                    "expected_window": expected[
                        max(0, first_diff - 5) : first_diff + 5
                    ],
                }
            )
        rows.append(row)

    try:
        server_info = request_json("GET", f"{args.base_url}/server_info", timeout=30.0)
    except Exception as exc:  # noqa: BLE001
        server_info = {"error": str(exc)}

    result = {
        "status": "ok" if not mismatches else "mismatch",
        "created_at": utc_now(),
        "run_name": args.run_name,
        "base_url": args.base_url,
        "mode": args.mode,
        "dataset": args.dataset,
        "sample_order": args.sample_order,
        "seed": args.seed,
        "models": {
            "target": args.target_model,
            "draft": args.draft_model if args.mode != "ar" else None,
            "tokenizer": args.tokenizer_path,
        },
        "decode": {
            "temperature": 0,
            "top_p": 1.0,
            "max_new_tokens": args.max_new_tokens,
            "attention_backend": "fa4",
            "page_size": 16,
            "tree_width": args.tree_width,
            "tree_budget": args.tree_budget,
            "tree_draft": args.tree_draft,
            "top2gap_beta": args.top2gap_beta,
            "top2gap_g0": args.top2gap_g0,
        },
        "warmup": {
            "num_prompts": len(warmup_records),
            "sample_indices": [row["sample_index"] for row in warmup_records],
        },
        "summary": summarize(rows),
        "tree_node_stats": diff_dflash_tree_node_stats(
            extract_dflash_tree_node_stats(server_info_before),
            extract_dflash_tree_node_stats(server_info),
        ),
        "losslessness": {
            "compared_to": str(args.compare_to) if args.compare_to else None,
            "token_exact": not mismatches,
            "mismatches": mismatches,
        },
        "prompts": rows,
        "server_info_before": server_info_before,
        "server_info": server_info,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    if mismatches:
        print(json.dumps({"mismatches": mismatches[:5]}, indent=2, sort_keys=True))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
