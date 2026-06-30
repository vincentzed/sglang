#!/usr/bin/env python3
"""Run the DFlash top2gap construction sweep on paper datasets."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TARGET_MODEL = "Qwen/Qwen3-8B"
DRAFT_MODEL = "JetSpec/jetspec-qwen3-8b"
LINEAR_BASELINE_TOK_S = {"gsm8k": 1152.8517874411418, "math500": 1505.24179546956}
LINEAR_COMPARE_TO = {
    "gsm8k": ROOT / "jetspec/runs/final_gsm8k_linear_31961.json",
    "math500": ROOT / "jetspec/runs/final_math500_linear_31961.json",
}


@dataclass(frozen=True)
class SweepConfig:
    algo: str
    width: int
    budget: int
    beta: float | None = None
    g0: float | None = None

    @property
    def slug(self) -> str:
        parts = [self.algo, f"w{self.width}", f"b{self.budget}"]
        if self.algo == "top2gap":
            parts.extend([f"beta{format_float(self.beta)}", f"g0{format_float(self.g0)}"])
        return "_".join(parts)


def format_float(value: float | None) -> str:
    if value is None:
        return "na"
    return str(value).replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="jetspec/runs/top2gap_lean_20260629")
    parser.add_argument("--port-base", type=int, default=31700)
    parser.add_argument("--num-samples", type=int, default=80)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--health-timeout-s", type=int, default=1800)
    parser.add_argument("--request-timeout-s", type=int, default=900)
    parser.add_argument("--stop-on-beat", action="store_true", default=True)
    parser.add_argument("--no-stop-on-beat", action="store_false", dest="stop_on_beat")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit-configs", type=int)
    parser.add_argument("--only-algo", choices=("accum_logp", "top2gap"))
    return parser.parse_args()


def build_configs() -> list[SweepConfig]:
    configs: list[SweepConfig] = []
    for beta, g0 in [(1.0, 1.0), (2.0, 0.5)]:
        for budget in [24, 16, 32, 48]:
            configs.append(
                SweepConfig(
                    algo="top2gap",
                    width=8,
                    budget=budget,
                    beta=beta,
                    g0=g0,
                )
            )
    return configs


def wait_health(base_url: str, process: subprocess.Popen[bytes], timeout_s: int, log_path: Path) -> None:
    deadline = time.monotonic() + timeout_s
    health_url = f"{base_url}/health"
    last_error: str | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = tail_text(log_path)
            raise RuntimeError(
                f"server exited before health, returncode={process.returncode}, "
                f"last_error={last_error}\n{tail}"
            )
        try:
            with urllib.request.urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    return
                last_error = f"HTTP {response.status}"
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = str(exc)
        time.sleep(1)
    raise TimeoutError(f"timed out waiting for {health_url}; last_error={last_error}\n{tail_text(log_path)}")


def tail_text(path: Path, max_bytes: int = 12000) -> str:
    if not path.exists():
        return f"<missing log {path}>"
    data = path.read_bytes()
    return data[-max_bytes:].decode("utf-8", errors="replace")


def launch_server(config: SweepConfig, port: int, log_path: Path) -> subprocess.Popen[bytes]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "7"
    env["SGLANG_ENABLE_OVERLAP_PLAN_STREAM"] = "1"
    env["PYTHONPATH"] = "python"
    env["HUGGING_FACE_HUB_TOKEN"] = env.get("HF_TOKEN", "")

    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        TARGET_MODEL,
        "--dtype",
        "bfloat16",
        "--speculative-algorithm",
        "DFLASH",
        "--speculative-draft-model-path",
        DRAFT_MODEL,
        "--speculative-num-draft-tokens",
        "16",
        "--speculative-dflash-tree-width",
        str(config.width),
        "--speculative-dflash-tree-budget",
        str(config.budget),
        "--speculative-dflash-tree-draft",
        config.algo,
        "--reasoning-parser",
        "qwen3",
        "--attention-backend",
        "fa4",
        "--page-size",
        "16",
        "--tp-size",
        "1",
        "--mem-fraction-static",
        "0.8",
        "--trust-remote-code",
        "--max-running-requests",
        "1",
        "--cuda-graph-max-bs-decode",
        "1",
        "--cuda-graph-backend-decode",
        "full",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
    ]
    if config.algo == "top2gap":
        cmd.extend(
            [
                "--speculative-dflash-top2gap-beta",
                str(config.beta),
                "--speculative-dflash-top2gap-g0",
                str(config.g0),
            ]
        )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("wb")
    process = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_file.close()
    return process


def stop_server(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=10)


def run_dataset(
    config: SweepConfig,
    dataset: str,
    base_url: str,
    out_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "jetspec/bench_paper_sglang.py",
        "--base-url",
        base_url,
        "--out",
        str(out_path),
        "--run-name",
        f"top2gap-sweep-{dataset}-{config.slug}",
        "--mode",
        "tree",
        "--dataset",
        dataset,
        "--num-samples",
        str(args.num_samples),
        "--target-model",
        TARGET_MODEL,
        "--draft-model",
        DRAFT_MODEL,
        "--tree-width",
        str(config.width),
        "--tree-budget",
        str(config.budget),
        "--tree-draft",
        config.algo,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--compare-to",
        str(LINEAR_COMPARE_TO[dataset]),
        "--flush-cache-before-run",
        "--health-timeout-s",
        str(args.health_timeout_s),
        "--request-timeout-s",
        str(args.request_timeout_s),
    ]
    if config.algo == "top2gap":
        cmd.extend(["--top2gap-beta", str(config.beta), "--top2gap-g0", str(config.g0)])

    subprocess.run(cmd, cwd=ROOT, check=True)
    return json.loads(out_path.read_text())


def result_row(config: SweepConfig, dataset: str, port: int, result_path: Path, result: dict[str, Any]) -> dict[str, Any]:
    summary = result["summary"]
    total_steps = summary["total_spec_verify_ct"]
    wall_s = summary["total_wall_latency_s"]
    tok_s = summary["throughput_wall_tok_s"]
    loss = result.get("losslessness") or {}
    tree_node_stats = result.get("tree_node_stats") or {}
    return {
        "accept_len": summary["accept_length"],
        "algo": config.algo,
        "beta": config.beta,
        "budget": config.budget,
        "dataset": dataset,
        "g0": config.g0,
        "lossless": bool(loss.get("token_exact", False)),
        "mean_tree_nodes": tree_node_stats.get("mean_num_nodes"),
        "mismatches": len(loss.get("mismatches") or []),
        "ms_per_step": wall_s / total_steps * 1000.0,
        "num_prompts": summary["num_prompts"],
        "port": port,
        "result_path": str(result_path),
        "steps_per_s": total_steps / wall_s,
        "tok_s": tok_s,
        "vs_linear": tok_s / LINEAR_BASELINE_TOK_S[dataset],
        "width": config.width,
    }


def write_summary_rows(summary_path: Path, rows: list[dict[str, Any]]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_existing_rows(
    configs: list[SweepConfig], out_dir: Path, port_base: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, config in enumerate(configs):
        port = port_base + index
        for dataset in ["gsm8k", "math500"]:
            result_path = out_dir / f"{dataset}_{config.slug}_{port}.json"
            if not result_path.exists():
                continue
            result = json.loads(result_path.read_text())
            rows.append(result_row(config, dataset, port, result_path, result))
    return rows


def main() -> None:
    args = parse_args()
    out_dir = ROOT / args.out_dir
    log_dir = ROOT / "jetspec/logs" / out_dir.name
    summary_path = out_dir / "summary.ndjson"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    configs = build_configs()
    if args.only_algo is not None:
        configs = [config for config in configs if config.algo == args.only_algo]
    if args.limit_configs is not None:
        configs = configs[: args.limit_configs]

    rows = load_existing_rows(configs, out_dir, args.port_base)
    write_summary_rows(summary_path, rows)

    print(f"sweep_configs={len(configs)} existing_rows={len(rows)} out_dir={out_dir}", flush=True)
    for index, config in enumerate(configs):
        port = args.port_base + index
        expected_paths = {
            dataset: out_dir / f"{dataset}_{config.slug}_{port}.json"
            for dataset in ["gsm8k", "math500"]
        }
        if not args.force and all(path.exists() for path in expected_paths.values()):
            print(f"skip_existing config={index + 1}/{len(configs)} {config.slug}", flush=True)
            continue

        base_url = f"http://127.0.0.1:{port}"
        log_path = log_dir / f"{config.slug}_{port}_server.log"
        print(f"launch config={index + 1}/{len(configs)} port={port} {config}", flush=True)
        process = launch_server(config, port, log_path)
        try:
            wait_health(base_url, process, args.health_timeout_s, log_path)
            print(f"ready port={port} {config.slug}", flush=True)
            config_rows: list[dict[str, Any]] = []
            for dataset, result_path in expected_paths.items():
                if not args.force and result_path.exists():
                    result = json.loads(result_path.read_text())
                else:
                    print(f"run dataset={dataset} {config.slug}", flush=True)
                    result = run_dataset(config, dataset, base_url, result_path, args)
                row = result_row(config, dataset, port, result_path, result)
                config_rows.append(row)
                rows = [old for old in rows if old["result_path"] != row["result_path"]]
                rows.append(row)
                write_summary_rows(summary_path, rows)
                print(
                    "done "
                    f"dataset={dataset} tok_s={row['tok_s']:.2f} "
                    f"accept={row['accept_len']:.2f} nodes={row['mean_tree_nodes']} "
                    f"lossless={row['lossless']} vs_linear={row['vs_linear']:.2f}",
                    flush=True,
                )

            if args.stop_on_beat and all(
                row["tok_s"] >= LINEAR_BASELINE_TOK_S[row["dataset"]] for row in config_rows
            ):
                print(f"beat_linear config={config.slug}", flush=True)
                return
        finally:
            stop_server(process)
            print(f"stopped port={port} {config.slug}", flush=True)

    print(f"sweep_complete rows={len(rows)} summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
