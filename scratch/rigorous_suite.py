"""VALUE: the *headline accuracy* result — 240 samples/config defeats the single-run
stochastic-loop noise (which made earlier single runs unrankable) to rank
bf16/fp16/fp32/recurrent on loop-frequency + accuracy. Reproduces the table in
REPLAYSSM_FP16_NOTES.md. Bring the servers up with launch_maxprec.sh first.

Rigorous bf16 vs fp16 vs fp32 (+ recurrent) e2e suite for the ReplaySSM spec verify.
60 distinct problems (AIME-2024 + AIME-2025) x N repeats x 4 configs, greedy, max 32k.
Beats single-run stochastic-loop noise by measuring LOOP FREQUENCY over 60*N samples
per config. Saves raw JSON incrementally."""
import concurrent.futures as cf
import re, json, statistics, time
import requests
from collections import Counter
from datasets import load_dataset

N = 4                                   # repeats per config
CONFIGS = {"recurrent": 31010, "bf16": 31011, "fp16": 31012, "fp32": 31013}
LOOP_TH, SEVERE_TH = 50, 200            # max-line-repeat thresholds
QUERY = ("Solve the following AIME problem step by step. The last line of your response "
         "should be of the form Answer: $ANSWER.\n\n{q}\n\nRemember to put your answer on "
         "its own line after \"Answer:\", as an integer 000-999.")

def load_problems():
    p = []
    for cfg in ("AIME2025-I", "AIME2025-II"):
        for r in load_dataset("opencompass/AIME2025", cfg, split="test"):
            p.append({"q": r["question"], "a": str(r["answer"])})
    for r in load_dataset("Maxwell-Jia/AIME_2024", split="train"):
        p.append({"q": r["Problem"], "a": str(r["Answer"])})
    return p

def max_rep(t):
    lines = [l.strip() for l in t.splitlines() if len(l.strip()) > 8]
    return Counter(lines).most_common(1)[0][1] if lines else 0

def ask(port, q):
    try:
        r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json={
            "model": "Qwen/Qwen3.6-35B-A3B",
            "messages": [{"role": "user", "content": QUERY.format(q=q)}],
            "temperature": 0, "max_tokens": 32768}, timeout=3600)
        c = r.json()["choices"][0]; msg = c["message"]
        full = (msg.get("reasoning_content") or "") + "\n" + (msg.get("content") or "")
        mm = re.search(r"(?i)Answer\s*:\s*([0-9]+)", msg.get("content") or "")
        return {"tok": r.json().get("usage", {}).get("completion_tokens", 0),
                "maxrep": max_rep(full),
                "ans": (mm.group(1) if mm else None),
                "trunc": c.get("finish_reason") == "length"}
    except Exception as e:
        return {"err": str(e)[:60], "maxrep": 0, "tok": 0, "ans": None, "trunc": False}

def run_config(name, port, problems):
    rows = []
    for rep_i in range(N):
        t0 = time.time()
        with cf.ThreadPoolExecutor(max_workers=len(problems)) as ex:
            futs = {ex.submit(ask, port, p["q"]): i for i, p in enumerate(problems)}
            for f in cf.as_completed(futs):
                i = futs[f]; rec = f.result()
                rec["i"] = i; rec["rep_i"] = rep_i; rec["gold"] = problems[i]["a"]
                rows.append(rec)
        json.dump(rows, open(f"scratch/rigorous_{name}.json", "w"))
        print(f"[{name}] repeat {rep_i+1}/{N} done in {time.time()-t0:.0f}s ({len(rows)} samples)", flush=True)
    return name, rows

def main():
    problems = load_problems()
    print(f"problems: {len(problems)} (AIME 24+25); configs: {list(CONFIGS)}; N={N} "
          f"-> {len(problems)*N} samples/config", flush=True)
    out = {}
    with cf.ThreadPoolExecutor(max_workers=len(CONFIGS)) as ex:
        futs = [ex.submit(run_config, n, p, problems) for n, p in CONFIGS.items()]
        for f in cf.as_completed(futs):
            n, rows = f.result(); out[n] = rows
    json.dump(out, open("scratch/rigorous_all.json", "w"))
    print("\n================= RIGOROUS AGGREGATE =================")
    print(f"{'config':>10} | {'acc mean±std':>14} | {'loop>50':>10} | {'loop>200':>10} | {'trunc':>10} | {'avg_tok':>8}")
    for n in CONFIGS:
        rows = [r for r in out[n] if "err" not in r]
        ns = max(len(rows), 1)
        per_rep = []
        for rp in range(N):
            rr = [r for r in rows if r["rep_i"] == rp]
            if rr: per_rep.append(sum(1 for r in rr if r["ans"] == r["gold"]) / len(rr))
        accm = statistics.mean(per_rep) if per_rep else 0.0
        accs = statistics.pstdev(per_rep) if len(per_rep) > 1 else 0.0
        l50 = sum(1 for r in rows if r["maxrep"] >= LOOP_TH)
        l200 = sum(1 for r in rows if r["maxrep"] >= SEVERE_TH)
        tr = sum(1 for r in rows if r["trunc"])
        at = sum(r["tok"] for r in rows) // ns
        print(f"{n:>10} | {accm:.3f}±{accs:.3f} | {l50:>3}/{ns:<5} | {l200:>3}/{ns:<5} | {tr:>3}/{ns:<5} | {at:>8}")
    print("raw -> scratch/rigorous_all.json")

if __name__ == "__main__":
    main()
