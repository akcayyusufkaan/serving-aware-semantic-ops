"""Entry point: python3 -m harness.runner --config configs/smoke.yaml

Starts the vLLM servers, then executes every run in the matrix that is not yet in
<output_dir>/<experiment>/runs.jsonl. Each finished run is appended as one line, so an
interrupted experiment resumes where it stopped.
"""
import argparse
import json
import os
import subprocess
import time
from importlib.metadata import PackageNotFoundError, version

from . import metrics, plans, workloads
from .config import load_config, run_matrix
from .servers import ServerPool


def env_info(cfg):
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        gpu = None
    try:
        vllm_version = version("vllm")
    except PackageNotFoundError:
        vllm_version = None
    return {
        "env": os.environ.get("RUN_ENV", "unknown"),  # set by the launcher
        "image_digest": os.environ.get("IMAGE_DIGEST"),
        "gpu_name": gpu,
        "vllm_version": vllm_version,
        "config_hash": cfg["_hash"],
    }


def done_ids(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {json.loads(line)["run_id"] for line in f if line.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", help="override output_dir from the config (does not change config_hash)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    out_dir = os.path.join(cfg["output_dir"], cfg["experiment"])
    os.makedirs(out_dir, exist_ok=True)
    env = env_info(cfg)
    with open(os.path.join(out_dir, "env.json"), "w") as f:
        json.dump(env, f, indent=2)

    results_path = os.path.join(out_dir, "runs.jsonl")
    done = done_ids(results_path)
    todo = [r for r in run_matrix(cfg) if r.run_id not in done]
    print(f"{len(done)} runs done, {len(todo)} remaining", flush=True)
    if not todo:
        return

    records = workloads.load(cfg["workload"])
    pool = ServerPool(cfg, out_dir)
    try:
        pool.start_all()

        # Warm-up so one-off first-request latency does not land in any run.
        for name in cfg["models"]:
            pool.activate(name)
            plans.chat(pool.url(name), name, "Hello", 8)

        for i, run in enumerate(todo, 1):
            ops = cfg["plans"][run.plan]

            # Same starting state for every run (not timed): first model awake, empty cache.
            pool.activate(plans.expand(ops[0])[0]["model"])
            pool.reset_cache()
            before = {n: metrics.snapshot(pool.url(n)) for n in cfg["models"]}
            pool.reset_counters()

            t0 = time.perf_counter()
            states, usage = plans.run_plan(pool, cfg, ops, run.strategy, records)
            wall = time.perf_counter() - t0

            after = {n: metrics.snapshot(pool.url(n)) for n in cfg["models"]}
            row = {
                "run_id": run.run_id,
                "experiment": cfg["experiment"],
                "plan": run.plan,
                "strategy": run.strategy,
                "repeat": run.repeat,
                **env,
                "n_records": len(records),
                "wall_time_s": wall,
                "n_model_switches": pool.n_switches,
                "switch_time_s": pool.switch_time_s,
                "usage": usage,
                "prefix_cache": {n: metrics.delta(before[n], after[n]) for n in cfg["models"]},
                "quality_f1": workloads.evaluate(cfg["workload"], records, states, ops[-1]["name"]),
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            with open(results_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[{i}/{len(todo)}] {run.plan}/{run.strategy}/r{run.repeat}: "
                  f"{wall:.1f} s, {pool.n_switches} switches", flush=True)
    finally:
        pool.stop_all()


if __name__ == "__main__":
    main()
