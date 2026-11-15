"""Runs Abacus (Palimpzest) on the self-hosted models and measures the plans it chooses.

Started by harness.runner when the config has a `palimpzest` section. For every policy it
1. resets the prefix cache and the switch counters,
2. runs harness/pz_query.py in the Palimpzest environment against the switching proxy: Abacus
   samples operators on the train chunks (optimization), picks a plan and runs it on the test chunks,
3. writes one line to pz_runs.jsonl: the chosen plan, Palimpzest's cost at API prices, measured
   time and model switches per phase, prefix-cache counters, and F1 on the test contracts.
"""
import hashlib
import json
import os
import subprocess
import tempfile
import time

from . import cuad, metrics, plans
from .proxy import SwitchingProxy
from .servers import ServerPool


def merge_predictions(records, preds):
    by_contract = {}
    for r in records:
        merged = by_contract.setdefault(r["contract_id"], {name: [] for name in cuad.NAMES})
        for name, spans in preds.get(r["id"], {}).items():
            merged[name].extend(s for s in spans if s not in merged[name])
    return by_contract


def run(cfg, out_dir, env):
    pzc = cfg["palimpzest"]
    results_path = os.path.join(out_dir, "pz_runs.jsonl")
    done = set()
    if os.path.exists(results_path):
        with open(results_path) as f:
            done = {json.loads(line)["run_id"] for line in f if line.strip()}
    jobs = []
    for spec in pzc["policies"]:
        for rep in range(cfg.get("repeats", 1)):
            key = f'{cfg["experiment"]}|{json.dumps(spec, sort_keys=True)}|{rep}|{cfg["_hash"]}'
            jobs.append((spec, rep, hashlib.sha1(key.encode()).hexdigest()[:12]))
    todo = [j for j in jobs if j[2] not in done]
    print(f"{len(jobs) - len(todo)} Palimpzest runs done, {len(todo)} remaining", flush=True)
    if not todo:
        return

    with open(pzc["test"]) as f:
        test_records = [json.loads(line) for line in f][: pzc.get("n_records")]
    with open(pzc["labels"]) as f:
        labels = json.load(f)
    labels = {cid: labels[cid] for cid in {r["contract_id"] for r in test_records}}

    pool = ServerPool(cfg, out_dir)
    proxy = SwitchingProxy(pool, pzc.get("proxy_port", 8000))
    try:
        pool.start_all()
        for name in cfg["models"]:
            pool.activate(name)
            plans.chat(pool.url(name), name, "Hello", 8)
        proxy.start()

        for i, (spec, rep, run_id) in enumerate(todo, 1):
            pool.reset_cache()
            before = {n: metrics.snapshot(pool.url(n)) for n in cfg["models"]}
            pool.reset_counters()
            models = {n: {"hf_id": cfg["models"][n]["hf_id"], **pzc["prices"][n]} for n in cfg["models"]}
            job = {"proxy_url": proxy.url(), "models": models, "policy": spec,
                   "test": pzc["test"], "train": pzc["train"], "n_records": pzc.get("n_records"),
                   "optimizer": pzc.get("optimizer", {})}
            with tempfile.TemporaryDirectory() as tmp:
                job_path, res_path = os.path.join(tmp, "job.json"), os.path.join(tmp, "result.json")
                json.dump(job, open(job_path, "w"))
                t0 = time.perf_counter()
                # Run in tmp: Palimpzest writes one file per unparsable answer to ./parse-answer-errors.
                proc = subprocess.run([pzc.get("python", "/opt/pz/bin/python"),
                                       os.path.join(os.path.dirname(os.path.abspath(__file__)), "pz_query.py"),
                                       job_path, res_path], cwd=tmp)
                wall = time.perf_counter() - t0
                if proc.returncode != 0:
                    raise RuntimeError(f"pz_query.py failed for policy {spec}")
                res = json.load(open(res_path))
                err_dir = os.path.join(tmp, "parse-answer-errors")
                parse_errors = len(os.listdir(err_dir)) if os.path.isdir(err_dir) else 0
            after = {n: metrics.snapshot(pool.url(n)) for n in cfg["models"]}

            boundary = res.get("optimization_end_time") or 0
            phases = {"optimization": [s for s in pool.switch_log if s[0] < boundary],
                      "execution": [s for s in pool.switch_log if s[0] >= boundary]}
            preds = res.pop("predictions")
            row = {
                "run_id": run_id,
                "experiment": cfg["experiment"],
                "policy": spec,
                "repeat": rep,
                **env,
                "n_test_chunks": len(test_records),
                "plan": res["plan"],
                "api_cost_usd": {"optimization": res["optimization_cost"], "execution": res["plan_execution_cost"]},
                "time_s": {"optimization": res["optimization_time"], "execution": res["plan_execution_time"],
                           "wall": wall},
                "model_switches": {k: {"n": len(v), "time_s": sum(d for _, d in v)} for k, v in phases.items()},
                "tokens": {"input": res["input_tokens"], "output": res["output_tokens"]},
                "parse_errors": parse_errors,
                "prefix_cache": {n: metrics.delta(before[n], after[n]) for n in cfg["models"]},
                "quality": cuad.score(labels, merge_predictions(test_records, preds)),
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            with open(results_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[{i}/{len(todo)}] {spec}: {wall:.0f} s, F1 {row['quality']['f1']:.3f}, "
                  f"API ${res['plan_execution_cost']:.4f}, {pool.n_switches} switches", flush=True)
    finally:
        proxy.stop()
        pool.stop_all()
