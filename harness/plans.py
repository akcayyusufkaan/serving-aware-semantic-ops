"""Executes a plan (a list of operators) under a given execution strategy.

Each operator is expanded into single-model steps:
  map -> one step
  moa -> one step per proposer, then an aggregator step
so strategies differ only in the order in which steps are executed.

Operators call vLLM's OpenAI-compatible API directly. Swapping in Palimpzest or LOTUS
operators only touches this module.
"""
import threading
from concurrent.futures import ThreadPoolExecutor

import requests


def expand(op):
    if op["op"] == "map":
        return [{"model": op["model"], "key": op["name"], "prompt": op["prompt"]}]
    if op["op"] == "moa":
        steps = [
            {"model": m, "key": f'{op["name"]}__p{i}', "prompt": op["prompt"]}
            for i, m in enumerate(op["proposers"])
        ]
        steps.append({
            "model": op["aggregator"],
            "key": op["name"],
            "prompt": op["aggregate_prompt"],
            "inputs": [s["key"] for s in steps],
        })
        return steps
    raise ValueError(f'unknown operator type: {op["op"]}')


def chat(url, model, prompt, max_tokens):
    r = requests.post(url + "/v1/chat/completions", json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }, timeout=600)
    r.raise_for_status()
    j = r.json()
    u = j.get("usage") or {}
    details = u.get("prompt_tokens_details") or {}
    return (j["choices"][0]["message"]["content"],
            u.get("prompt_tokens", 0), u.get("completion_tokens", 0), details.get("cached_tokens") or 0)


class Usage:
    """Per-model token counts; thread-safe because batched steps run concurrently."""

    def __init__(self, models):
        self.lock = threading.Lock()
        self.by_model = {m: {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0} for m in models}

    def add(self, model, p, c, cached):
        with self.lock:
            d = self.by_model[model]
            d["prompt_tokens"] += p
            d["completion_tokens"] += c
            d["cached_tokens"] += cached


def _run_step(pool, step, record, state, usage, max_tokens):
    fields = {"text": record["text"], **state}
    if "inputs" in step:
        fields["proposals"] = "\n\n".join(state[k] for k in step["inputs"])
    prompt = step["prompt"].format(**fields)
    out, p, c, cached = chat(pool.url(step["model"]), step["model"], prompt, max_tokens)
    usage.add(step["model"], p, c, cached)
    state[step["key"]] = out


def run_plan(pool, cfg, ops, strategy, records):
    steps = [s for op in ops for s in expand(op)]
    states = {r["id"]: {} for r in records}
    usage = Usage(cfg["models"])
    max_tokens = cfg.get("max_tokens", 256)

    if strategy == "naive":
        # Record at a time, one request in flight; the model is switched whenever a step needs it.
        for r in records:
            for s in steps:
                pool.activate(s["model"])
                _run_step(pool, s, r, states[r["id"]], usage, max_tokens)

    elif strategy == "batched":
        # Step at a time: every record goes through a step before the next one starts,
        # with requests sent concurrently so vLLM can batch them.
        for s in steps:
            pool.activate(s["model"])
            with ThreadPoolExecutor(max_workers=cfg.get("max_concurrency", 32)) as ex:
                futures = [ex.submit(_run_step, pool, s, r, states[r["id"]], usage, max_tokens) for r in records]
                for f in futures:
                    f.result()
    else:
        raise ValueError(f"unknown strategy: {strategy}")

    return states, usage.by_model
