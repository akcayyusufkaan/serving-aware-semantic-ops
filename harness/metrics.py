"""Prefix cache counters from vLLM's /metrics endpoint.

The counters are cumulative, so per-run values are the difference between two snapshots.
"""
import requests

COUNTERS = ("vllm:prefix_cache_queries", "vllm:prefix_cache_hits")


def snapshot(url):
    text = requests.get(url + "/metrics", timeout=10).text
    out = {c: 0.0 for c in COUNTERS}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name = line.split("{")[0].split(" ")[0]
        for c in COUNTERS:
            if name in (c, c + "_total"):  # Prometheus may expose counters with a _total suffix
                out[c] += float(line.rsplit(" ", 1)[1])
    return out


def delta(before, after):
    return {k: after[k] - before[k] for k in before}
