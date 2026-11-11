"""Probes how Palimpzest 1.5.3 plans and executes over two self-hosted vLLM models.

Usage: python pz_probe.py <scenario>      (scenarios: see SCENARIOS below)

Environment:
  PATCH_GUARD=1        lift Palimpzest's one-vLLM-model-per-run check at runtime
  PZ_OFFLINE=1         serve placeholder model metadata instead of fetching it from S3
  PZ_MODEL_A/B         served model names (default: Llama-3.1-8B / Qwen2.5-7B ids)
  PZ_PORT_A/B          server ports (default: 8001 / 8002)
  BUDGET               sample_budget for the sentinel scenarios (default: 40)

The sentinel scenarios use LabelValidator, which scores mock_vllm.py outputs and is
meaningless against real models.
"""
import inspect
import os
import sys
import textwrap

import requests

# Palimpzest fetches model metadata from S3 at import time and fails to import without it.
if os.environ.get("PZ_OFFLINE") == "1":
    class _AnyModel(dict):
        def get(self, key, default=None):
            return {"is_text_model": True, "usd_per_input_token": 1e-6, "usd_per_output_token": 1e-6,
                    "seconds_per_output_token": 0.01, "MMLU_Pro_score": 50.0, "provider": "placeholder"}

    class _Response:
        def json(self):
            return _AnyModel()

    _get = requests.get
    requests.get = lambda url, *a, **k: _Response() if "palimpzest-research" in url else _get(url, *a, **k)

import palimpzest as pz  # noqa: E402
from palimpzest.query.processor import query_processor_factory as qpf  # noqa: E402

GUARD = 'raise ValueError("Only one vLLM model can be used per run. Multiple vLLM models found in available_models.")'

if os.environ.get("PATCH_GUARD") == "1":
    fn = qpf.QueryProcessorFactory._config_validation_and_normalization.__func__
    src = textwrap.dedent(inspect.getsource(fn)).replace(GUARD, "pass")
    ns = {}
    exec(src, qpf.__dict__, ns)
    qpf.QueryProcessorFactory._config_validation_and_normalization = ns["_config_validation_and_normalization"]

# Names must match vLLM's --served-model-name (the Hugging Face id by default).
NAME_A = os.environ.get("PZ_MODEL_A", "meta-llama/Llama-3.1-8B-Instruct")
NAME_B = os.environ.get("PZ_MODEL_B", "Qwen/Qwen2.5-7B-Instruct")
PORT_A = int(os.environ.get("PZ_PORT_A", "8001"))
PORT_B = int(os.environ.get("PZ_PORT_B", "8002"))
A = pz.Model(f"hosted_vllm/{NAME_A}", api_base=f"http://127.0.0.1:{PORT_A}/v1")
B = pz.Model(f"hosted_vllm/{NAME_B}", api_base=f"http://127.0.0.1:{PORT_B}/v1")

COUNTERS = ("vllm:request_success_total", "vllm:prompt_tokens_total",
            "vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total")


def vllm_counters():
    """Request and cache counters per server; empty when running against the mock."""
    out = {}
    for name, port in (("A", PORT_A), ("B", PORT_B)):
        try:
            text = requests.get(f"http://127.0.0.1:{port}/metrics", timeout=5).text
        except requests.RequestException:
            continue
        values = {}
        for line in text.splitlines():
            for key in COUNTERS:
                if line.startswith(key + "{") or line.startswith(key + " "):
                    try:
                        values[key] = values.get(key, 0) + float(line.rsplit(" ", 1)[1])
                    except ValueError:
                        pass
        if values:
            out[name] = values
    return out


REVIEWS = [{"text": t} for t in [
    "Battery lasts two days, great phone.", "Screen cracked after a week.", "Fast shipping, item as described.",
    "Stopped working after a month, refund refused.", "Decent value for the price.", "Terrible support, never again."]]
FILTER = "The review is about a product or a purchase."
SENTIMENT = [{"name": "sentiment", "desc": "positive or negative", "type": str}]
SUMMARY = [{"name": "summary", "desc": "one sentence summary", "type": str}]

ds = pz.MemoryDataset(id="reviews", vals=REVIEWS)


def pipeline(d):
    return d.sem_filter(FILTER).sem_map(SENTIMENT).sem_map(SUMMARY)


def config(**kw):
    return pz.QueryProcessorConfig(**{"available_models": [A, B], "progress": False, "verbose": False, **kw})


class LabelValidator(pz.Validator):
    """Prefers model A for sentiment and model B for summary, based on mock outputs."""

    def __init__(self):
        self.model = None

    def map_score_fn(self, fields, input_record, output):
        if "summary" in fields:
            return 1.0 if output.get("summary") == f"mock_p{PORT_B}" else 0.0
        if "sentiment" in fields:
            return 1.0 if output.get("sentiment") == f"mock_p{PORT_A}" else 0.0
        return 1.0

    def filter_score_fn(self, filter_str, input_record, output):
        return 1.0


def sentinel(**kw):
    budget = int(os.environ.get("BUDGET", "40"))
    return pipeline(ds).optimize_and_run(config(sample_budget=budget, **kw), validator=LabelValidator())


def split_mixed():
    """Forces a mixed plan by running each operator as its own query, locked to one model."""
    lock = dict(optimizer_strategy="none", allow_mixtures=False, allow_critic=False, allow_rag_reduction=False,
                allow_bonded_query=True, execution_strategy="sequential")
    o1 = ds.sem_filter(FILTER).run(config(available_models=[B], **lock))
    d1 = pz.MemoryDataset(id="after_filter", vals=[{"text": r.text} for r in o1.data_records])
    o2 = d1.sem_map(SENTIMENT).run(config(available_models=[A], **lock))
    d2 = pz.MemoryDataset(id="after_sentiment",
                          vals=[{"text": r.text, "sentiment": r.sentiment} for r in o2.data_records])
    return d2.sem_map(SUMMARY).run(config(available_models=[B], **lock))


SCENARIOS = {
    "two_models_default": lambda: pipeline(ds).run(config()),
    "two_models_mintime": lambda: pipeline(ds).run(config(policy=pz.MinTime())),
    "two_models_none": lambda: pipeline(ds).run(config(optimizer_strategy="none")),
    "two_models_none_seq": lambda: pipeline(ds).run(config(optimizer_strategy="none", execution_strategy="sequential")),
    "two_models_none_pipe": lambda: pipeline(ds).run(config(optimizer_strategy="none", execution_strategy="pipelined")),
    "two_models_locked_flags": lambda: pipeline(ds).run(config(allow_mixtures=False, allow_critic=False,
                                                               allow_rag_reduction=False, allow_split_merge=False)),
    "one_model_locked": lambda: pipeline(ds).run(config(available_models=[B], optimizer_strategy="none",
                                                        allow_mixtures=False, allow_critic=False,
                                                        allow_rag_reduction=False, allow_model_selection=False)),
    "none_order_BA": lambda: pipeline(ds).run(config(available_models=[B, A], optimizer_strategy="none")),
    "split_mixed": split_mixed,
    "sentinel_default": sentinel,
    "sentinel_noextras": lambda: sentinel(allow_mixtures=False, allow_critic=False,
                                          allow_rag_reduction=False, allow_bonded_query=True),
}


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in SCENARIOS:
        sys.exit(f"usage: python pz_probe.py <{'|'.join(SCENARIOS)}>")
    for m in (A, B):
        print("model", m, {k: m.model_specs[k] for k in ("MMLU_Pro_score", "usd_per_input_token",
                                                          "seconds_per_output_token")})
    try:
        before = vllm_counters()
        out = SCENARIOS[sys.argv[1]]()
        print("PLAN:", getattr(out, "executed_plans", ""))
        print("records out:", len(out.data_records))
        for r in out.data_records[:3]:
            print("  ", {k: getattr(r, k, None) for k in ("text", "sentiment", "summary")})
        after = vllm_counters()
        for k in after:
            print("vLLM", k, {m: after[k][m] - before.get(k, {}).get(m, 0) for m in after[k]})
    except Exception as e:
        print("ERROR:", type(e).__name__, e)


if __name__ == "__main__":
    main()
