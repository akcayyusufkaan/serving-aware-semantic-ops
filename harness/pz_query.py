"""Runs the Abacus CUAD query through Palimpzest against the switching proxy. Runs in the
Palimpzest environment (/opt/pz in the image), started by harness/pz_runner.py.

Usage: python pz_query.py <job.json> <out.json>
job.json: proxy_url, models {name: {usd_per_input_token, usd_per_output_token}}, policy, test, train,
          optimizer settings. out.json: plan, Palimpzest's stats (costs at the given prices) and
          the extracted spans per chunk.

Differences from Palimpzest 1.5.3 defaults, all needed for self-hosted models:
- the check allowing only one vLLM model per run is lifted at runtime;
- vLLM models, which Palimpzest prices at 0, get the API prices from the job, and their input tokens
  are counted (Palimpzest records 0 for vLLM);
- Abacus's validator scores sampled outputs against CUAD labels instead of calling an OpenAI judge.
"""
import inspect
import json
import os
import sys
import textwrap

import requests

# Palimpzest fetches metadata for API models from S3 at import time. Only self-hosted models are used
# here, so the fetch is answered locally: the run does not depend on that bucket being reachable.
class _AnyModel(dict):
    def get(self, key, default=None):
        return {"is_text_model": True, "usd_per_input_token": 0.0, "usd_per_output_token": 0.0,
                "seconds_per_output_token": 0.01, "MMLU_Pro_score": 0.0, "provider": "unused"}


class _Response:
    def json(self):
        return _AnyModel()


_get = requests.get
requests.get = lambda url, *a, **k: _Response() if "palimpzest-research" in url else _get(url, *a, **k)

import palimpzest as pz  # noqa: E402
from palimpzest.prompts.prompt_manager import PromptManager  # noqa: E402
from palimpzest.query.processor import query_processor_factory as qpf  # noqa: E402
from palimpzest.utils.model_info_helpers import predict_local_model_metrics  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from harness.cuad import CATEGORIES, NAMES, _entry  # noqa: E402

GUARD = 'raise ValueError("Only one vLLM model can be used per run. Multiple vLLM models found in available_models.")'


def lift_vllm_guard():
    fn = qpf.QueryProcessorFactory._config_validation_and_normalization.__func__
    src = textwrap.dedent(inspect.getsource(fn))
    assert GUARD in src, "Palimpzest version changed: one-vLLM-model check not found"
    ns = {}
    exec(src.replace(GUARD, "pass"), qpf.__dict__, ns)
    qpf.QueryProcessorFactory._config_validation_and_normalization = ns["_config_validation_and_normalization"]


def count_vllm_input_tokens():
    """Palimpzest records 0 input tokens for vLLM models, so their cost would only count output
    tokens. Count every prompt token at the input price, as an API without a cache discount would."""
    extract = PromptManager.extract_usage_stats

    def patched(self, usage, is_audio_op):
        stats = extract(self, usage, is_audio_op)
        if self.model.is_vllm_model():
            stats["input_text_tokens"] = usage.get("prompt_tokens") or 0
        return stats

    PromptManager.extract_usage_stats = patched


def load_chunks(path):
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    vals = [{"chunk_id": r["id"], "contract": r["text"]} for r in rows]
    labels = {r["id"]: r.get("labels", {}) for r in rows}
    return vals, labels


def as_spans(value):
    """Output field value as a list of strings. Missing fields come back as None or NaN."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v is not None]
    return []


class CUADValidator(pz.Validator):
    """Per-chunk F1 against the CUAD label spans that occur in the chunk (Abacus scores per contract)."""

    def __init__(self, labels):
        self.labels = labels
        self.model = None

    def map_score_fn(self, fields, input_record, output):
        labels = self.labels.get(input_record.get("chunk_id"), {})
        tp = fp = fn = 0
        for field in fields:
            t, f_p, f_n = _entry(labels.get(field, []), as_spans(output.get(field)), "Parties" in field)
            tp, fp, fn = tp + t, fp + f_p, fn + f_n
        if tp + fp + fn == 0:
            return 1.0   # nothing to find and nothing claimed
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def build_query(ds):
    # Abacus's "one-convert" CUAD query (abacus-research/cuad-demo.py), applied to chunks.
    cols = [{"name": name,
             "type": list[str],
             "desc": f"Extract the text spans (if they exist) from the contract corresponding to: {desc}. "
                     "If no spans exist, return an empty list. Quote text spans verbatim (do not summarize or paraphrase)."}
            for name, desc in CATEGORIES]
    return ds.sem_map(cols, depends_on=["contract"])


def policy_from(spec):
    kind = spec["kind"]
    if kind == "maxquality":
        return pz.MaxQuality()
    if kind == "mincost":
        return pz.MinCost()
    if kind == "mintime":
        return pz.MinTime()
    if kind == "maxquality_at_cost":
        return pz.MaxQualityAtFixedCost(max_cost=spec["max_cost"])
    raise ValueError(f"unknown policy {kind}")


def main():
    job = json.load(open(sys.argv[1]))
    lift_vllm_guard()
    count_vllm_input_tokens()

    models = []
    for name, price in job["models"].items():
        # Served under the short name; priors (MMLU-Pro, speed) are predicted from the Hugging Face id.
        m = pz.Model(f"hosted_vllm/{name}", api_base=job["proxy_url"] + "/v1")
        priors = predict_local_model_metrics(f"hosted_vllm/{price['hf_id']}")
        m.model_specs.update(MMLU_Pro_score=priors["MMLU_Pro_score"],
                             seconds_per_output_token=priors["seconds_per_output_token"],
                             usd_per_input_token=price["usd_per_input_token"],
                             usd_per_output_token=price["usd_per_output_token"])
        models.append(m)

    test_vals, _ = load_chunks(job["test"])
    if job.get("n_records"):
        test_vals = test_vals[: job["n_records"]]
    train_vals, train_labels = load_chunks(job["train"])

    query = build_query(pz.MemoryDataset(id="cuad", vals=test_vals))
    train = pz.MemoryDataset(id="cuad", vals=train_vals)
    opt = job["optimizer"]
    config = pz.QueryProcessorConfig(
        policy=policy_from(job["policy"]),
        available_models=models,
        execution_strategy=opt.get("execution_strategy", "parallel"),
        max_workers=opt.get("max_workers", 32),
        allow_bonded_query=True,
        allow_critic=opt.get("allow_critic", True),
        allow_mixtures=opt.get("allow_mixtures", True),
        allow_rag_reduction=opt.get("allow_rag_reduction", False),
        k=opt.get("k", 6),
        j=opt.get("j", 4),
        sample_budget=opt.get("sample_budget", 50),
        seed=opt.get("seed", 0),
        progress=False,
        verbose=False,
    )
    out = query.optimize_and_run(config=config, train_dataset=train, validator=CUADValidator(train_labels))

    stats = out.execution_stats
    preds = {}
    for r in out.data_records:
        preds[r.chunk_id] = {name: as_spans(getattr(r, name, None)) for name in NAMES}
    result = {
        "plan": list(stats.plan_strs.values()),
        "optimization_time": stats.optimization_time,
        "optimization_cost": stats.optimization_cost,
        "plan_execution_time": stats.plan_execution_time,
        "plan_execution_cost": stats.plan_execution_cost,
        "total_execution_cost": stats.total_execution_cost,
        "start_time": stats.start_time,
        "optimization_end_time": stats.optimization_end_time,
        "input_tokens": stats.input_text_tokens,
        "output_tokens": stats.output_text_tokens,
        "predictions": preds,
    }
    json.dump(result, open(sys.argv[2], "w"))


if __name__ == "__main__":
    main()
