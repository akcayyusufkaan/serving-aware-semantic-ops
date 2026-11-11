"""Workload loading and quality evaluation."""
import json


def load(wcfg):
    if wcfg["name"] == "inline":
        return [{"id": str(i), "text": t} for i, t in enumerate(wcfg["texts"])]
    records = []
    with open(wcfg["path"]) as f:
        for line in f:
            records.append(json.loads(line))  # {"id": ..., "text": ..., "label": ...}
    return records[: wcfg.get("n_records", len(records))]


def evaluate(wcfg, records, states, final_key):
    # TODO: F1 against CUAD labels. Inline workloads have no labels.
    return None
