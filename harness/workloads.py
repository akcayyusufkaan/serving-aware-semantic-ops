"""Workload loading and quality evaluation."""
import json

from . import cuad


def load(wcfg):
    if wcfg["name"] == "inline":
        return [{"id": str(i), "text": t} for i, t in enumerate(wcfg["texts"])]
    records = []
    with open(wcfg["path"]) as f:
        for line in f:
            records.append(json.loads(line))
    return records[: wcfg.get("n_records", len(records))]


def evaluate(wcfg, records, states, keys):
    """Quality of one run. keys: the state keys holding the final answers of the plan's operators."""
    if wcfg["name"] != "cuad":
        return None
    with open(wcfg["labels"]) as f:
        labels = json.load(f)
    labels = {cid: labels[cid] for cid in {r["contract_id"] for r in records}}  # n_records may cut contracts
    preds, failures = cuad.merge_chunks(records, states, keys)
    return {**cuad.score(labels, preds), "unparsed_answers": failures}
