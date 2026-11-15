"""Builds the CUAD workload: contracts selected as in Abacus, split into chunks, plus labels.

Usage:
  python3 scripts/prepare_cuad.py --cuad-json <CUAD>/test.json --n 25 --out-dir data
  python3 scripts/prepare_cuad.py --cuad-json <CUAD>/train_separate_questions.json --split train --n 25 --out-dir data

Contract selection matches Palimpzest's abacus-research/cuad-demo.py: titles in order of first
appearance, shuffled with numpy's default_rng(seed), first n (Abacus uses 100 test contracts for
the run and 25 train contracts for its validator). Writes, with <name> = cuad_<n> for the test split
and cuad_train_<n> for the train split:
  <out-dir>/<name>.jsonl        one record per chunk: id, contract_id, chunk, text, labels
  <out-dir>/<name>_labels.json  contract_id -> category -> list of label spans
A chunk's labels are the contract's label spans that occur verbatim in that chunk; Abacus's
validator scores sampled outputs per record, which needs labels at that level.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from harness.cuad import NAMES, label_category  # noqa: E402


def chunk_text(text, max_chars):
    """Split on line breaks into chunks of at most max_chars; overlong lines are split on spaces."""
    pieces = []
    for line in text.split("\n"):
        while len(line) > max_chars:
            cut = line.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            pieces.append(line[:cut])
            line = line[cut:].lstrip()
        pieces.append(line)
    chunks, cur = [], ""
    for p in pieces:
        if cur and len(cur) + 1 + len(p) > max_chars:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n{p}" if cur else p
    if cur.strip():
        chunks.append(cur)
    return [c for c in chunks if c.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuad-json", required=True)
    ap.add_argument("--split", choices=["test", "train"], default="test")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk-chars", type=int, default=4000)
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    raw = json.load(open(args.cuad_json))
    titles, contracts = [], {}
    for article in raw["data"]:
        title = article.get("title", "").strip()
        for para in article["paragraphs"]:
            if title not in contracts:
                titles.append(title)
                contracts[title] = {"id": para["qas"][0]["id"], "text": para["context"].strip(), "qas": []}
            contracts[title]["qas"].extend(para["qas"])

    np.random.default_rng(seed=args.seed).shuffle(titles)
    selected = titles[: args.n]

    os.makedirs(args.out_dir, exist_ok=True)
    records, labels, found = [], {}, set()
    for title in selected:
        c = contracts[title]
        cid = c["id"]
        labels[cid] = {name: [] for name in NAMES}
        for qa in c["qas"]:
            name = label_category(qa["id"])
            assert name in labels[cid], name
            labels[cid][name].extend(a["text"] for a in qa.get("answers", []))
        for i, chunk in enumerate(chunk_text(c["text"], args.chunk_chars)):
            chunk_labels = {name: [s for s in spans if s in chunk] for name, spans in labels[cid].items()}
            records.append({"id": f"{cid}#{i}", "contract_id": cid, "chunk": i, "text": chunk,
                            "labels": {k: v for k, v in chunk_labels.items() if v}})
            found.update((cid, name, s) for name, spans in chunk_labels.items() for s in spans)

    name = f"cuad_{args.n}" if args.split == "test" else f"cuad_train_{args.n}"
    base = os.path.join(args.out_dir, name)
    with open(base + ".jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    with open(base + "_labels.json", "w") as f:
        json.dump(labels, f)
    n_labels = sum(len(set(v)) for cat in labels.values() for v in cat.values())
    print(f"{len(selected)} contracts, {len(records)} chunks, {n_labels} label spans "
          f"({len(found)} found verbatim inside one chunk) -> {base}.jsonl")


if __name__ == "__main__":
    main()
