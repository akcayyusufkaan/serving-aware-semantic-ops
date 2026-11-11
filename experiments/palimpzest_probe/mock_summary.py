"""Summarizes a mock_vllm log: requests per port and how often consecutive requests changed port."""
import collections
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1])]
print("requests:", len(rows), "| by port:", dict(collections.Counter(r["port"] for r in rows)))
print("port changes between consecutive requests:", sum(a["port"] != b["port"] for a, b in zip(rows, rows[1:])))
