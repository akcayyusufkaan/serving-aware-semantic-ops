"""Experiment config loading and run matrix generation."""
import hashlib
import itertools
import json
from dataclasses import dataclass

import yaml


@dataclass(frozen=True)
class Run:
    plan: str
    strategy: str
    repeat: int
    run_id: str


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # The hash is part of every run_id, so editing the config never mixes old and new results.
    cfg["_hash"] = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    return cfg


def run_matrix(cfg):
    """One Run per (plan, strategy, repeat)."""
    runs = []
    for plan, strategy, rep in itertools.product(cfg["plans"], cfg["strategies"], range(cfg["repeats"])):
        key = f'{cfg["experiment"]}|{plan}|{strategy}|{rep}|{cfg["_hash"]}'
        run_id = hashlib.sha1(key.encode()).hexdigest()[:12]
        runs.append(Run(plan, strategy, rep, run_id))
    return runs
