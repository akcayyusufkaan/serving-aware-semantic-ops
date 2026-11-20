"""Proxy test against the stub vLLM: concurrent requests for two models, in random order.

Run from the repository root: PATH=$PWD/tests/stub_vllm:$PATH python3 tests/test_proxy.py
The stub answers 409 to any completion sent while it is asleep, so a wrong switch shows up as a failure.
"""
import json
import os
import random
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import requests

sys.path.insert(0, ".")
from harness.proxy import SwitchingProxy  # noqa: E402
from harness.servers import ServerPool  # noqa: E402

cfg = {"models": {"a": {"hf_id": "x", "port": 8101}, "b": {"hf_id": "y", "port": 8102}},
       "vllm": {"gpu_memory_utilization": 0.5, "max_model_len": 512}}
tmp = tempfile.mkdtemp()
os.environ["STUB_LOG"] = os.path.join(tmp, "requests.jsonl")   # the stub logs every completion it serves
pool = ServerPool(cfg, tmp)
pool.start_all()
proxy = SwitchingProxy(pool, 8100, max_tokens=123)
proxy.start()


def call(model):
    r = requests.post(proxy.url() + "/v1/chat/completions",
                      json={"model": f"hosted_vllm/{model}", "messages": [{"role": "user", "content": "hi"}]})
    return r.status_code


try:
    random.seed(0)
    models = [random.choice("ab") for _ in range(300)]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=32) as ex:
        codes = list(ex.map(call, models))
    bad = [c for c in codes if c != 200]
    print(f"{len(codes)} requests, {len(bad)} failed, {pool.n_switches} switches, {time.time() - t0:.1f} s")
    unknown = requests.post(proxy.url() + "/v1/chat/completions", json={"model": "zzz", "messages": []}).status_code
    print("unknown model ->", unknown)
    capped = requests.post(proxy.url() + "/v1/chat/completions",
                           json={"model": "a", "max_tokens": 7, "messages": [{"role": "user", "content": "hi"}]}).status_code
    # A long prompt: the stub counts ~490 of its 512 tokens, so the limit must drop below 123.
    long = requests.post(proxy.url() + "/v1/chat/completions",
                         json={"model": "a", "messages": [{"role": "user", "content": "x" * 1900}]}).status_code
    with open(os.environ["STUB_LOG"]) as f:
        limits = [json.loads(line)["max_tokens"] for line in f]
    print("max_tokens seen by the servers:", sorted(set(limits)), "| capped by proxy:", proxy.n_capped,
          "| truncated:", proxy.n_truncated)
    assert not bad and unknown == 404 and capped == 200 and long == 200
    assert limits.count(123) == 300 and limits[-2] == 7 and 0 < limits[-1] < 123 and proxy.n_capped == 301
    assert proxy.n_truncated == 1   # only the request with max_tokens 7 (the stub stops at "length" below 10)
    print("OK")
finally:
    proxy.stop()
    pool.stop_all()
