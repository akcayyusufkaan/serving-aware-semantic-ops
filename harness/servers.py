"""Manages one vLLM server per model and keeps exactly one of them awake.

Sleeping servers use level 1 sleep: weights are offloaded to CPU memory and the KV cache is dropped.
"""
import os
import subprocess
import time

import requests


class ServerPool:
    def __init__(self, cfg, log_dir):
        self.cfg = cfg
        self.log_dir = log_dir
        self.procs = {}
        self.awake = None
        self.n_switches = 0
        self.switch_time_s = 0.0

    def url(self, name):
        return f'http://127.0.0.1:{self.cfg["models"][name]["port"]}'

    def _cmd(self, name):
        m = self.cfg["models"][name]
        v = self.cfg["vllm"]
        cmd = [
            "vllm", "serve", m["hf_id"],
            "--port", str(m["port"]),
            "--served-model-name", name,
            "--enable-sleep-mode",
            "--enable-prompt-tokens-details",  # adds cached_tokens to the usage block
            "--gpu-memory-utilization", str(v["gpu_memory_utilization"]),
            "--max-model-len", str(v["max_model_len"]),
        ]
        if v.get("enforce_eager"):
            cmd.append("--enforce-eager")
        return cmd

    def _post(self, name, path, timeout=600):
        r = requests.post(self.url(name) + path, timeout=timeout)
        r.raise_for_status()

    def _wait_healthy(self, name, timeout=1800):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.procs[name].poll() is not None:
                raise RuntimeError(f"server {name} exited, see {self.log_dir}/vllm_{name}.log")
            try:
                if requests.get(self.url(name) + "/health", timeout=2).status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(2)
        raise TimeoutError(f"server {name} not healthy after {timeout} s")

    def start_all(self):
        # vLLM checks free GPU memory at startup, so each server is put to sleep
        # before the next one starts. The last one stays awake.
        env = {**os.environ, "VLLM_SERVER_DEV_MODE": "1"}
        env.update({k: str(v) for k, v in self.cfg["vllm"].get("env", {}).items()})
        names = list(self.cfg["models"])
        for i, name in enumerate(names):
            log = open(os.path.join(self.log_dir, f"vllm_{name}.log"), "a")
            self.procs[name] = subprocess.Popen(self._cmd(name), env=env, stdout=log, stderr=subprocess.STDOUT)
            self._wait_healthy(name)
            if i < len(names) - 1:
                self._post(name, "/sleep?level=1")
        self.awake = names[-1]

    def activate(self, name):
        """Wake `name`, putting the currently awake server to sleep first. Timed as a model switch."""
        if self.awake == name:
            return
        t0 = time.perf_counter()
        if self.awake is not None:
            self._post(self.awake, "/sleep?level=1")
        self._post(name, "/wake_up")
        self.switch_time_s += time.perf_counter() - t0
        self.n_switches += 1
        self.awake = name

    def reset_cache(self):
        # Only the awake server holds a KV cache.
        self._post(self.awake, "/reset_prefix_cache")

    def reset_counters(self):
        self.n_switches = 0
        self.switch_time_s = 0.0

    def stop_all(self):
        for p in self.procs.values():
            p.terminate()
        for p in self.procs.values():
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                p.kill()
