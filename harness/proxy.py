"""One OpenAI-compatible endpoint in front of the vLLM servers of a ServerPool.

Clients such as Palimpzest send requests for any model to this proxy. The proxy wakes the
requested model (putting the awake one to sleep) and forwards the request.

Switching policy: the awake model keeps being served while requests for it arrive, so concurrent
requests are batched; once a request for another model has waited HOLD_S seconds, new requests for
the awake model are held back, and the switch happens when the in-flight ones have finished.
Switch count and time are kept by the pool.

max_tokens: requests that set no output limit get this one. Palimpzest sends none, so a model that
keeps repeating spans generates until the context is full (minutes per request on a small KV cache).
vLLM rejects a request whose prompt plus max_tokens exceeds max_model_len, so the limit is lowered
to what is left after the prompt, counted with the server's /tokenize.
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests


HOLD_S = 5.0


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 256   # the default listen backlog (5) resets connections under concurrent clients


class SwitchingProxy:
    def __init__(self, pool, port, max_tokens=None):
        self.pool = pool
        self.port = port
        self.max_tokens = max_tokens
        self.n_capped = 0      # requests that got max_tokens from the proxy
        self.n_truncated = 0   # answers that stopped at max_tokens (finish_reason "length")
        self.cond = threading.Condition()
        self.inflight = 0
        self.waiting = {}   # model -> arrival times of requests waiting for it
        self.server = _Server(("127.0.0.1", port), self._handler())
        self.thread = None

    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def start(self):
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self):
        if self.thread:   # shutdown() waits for serve_forever, which only runs after start()
            self.server.shutdown()
        self.server.server_close()

    def _acquire(self, model):
        with self.cond:
            arrived = time.monotonic()
            self.waiting.setdefault(model, []).append(arrived)
            while True:
                others = [t for m, ts in self.waiting.items() if m != model for t in ts]
                overdue = bool(others) and time.monotonic() - min(others) >= HOLD_S
                if self.pool.awake == model and not overdue:
                    break
                if self.pool.awake != model and self.inflight == 0:
                    self.pool.activate(model)   # sleeps the awake model, wakes this one
                    break
                self.cond.wait(timeout=0.5)
            self.waiting[model].remove(arrived)
            self.inflight += 1
            self.cond.notify_all()

    def _limit(self, model, req):
        """proxy.max_tokens, or less when the prompt leaves less room in the context."""
        with self.cond:
            self.n_capped += 1
        try:
            t = requests.post(self.pool.url(model) + "/tokenize", timeout=60,
                              json={"model": model, "messages": req.get("messages", []), "add_generation_prompt": True}).json()
            return max(1, min(self.max_tokens, t["max_model_len"] - t["count"]))
        except (requests.RequestException, ValueError, KeyError):
            return self.max_tokens

    @staticmethod
    def _truncated(response):
        try:
            return any(c.get("finish_reason") == "length" for c in response.json().get("choices", []))
        except ValueError:
            return False

    def _release(self):
        with self.cond:
            self.inflight -= 1
            self.cond.notify_all()

    def _handler(self):
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _reply(self, status, body, content_type="application/json"):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path in ("/health", "/v1/models"):
                    models = [{"id": m, "object": "model"} for m in proxy.pool.cfg["models"]]
                    return self._reply(200, json.dumps({"object": "list", "data": models}).encode())
                self._reply(404, b"{}")

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                try:
                    req = json.loads(body)
                    model = req.get("model", "")
                except ValueError:
                    return self._reply(400, b'{"error": "invalid JSON"}')
                model = model.split("/")[-1]
                if model not in proxy.pool.cfg["models"]:
                    return self._reply(404, json.dumps({"error": f"unknown model {model}"}).encode())
                cap = proxy.max_tokens and not (req.get("max_tokens") or req.get("max_completion_tokens"))
                proxy._acquire(model)
                try:
                    if cap:
                        req["max_tokens"] = proxy._limit(model, req)
                        body = json.dumps(req).encode()
                    r = requests.post(proxy.pool.url(model) + self.path, data=body,
                                      headers={"Content-Type": "application/json"}, timeout=1200)
                    self._reply(r.status_code, r.content, r.headers.get("Content-Type", "application/json"))
                    if r.ok and proxy._truncated(r):
                        with proxy.cond:
                            proxy.n_truncated += 1
                except requests.RequestException as e:
                    self._reply(502, json.dumps({"error": str(e)}).encode())
                finally:
                    proxy._release()

        return Handler
