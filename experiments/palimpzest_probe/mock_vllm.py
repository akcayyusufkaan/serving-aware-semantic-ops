"""Minimal stand-in for vLLM's OpenAI-compatible chat endpoint, for testing without a GPU.

Usage: python mock_vllm.py <log.jsonl> <port> [<port> ...]

Every request is appended to the log with the port it hit. Map answers contain
"mock_p<port>" so the serving model can be identified from the output.
"""
import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOG = sys.argv[1]
PORTS = [int(p) for p in sys.argv[2:]]
lock = threading.Lock()


def handler(port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            text = json.dumps(body["messages"])
            is_filter = "TRUE" in text and "FALSE" in text
            if is_filter:
                answer = "REASONING: mock\nANSWER: TRUE\n---"
            else:
                fields = set(re.findall(r"- (\w+):", text))
                payload = json.dumps({f: f"mock_p{port}" for f in fields})
                answer = f"REASONING: mock\nANSWER:\n```json\n{payload}\n```\n---"
            with lock, open(LOG, "a") as f:
                f.write(json.dumps({"t": time.time(), "port": port, "model": body.get("model"),
                                    "kind": "filter" if is_filter else "map", "snippet": text[-160:]}) + "\n")
            time.sleep(0.05)
            out = json.dumps({
                "id": "mock", "object": "chat.completion", "created": int(time.time()), "model": body.get("model"),
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": answer}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    return Handler


if __name__ == "__main__":
    for p in PORTS:
        server = ThreadingHTTPServer(("127.0.0.1", p), handler(p))
        threading.Thread(target=server.serve_forever, daemon=True).start()
    print("listening on", PORTS, flush=True)
    threading.Event().wait()
