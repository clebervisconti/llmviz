#!/usr/bin/env python3
"""
mlx-proxy.py — tiny localhost reverse proxy that fronts the Cloudflare-Access-gated
MLX endpoint, injecting the CF-Access service-token headers. Lets any OpenAI client on
the VPS (e.g. the Hermes agent) use a normal base_url (http://127.0.0.1:PORT/v1) without
needing to send custom Access headers itself.

stdlib only (no deps). Streams responses so SSE (stream=true) chat completions work.

Env: LLMVIZ_MLX_URL, LLMVIZ_MLX_CF_ID, LLMVIZ_MLX_CF_SECRET, MLX_PROXY_PORT (default 8083).
"""
import http.client
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

UPSTREAM = os.environ.get("LLMVIZ_MLX_URL", "https://mlx.cybersphere.com.br").rstrip("/")
CF_ID = os.environ.get("LLMVIZ_MLX_CF_ID", "")
CF_SECRET = os.environ.get("LLMVIZ_MLX_CF_SECRET", "")
PORT = int(os.environ.get("MLX_PROXY_PORT", "8083"))
_U = urlparse(UPSTREAM)
_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
        "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length"}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _relay(self, method: str):
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length) if length else None
        body = self._gemma_fix(body)
        conn_cls = http.client.HTTPSConnection if _U.scheme == "https" else http.client.HTTPConnection
        up = conn_cls(_U.netloc, timeout=600)
        try:
            up.putrequest(method, self.path, skip_host=False, skip_accept_encoding=True)
            # Send a CLEAN, minimal header set. Do NOT forward the client's User-Agent /
            # Accept-Encoding / cookies — Cloudflare's bot protection 502s on python-httpx's
            # default headers. Only what the upstream OpenAI API needs.
            up.putheader("Content-Type", self.headers.get("Content-Type", "application/json"))
            up.putheader("Accept", self.headers.get("Accept", "application/json"))
            up.putheader("User-Agent", "LLMViz-Proxy/1.0")
            up.putheader("CF-Access-Client-Id", CF_ID)
            up.putheader("CF-Access-Client-Secret", CF_SECRET)
            if body is not None:
                up.putheader("Content-Length", str(len(body)))
            up.endheaders()
            if body:
                up.send(body)
            resp = up.getresponse()

            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() in _HOP:
                    continue
                self.send_header(k, v)
            # Delimit the body by closing the connection at EOF. This is correct whether the
            # upstream used Content-Length OR chunked (http.client de-chunks for us, and we
            # strip transfer-encoding) — and it works for SSE (stream=true) too.
            self.send_header("Connection", "close")
            self.close_connection = True
            self.end_headers()
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except Exception as e:  # noqa: BLE001
            try:
                self.send_error(502, f"proxy error: {e}")
            except Exception:
                pass
        finally:
            up.close()

    def _gemma_fix(self, body):
        """Gemma 2's chat template rejects the 'system' role. Merge any system messages
        into the first user turn so clients that send a system prompt (e.g. Hermes) work."""
        if not body or "chat/completions" not in self.path:
            return body
        try:
            obj = json.loads(body)
            msgs = obj.get("messages")
            if not isinstance(msgs, list):
                return body
            sys_txt = "\n\n".join(
                m["content"] for m in msgs
                if m.get("role") == "system" and isinstance(m.get("content"), str)
            )
            if not sys_txt:
                return body
            rest = [m for m in msgs if m.get("role") != "system"]
            for m in rest:
                if m.get("role") == "user":
                    m["content"] = sys_txt + "\n\n" + (m.get("content") or "")
                    break
            else:
                rest.insert(0, {"role": "user", "content": sys_txt})
            obj["messages"] = rest
            return json.dumps(obj).encode()
        except Exception:
            return body

    def do_GET(self):
        self._relay("GET")

    def do_POST(self):
        self._relay("POST")

    def log_message(self, *a):  # quiet
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
