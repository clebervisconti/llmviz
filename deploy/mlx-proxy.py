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
        conn_cls = http.client.HTTPSConnection if _U.scheme == "https" else http.client.HTTPConnection
        up = conn_cls(_U.netloc, timeout=600)
        try:
            up.putrequest(method, self.path, skip_host=False, skip_accept_encoding=True)
            for k, v in self.headers.items():
                if k.lower() not in _HOP:
                    up.putheader(k, v)
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

    def do_GET(self):
        self._relay("GET")

    def do_POST(self):
        self._relay("POST")

    def log_message(self, *a):  # quiet
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
