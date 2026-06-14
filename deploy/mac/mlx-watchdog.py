#!/usr/bin/env python3
"""
mlx-watchdog.py — runs the MLX (mlx_lm) server and keeps it healthy.

The plain `mlx_lm.server` can (a) crash (Metal OOM) or (b) HANG — stay alive but stop
answering generation requests (observed under concurrent load; /v1/models still returned
200 while /v1/completions hung). A bare restart-on-exit supervisor misses (b). This
watchdog probes GENERATION periodically and kills + restarts on hang OR crash.

Layering: launchd (LaunchAgent) keeps THIS watchdog alive across crashes/reboots; the
watchdog keeps the MLX server alive across crashes/hangs.

Env: HF_HUB_CACHE, HF_HOME, MLX_MODEL, MLX_PORT, MLX_PROBE_INTERVAL, MLX_PROBE_TIMEOUT,
     MLX_MAX_FAILS.
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

MODEL = os.environ.get("MLX_MODEL", "mlx-community/gemma-2-9b-it-4bit")
PORT = os.environ.get("MLX_PORT", "8081")
PROBE_INTERVAL = int(os.environ.get("MLX_PROBE_INTERVAL", "30"))   # seconds between probes
PROBE_TIMEOUT = int(os.environ.get("MLX_PROBE_TIMEOUT", "60"))     # a real step is ~1-2s
MAX_FAILS = int(os.environ.get("MLX_MAX_FAILS", "2"))             # consecutive fails → restart
START_GRACE = int(os.environ.get("MLX_START_GRACE", "180"))       # model load can take a while

_child = None


def log(msg):
    print(f"[watchdog {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def start_server():
    global _child
    log(f"starting mlx server: {MODEL} on :{PORT}")
    _child = subprocess.Popen(
        [sys.executable, "-m", "mlx_lm", "server", "--model", MODEL,
         "--host", "127.0.0.1", "--port", PORT],
        env=os.environ.copy(),
    )


def stop_server():
    global _child
    if _child and _child.poll() is None:
        log("stopping mlx server")
        _child.terminate()
        try:
            _child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            _child.kill()
    _child = None


def models_up():
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=8)
        return True
    except Exception:
        return False


def generation_ok():
    """The real health check: can it actually generate? (/v1/models can lie during a hang.)"""
    body = json.dumps({"model": MODEL, "prompt": "ok", "max_tokens": 1}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=PROBE_TIMEOUT) as r:
            return r.status == 200
    except Exception as e:
        log(f"generation probe failed: {e}")
        return False


def _term(*_):
    stop_server()
    sys.exit(0)


def main():
    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)
    while True:
        start_server()
        # wait for the model to load (bounded)
        t0 = time.time()
        while time.time() - t0 < START_GRACE:
            if _child.poll() is not None:
                break
            if models_up():
                log("mlx server is up")
                break
            time.sleep(5)
        # health loop
        fails = 0
        while True:
            time.sleep(PROBE_INTERVAL)
            if _child.poll() is not None:
                log("mlx server exited (crash) — restarting")
                break
            if generation_ok():
                fails = 0
            else:
                fails += 1
                log(f"unhealthy ({fails}/{MAX_FAILS})")
                if fails >= MAX_FAILS:
                    log("hang detected — killing + restarting")
                    stop_server()
                    break
        stop_server()
        time.sleep(3)


if __name__ == "__main__":
    main()
