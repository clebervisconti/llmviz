"""
LLMViz — FastAPI backend.

Serves the static frontend and a small JSON API that drives the visualization:
  GET  /api/models         -> tiers + availability
  POST /api/tokenize       -> token chips for a prompt
  POST /api/generate_step  -> one generation step with downsampled internals
  GET  /api/health         -> liveness + memory

DEMO mode (default) runs with no ML deps. Live GPT-2 tiers run only if torch +
transformers are installed (see requirements.txt / DEPLOYMENT.md). A single asyncio
lock serializes inference so concurrent students can't OOM the 4GB VPS.
"""
from __future__ import annotations

import asyncio
import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import demo_script
from .internals import MAX_SEQ
from .model_manager import MANAGER, MODELS_BY_ID, live_enabled, tier_live, torch_available

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE, "static")

app = FastAPI(title="LLMViz", docs_url=None, redoc_url=None)

# Serialize inference: one model run at a time (memory guard).
_infer_lock = asyncio.Lock()


@app.middleware("http")
async def cache_headers(request, call_next):
    """index.html must never be cached (so asset-version bumps take effect immediately);
    versioned /static/* assets and /api responses get sensible caching."""
    resp = await call_next(request)
    path = request.url.path
    if path == "/" or path.endswith(".html"):
        resp.headers["Cache-Control"] = "no-store, must-revalidate"
    elif path.startswith("/static/"):
        resp.headers["Cache-Control"] = "public, max-age=3600"
    elif path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


class TokenizeReq(BaseModel):
    prompt: str = Field(default="", max_length=2000)
    model: str = "demo"


class GenerateReq(BaseModel):
    prompt: str = Field(default="", max_length=2000)
    model: str = "demo"
    temperature: float = Field(default=0.8, ge=0.05, le=2.0)
    top_k: int = Field(default=10, ge=1, le=50)
    generated: List[int] = Field(default_factory=list)
    generated_text: str = Field(default="", max_length=4000)   # used by the MLX/GEMMA tier
    head: Optional[int] = None
    focus_layer: Optional[int] = None
    want_qkv: bool = False        # expanded-block view: extract real Q/K/V for the focus layer


def _engine_of(tier: str) -> str:
    spec = MODELS_BY_ID.get(tier) or {}
    return spec.get("engine", "demo" if spec.get("hf") is None else "hf")


def _is_mlx(tier: str) -> bool:
    return MODELS_BY_ID.get(tier, {}).get("engine") == "mlx"


def _is_demo(tier: str) -> bool:
    spec = MODELS_BY_ID.get(tier)
    return spec is None or (spec.get("hf") is None and spec.get("engine") != "mlx")


@app.get("/api/health")
def health():
    mem_mb = None
    try:
        import resource
        mem_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    except Exception:
        pass
    return {
        "status": "ok",
        "torch": torch_available(),
        "live": live_enabled(),
        "max_seq": MAX_SEQ,
        "mem_mb": mem_mb,
    }


@app.get("/api/models")
def models():
    return {"models": MANAGER.public_registry()}


def _use_scripted(tier: str) -> bool:
    """Scripted engine when the DEMO tier is chosen, OR when a live tier isn't enabled
    (torch absent, or gated off for memory) — so the size selector still changes the
    diagram and the lesson always runs. Responses carry engine='scripted'|'live'."""
    return _is_demo(tier) or not tier_live(tier)


@app.post("/api/tokenize")
def tokenize(req: TokenizeReq):
    if _is_mlx(req.model):
        from . import mlx_backend
        if mlx_backend.configured():
            try:
                t = mlx_backend.tokenize(req.prompt or " ")
                return {"tokens": t, "count": len(t), "engine": "mlx"}
            except Exception as e:  # noqa: BLE001
                raise HTTPException(502, f"MLX tokenize failed: {e}")
    if _use_scripted(req.model):
        t = demo_script.tokenize(req.prompt)
        return {"tokens": t, "count": len(t), "engine": "scripted"}
    try:
        from . import inference
        toks = inference.tokenize(req.prompt or " ", req.model)
        return {"tokens": toks, "count": len(toks), "engine": "live"}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"tokenize failed: {e}")


@app.post("/api/generate_step")
async def generate_step(req: GenerateReq):
    prompt = req.prompt.strip() or demo_script.DEFAULT_PROMPT

    # GEMMA tier → MLX server on the Mac (via tunnel). Serialize: the single MLX server
    # handles one generation at a time, so queue here rather than firing concurrent requests.
    if _is_mlx(req.model):
        from . import mlx_backend
        if not mlx_backend.configured():
            raise HTTPException(503, "GEMMA backend not configured")
        try:
            await asyncio.wait_for(_infer_lock.acquire(), timeout=45)
        except asyncio.TimeoutError:
            raise HTTPException(503, "model busy — try again in a moment")
        try:
            return await asyncio.to_thread(
                mlx_backend.generate_step,
                prompt, req.model, req.temperature, req.top_k, req.generated_text,
            )
        except Exception as e:  # noqa: BLE001
            raise HTTPException(502, f"MLX generation failed: {e}")
        finally:
            _infer_lock.release()

    if _use_scripted(req.model):
        # pure-python and fast; uses the selected tier's geometry (layers/heads)
        out = demo_script.generate_step(
            prompt, req.model, req.temperature, req.top_k,
            req.generated, head=req.head, focus_layer=req.focus_layer,
            want_qkv=req.want_qkv,
        )
        out["engine"] = "scripted"
        return out
    # Queue (don't reject) when another inference is in flight — one model runs at a
    # time on this box, but callers wait their turn rather than seeing a "busy" error.
    # Bounded so a stuck request can't hang the UI forever.
    try:
        await asyncio.wait_for(_infer_lock.acquire(), timeout=45)
    except asyncio.TimeoutError:
        raise HTTPException(503, "model warming up — try again in a moment")
    try:
        from . import inference
        out = await asyncio.to_thread(   # blocking CPU forward pass off the event loop
            inference.generate_step,
            prompt, req.model, req.temperature, req.top_k,
            req.generated, req.head, req.focus_layer, req.want_qkv,
        )
        out["engine"] = "live"
        return out
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"generation failed: {e}")
    finally:
        _infer_lock.release()


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/favicon.ico")
def favicon():
    return FileResponse(os.path.join(STATIC_DIR, "img", "cv-monogram.png"))


# Static assets last so it doesn't shadow the API routes.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.exception_handler(404)
async def not_found(_req, _exc):
    return JSONResponse({"error": "not found"}, status_code=404)
