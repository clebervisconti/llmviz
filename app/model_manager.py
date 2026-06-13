"""
model_manager.py — model registry + lazy loader for the real GPT-2 tiers.

Design goals (see docs/ARCHITECTURE.md §6):
  - torch/transformers are OPTIONAL. If they aren't installed, the app still runs in
    DEMO mode; live tiers report available=False.
  - Load at most what's needed; keep NANO/MICRO resident, lazy-load SMALL and free it
    after use to respect the 4GB VPS budget.
  - A single asyncio lock (held in main.py) serializes inference so two students can't
    run models concurrently and OOM the box.
"""
from __future__ import annotations

import os

# Production gate: even if torch is installed, live models stay OFF unless explicitly
# enabled — protects the shared 4GB VPS from OOM. Flip per-tier once benchmarked.
#   LLMVIZ_LIVE=1                 enable live inference at all
#   LLMVIZ_LIVE_TIERS=nano,micro  which tiers may run live (others serve scripted)
LIVE_ENABLED = os.environ.get("LLMVIZ_LIVE", "0") == "1"
LIVE_TIERS = set(t.strip() for t in os.environ.get("LLMVIZ_LIVE_TIERS", "nano,micro,small").split(",") if t.strip())

# The tier registry is plain data — importing it never needs torch.
MODELS = [
    {"id": "demo",  "label": "DEMO",  "hf": None,          "params": "scripted",
     "layers": 6,  "heads": 8,  "dim": 32,   "default_demo": True,
     "blurb": "A scripted, instant walkthrough — perfect for lectures. No real model runs."},
    {"id": "nano",  "label": "NANO",  "hf": "distilgpt2",  "params": "82M",
     "layers": 6,  "heads": 12, "dim": 768,
     "blurb": "DistilGPT-2 — a distilled, 6-layer real model. Fast and small."},
    {"id": "micro", "label": "MICRO", "hf": "gpt2",        "params": "124M",
     "layers": 12, "heads": 12, "dim": 768, "default_live": True,
     "blurb": "GPT-2 small — the original 124M model. 12 layers, 12 heads."},
    {"id": "small", "label": "SMALL", "hf": "gpt2-medium", "params": "355M",
     "layers": 24, "heads": 16, "dim": 1024, "lazy": True,
     "blurb": "GPT-2 medium — 355M params, 24 layers. Loaded on demand."},
    {"id": "gemma", "label": "GEMMA", "hf": None, "params": "9B",
     "layers": 42, "heads": 16, "dim": 3584, "engine": "mlx",
     "blurb": "Gemma 2 9B on Apple MLX (Mac mini, via tunnel) — a real, capable model. Shows "
              "real tokens, probabilities & generated text; attention/layers are white-box only "
              "(use NANO/MICRO for those)."},
]
MODELS_BY_ID = {m["id"]: m for m in MODELS}
# On the shared 4GB VPS, keep at most ONE live model resident at a time (~650MB each).
# Switching tiers unloads the previous model. Benchmarked 2026-06-13: distilgpt2 654MB/46ms,
# gpt2 634MB/452ms per forward pass with attentions; gpt2-medium left scripted (too tight).
MAX_RESIDENT = 1


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except Exception:
        return False


def live_enabled() -> bool:
    """Live inference is on only if explicitly enabled AND torch is importable."""
    return LIVE_ENABLED and torch_available()


def tier_live(tier: str) -> bool:
    """Whether this specific tier should run the real model (vs. scripted fallback)."""
    return live_enabled() and tier in LIVE_TIERS


class ModelManager:
    """Lazy cache of (tokenizer, model) per tier. Thread-safety is handled by the
    single-flight lock in main.py — this class assumes serialized access."""

    def __init__(self) -> None:
        self._cache: dict[str, tuple] = {}

    def public_registry(self) -> list[dict]:
        """The /api/models payload: tiers + whether each is actually runnable here."""
        from . import mlx_backend
        out = []
        for m in MODELS:
            entry = {k: m[k] for k in ("id", "label", "params", "layers", "heads", "dim", "blurb")}
            if m["id"] == "demo":
                entry["available"] = True
            elif m.get("engine") == "mlx":
                entry["available"] = mlx_backend.configured()
            else:
                entry["available"] = tier_live(m["id"])
            entry["engine"] = m.get("engine", "hf")
            entry["default"] = bool(m.get("default_demo"))   # DEMO is the landing default
            out.append(entry)
        return out

    def get(self, tier: str):
        """Return (tokenizer, model) for a live tier, loading if needed. Raises if torch
        is missing or the tier is unknown/demo."""
        spec = MODELS_BY_ID.get(tier)
        if spec is None or spec["hf"] is None:
            raise ValueError(f"'{tier}' is not a live model tier")
        if not torch_available():
            raise RuntimeError("torch/transformers not installed — live models unavailable")

        if tier in self._cache:
            return self._cache[tier]

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(2)          # match the 2-vCPU box; predictable CPU use
        tok = AutoTokenizer.from_pretrained(spec["hf"])
        model = AutoModelForCausalLM.from_pretrained(
            spec["hf"], attn_implementation="eager"   # eager attn so output_attentions works
        )
        model.eval()
        torch.set_grad_enabled(False)

        # Keep only MAX_RESIDENT models in memory: evict the others before caching this one.
        if len(self._cache) >= MAX_RESIDENT:
            for k in list(self._cache):
                del self._cache[k]
            import gc; gc.collect()
        self._cache[tier] = (tok, model)
        return self._cache[tier]


MANAGER = ModelManager()
