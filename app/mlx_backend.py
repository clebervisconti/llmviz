"""
mlx_backend.py — the GEMMA tier. Talks to an MLX (mlx_lm.server) OpenAI-compatible
endpoint (the Mac mini, reached via the cloudflared tunnel behind Cloudflare Access).

What it can show honestly from an inference server:
  - real tokenization of the prompt
  - real next-token probability bars (from the server's logprobs)
  - real token-by-token generated text
It CANNOT expose attention or hidden states (no inference server does), so the GEMMA
tier sets caps.attention=False / caps.embeddings=False and the frontend hides those
panels — the white-box GPT-2 tiers remain for the internals lesson.

Config (env): LLMVIZ_MLX_URL, LLMVIZ_MLX_MODEL, LLMVIZ_MLX_LAYERS,
              LLMVIZ_MLX_CF_ID, LLMVIZ_MLX_CF_SECRET (Cloudflare Access service token).
"""
from __future__ import annotations

import json
import os
import urllib.request

import numpy as np

from . import internals

MLX_URL = os.environ.get("LLMVIZ_MLX_URL", "").rstrip("/")
MLX_MODEL = os.environ.get("LLMVIZ_MLX_MODEL", "mlx-community/gemma-2-9b-it-4bit")
MLX_LAYERS = int(os.environ.get("LLMVIZ_MLX_LAYERS", "42"))   # Gemma-2-9B = 42 layers
CF_ID = os.environ.get("LLMVIZ_MLX_CF_ID", "")
CF_SECRET = os.environ.get("LLMVIZ_MLX_CF_SECRET", "")

_tok = None


def configured() -> bool:
    return bool(MLX_URL)


def _tokenizer():
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(MLX_MODEL)
    return _tok


def _post(path: str, payload: dict, timeout: int = 120) -> dict:
    headers = {"Content-Type": "application/json"}
    if CF_ID and CF_SECRET:
        headers["CF-Access-Client-Id"] = CF_ID
        headers["CF-Access-Client-Secret"] = CF_SECRET
    req = urllib.request.Request(MLX_URL + path, data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def tokenize(prompt: str):
    tok = _tokenizer()
    ids = tok.encode(prompt, add_special_tokens=False)[: internals.MAX_SEQ]
    return [{"id": int(t), "text": tok.decode([t]), "i": i} for i, t in enumerate(ids)]


def _templated_prompt(prompt: str) -> str:
    """Wrap the user's prompt in Gemma's chat template so the IT model behaves well."""
    tok = _tokenizer()
    try:
        return tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return prompt


def generate_step(prompt: str, tier: str, temperature: float, top_k: int,
                  generated_text: str = "", **_) -> dict:
    tok = _tokenizer()
    base = _templated_prompt(prompt)
    full = base + (generated_text or "")

    resp = _post("/v1/completions", {
        "model": MLX_MODEL,
        "prompt": full,
        "max_tokens": 1,
        "temperature": float(temperature),
        "logprobs": max(5, min(int(top_k), 20)),
    })
    choice = resp["choices"][0]
    lp = (choice.get("logprobs") or {}).get("top_logprobs") or [[]]
    pairs = lp[0] if lp else []          # [[token_id, logprob], ...] for the next token

    # decode to a probability distribution for the bars
    dist = []
    for entry in pairs:
        tid, logprob = entry[0], entry[1]
        dist.append({"id": int(tid), "text": tok.decode([int(tid)]), "p": float(np.exp(logprob))})
    s = sum(d["p"] for d in dist) or 1.0
    for d in dist:
        d["p"] = round(d["p"] / s, 4)    # renormalize the visible top-k

    gen_piece = choice.get("text", "")
    sampled = {"id": (dist[0]["id"] if dist else 0), "text": gen_piece or (dist[0]["text"] if dist else "")}

    # token chips: the user's prompt tokens + the generated text so far
    tokens = tokenize(prompt)
    if generated_text:
        for j, t in enumerate(tok.encode(generated_text, add_special_tokens=False)):
            tokens.append({"id": int(t), "text": tok.decode([t]), "i": len(tokens)})
    tokens = tokens[: internals.MAX_SEQ]

    # architecture-only layer blocks (real depth, but no activation data from a server)
    layers = [{"index": i, "hidden_norm": None} for i in range(MLX_LAYERS)]

    finished = choice.get("finish_reason") in ("stop", "length") and not gen_piece.strip()
    done = finished or len(tok.encode((generated_text or "") + gen_piece)) >= internals.MAX_SEQ

    return {
        "step": len(tok.encode(generated_text)) if generated_text else 0,
        "engine": "mlx",
        "model_label": "GEMMA 2 9B",
        "caps": {"attention": False, "embeddings": False, "layers_static": True},
        "tokens": tokens,
        "embeddings_2d": [],
        "layers": layers,
        "logits_raw": dist,
        "logits_sampled": dist,
        "sampled": sampled,
        "done": bool(done),
    }
