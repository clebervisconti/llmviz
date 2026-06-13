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

import http.client
import json
import os
from urllib.parse import urlparse

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
    # http.client (not urllib): avoids urllib's default "Python-urllib" User-Agent,
    # which Cloudflare's bot protection blocks in front of the Access-gated endpoint.
    u = urlparse(MLX_URL)
    Conn = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    conn = Conn(u.netloc, timeout=timeout)
    body = json.dumps(payload).encode()
    try:
        conn.putrequest("POST", path, skip_host=False, skip_accept_encoding=True)
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Content-Length", str(len(body)))
        conn.putheader("User-Agent", "LLMViz/1.0")
        if CF_ID and CF_SECRET:
            conn.putheader("CF-Access-Client-Id", CF_ID)
            conn.putheader("CF-Access-Client-Secret", CF_SECRET)
        conn.endheaders()
        conn.send(body)
        resp = conn.getresponse()
        data = resp.read()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {data[:200].decode(errors='replace')}")
        return json.loads(data)
    finally:
        conn.close()


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

    # Stop cleanly at Gemma's end-of-turn / eos markers (instruct model). Without this the
    # model keeps emitting past its real answer and drifts into markdown/filler.
    STOP_MARKERS = ("<end_of_turn>", "<eos>", "<start_of_turn>")
    gen_piece = choice.get("text", "")
    # NOTE: with max_tokens=1, finish_reason is "length" EVERY step — that is NOT a real stop.
    # Only "stop" (model emitted EOS/end-of-turn) or a stop marker in the text ends generation.
    done = choice.get("finish_reason") == "stop"
    for mk in STOP_MARKERS:
        if mk in gen_piece:
            gen_piece = gen_piece.split(mk)[0]
            done = True
    if gen_piece == "":
        done = True   # nothing generated → natural end
    sampled = {"id": (dist[0]["id"] if dist else 0), "text": gen_piece}

    # token chips: the user's prompt tokens + the generated text so far
    tokens = tokenize(prompt)
    if generated_text:
        for j, t in enumerate(tok.encode(generated_text, add_special_tokens=False)):
            tokens.append({"id": int(t), "text": tok.decode([t]), "i": len(tokens)})
    tokens = tokens[: internals.MAX_SEQ]

    # architecture-only layer blocks (real depth, but no activation data from a server)
    layers = [{"index": i, "hidden_norm": None} for i in range(MLX_LAYERS)]

    done = done or len(tok.encode((generated_text or "") + gen_piece)) >= internals.MAX_SEQ

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
