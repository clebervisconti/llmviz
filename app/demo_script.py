"""
demo_script.py — the scripted DEMO model (no torch required).

DEMO is the default landing experience: it produces a deterministic, plausible-looking
run with the SAME response shape as the real model, so the frontend code is identical.
It is honestly labelled "scripted" in the UI — the numbers are synthesized, not from a
real LLM, but they are internally consistent (causal attention, stable embeddings, a
coherent continuation) so the lesson reads correctly.

The number of layers/heads follows the selected tier's spec, so even in DEMO mode the
model-size selector visibly changes the diagram — demonstrating the headline feature.
"""
from __future__ import annotations

import re
import numpy as np

from . import internals
from .model_manager import MODELS_BY_ID

# A fixed, coherent continuation the demo "generates" token by token.
CONTINUATION = ["mat", "and", "watched", "the", "rain", "fall", "quietly", "."]

# A small curated vocabulary for the next-token probability bars.
DEMO_VOCAB = [
    "mat", "floor", "chair", "and", "but", "then", "watched", "saw", "heard",
    "the", "a", "rain", "sun", "snow", "fall", "rise", "shine", "quietly",
    "softly", "slowly", "outside", "window", "room", ".", ",", "—",
]
VOCAB_INDEX = {w: i for i, w in enumerate(DEMO_VOCAB)}
DEFAULT_PROMPT = "The cat sat on the"

_word_re = re.compile(r"\w+|[^\w\s]")


def _token_id(text: str) -> int:
    """Stable pseudo-id for a token string (just for display; demo only)."""
    return abs(hash(text)) % 50000


def tokenize(prompt: str) -> list[dict]:
    raw = _word_re.findall(prompt.strip()) or _word_re.findall(DEFAULT_PROMPT)
    toks = []
    for i, w in enumerate(raw):
        disp = w if i == 0 else (" " + w if re.match(r"\w", w) else w)
        toks.append({"id": _token_id(w), "text": disp, "i": i})
    return toks[: internals.MAX_SEQ]


def _vec(text: str, dim: int) -> np.ndarray:
    """Deterministic embedding vector for a token string."""
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    return rng.standard_normal(dim).astype(np.float32)


def _causal_attention(seq: int, heads: int, layer: int, rng: np.random.Generator) -> np.ndarray:
    """Build a (heads, seq, seq) causal attention tensor with recency bias and a bit
    of per-(layer,head) structure, then softmax over valid (earlier) keys."""
    out = np.zeros((heads, seq, seq), dtype=np.float32)
    for h in range(heads):
        # each head gets a slightly different recency vs. content bias
        recency = 0.5 + 0.5 * ((h + layer) % 4) / 3.0
        for q in range(seq):
            scores = np.full(seq, -1e9, dtype=np.float32)
            for k in range(q + 1):
                dist = q - k
                base = -recency * dist                     # prefer nearby tokens
                base += 0.6 * rng.standard_normal()        # texture
                if k == 0:
                    base += 0.8                            # attention-sink on first token
                scores[k] = base
            out[h, q] = internals.softmax(scores)
    return out


def _synth_qkv(seq: int, head_dim: int, n_heads: int, layer: int,
               rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthesize believable (seq, n_heads, head_dim) Q/K/V for the DEMO expanded-block
    view. Q/K norms drift up with position; V carries a mild depth-dependent gain. Honestly
    scripted — internally consistent so the lesson reads correctly, but not from a real LLM."""
    def make(scale_with_pos: float, base: float) -> np.ndarray:
        t = np.empty((seq, n_heads, head_dim), dtype=np.float32)
        for i in range(seq):
            gain = base + scale_with_pos * (i / max(seq - 1, 1))
            t[i] = rng.standard_normal((n_heads, head_dim)).astype(np.float32) * gain
        return t
    q = make(0.6, 0.8)
    k = make(0.5, 0.9)
    v = make(0.2, 1.0 + 0.04 * layer)
    return q, k, v


def _logits(step: int, temperature: float, top_k: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Raw vocab logits (target word on top) and the temperature/top-k-processed version."""
    rng = np.random.default_rng(1000 + step)
    raw = rng.standard_normal(len(DEMO_VOCAB)).astype(np.float32) * 1.2
    target = CONTINUATION[step] if step < len(CONTINUATION) else "."
    tgt_idx = VOCAB_INDEX.get(target, VOCAB_INDEX["."])
    raw[tgt_idx] = raw.max() + 2.5            # make the target clearly the top token

    # processed = temperature scaling then top-k masking (what sampling actually sees)
    t = max(0.05, float(temperature))
    scaled = raw / t
    proc = scaled.copy()
    k = max(1, min(int(top_k), len(DEMO_VOCAB)))
    if k < len(DEMO_VOCAB):
        cutoff = np.sort(scaled)[-k]
        proc[scaled < cutoff] = -1e9
    sampled = {"id": _token_id(target), "text": (" " + target if target.isalnum() else target)}
    return raw, proc, sampled


def generate_step(prompt: str, tier: str, temperature: float, top_k: int,
                  generated: list[int], head=None, focus_layer=None,
                  want_qkv: bool = False) -> dict:
    spec = MODELS_BY_ID.get(tier, MODELS_BY_ID["demo"])
    n_layers, n_heads, dim = spec["layers"], spec["heads"], min(spec["dim"], 48)

    step = len(generated)
    # full current sequence = prompt tokens + already-generated continuation tokens
    tokens = tokenize(prompt)
    for j in range(step):
        w = CONTINUATION[j] if j < len(CONTINUATION) else "."
        disp = " " + w if w.isalnum() else w
        tokens.append({"id": _token_id(w), "text": disp, "i": len(tokens)})
    tokens = tokens[: internals.MAX_SEQ]
    seq = len(tokens)

    # embeddings + per-layer hidden states (norms drift across depth)
    embeddings = np.stack([_vec(t["text"], dim) for t in tokens])
    hiddens = []
    h = embeddings.copy()
    for li in range(n_layers):
        h = h + 0.15 * np.tanh(h) + 0.05 * (li + 1)   # cheap depth-dependent transform
        hiddens.append(h.copy())

    attentions = [
        _causal_attention(seq, n_heads, li, np.random.default_rng(7 * li + 13))
        for li in range(n_layers)
    ]

    raw, proc, sampled = _logits(step, temperature, top_k)
    done = step >= len(CONTINUATION) - 1 or seq >= internals.MAX_SEQ

    qkv = None
    if want_qkv:
        focus = (n_layers - 1) if focus_layer is None else max(0, min(int(focus_layer), n_layers - 1))
        focus_layer = focus
        head_dim = max(1, dim // n_heads)
        qkv = _synth_qkv(seq, head_dim, n_heads, focus, np.random.default_rng(31 * focus + 5))

    return internals.assemble_step(
        step=step, tokens=tokens, embeddings=embeddings,
        attentions=attentions, hiddens=hiddens,
        logits_raw=raw, logits_sampled=proc, sampled=sampled,
        id_to_text=_logits_id_to_text,   # demo logit rows are indexed by DEMO_VOCAB position
        done=done, head=head, focus_layer=focus_layer, qkv=qkv,
    )


def _logits_id_to_text(i: int) -> str:
    return DEMO_VOCAB[i] if 0 <= i < len(DEMO_VOCAB) else "?"
