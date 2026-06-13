"""
inference.py — the real model path. Runs a GPT-2-family model on CPU and extracts
genuine attention + hidden states, then hands the numpy arrays to internals.assemble_step
so the response matches the DEMO path exactly.

Only imported lazily by main.py when a live tier is requested AND torch is available.
"""
from __future__ import annotations

import numpy as np

from . import internals
from .model_manager import MANAGER


def tokenize(prompt: str, tier: str) -> list[dict]:
    tok, _ = MANAGER.get(tier)
    ids = tok.encode(prompt)[: internals.MAX_SEQ]
    return [
        {"id": int(tid), "text": tok.decode([tid]), "i": i}
        for i, tid in enumerate(ids)
    ]


def _apply_sampling(logits_row: np.ndarray, temperature: float, top_k: int) -> np.ndarray:
    """Return the processed logits the sampler actually sees (temperature + top-k mask)."""
    t = max(0.05, float(temperature))
    scaled = logits_row / t
    k = max(1, int(top_k))
    if k < scaled.size:
        cutoff = np.sort(scaled)[-k]
        scaled = np.where(scaled < cutoff, -1e9, scaled)
    return scaled


def generate_step(prompt: str, tier: str, temperature: float, top_k: int,
                  generated: list[int], head=None, focus_layer=None) -> dict:
    import torch

    tok, model = MANAGER.get(tier)
    prompt_ids = tok.encode(prompt)
    ids = (prompt_ids + list(generated))[: internals.MAX_SEQ]
    input_ids = torch.tensor([ids])

    out = model(
        input_ids,
        output_attentions=True,
        output_hidden_states=True,
        use_cache=False,
    )

    # hidden_states: tuple of (1, seq, dim) — index 0 is the embedding output
    hs = [h[0].numpy() for h in out.hidden_states]
    embeddings = hs[0]
    hiddens = hs[1:]                       # one per transformer layer

    # attentions: tuple of (1, heads, seq, seq) per layer
    attentions = [a[0].numpy() for a in out.attentions]

    # logits for the NEXT token = last position
    logits_raw = out.logits[0, -1].numpy()
    logits_proc = _apply_sampling(logits_raw, temperature, top_k)

    # sample one token from the processed distribution
    probs = internals.softmax(logits_proc)
    rng = np.random.default_rng()         # genuine sampling; temperature/top-k already applied
    next_id = int(rng.choice(len(probs), p=probs))
    sampled = {"id": next_id, "text": tok.decode([next_id])}

    tokens = [
        {"id": int(t), "text": tok.decode([t]), "i": i} for i, t in enumerate(ids)
    ]
    done = len(ids) >= internals.MAX_SEQ

    return internals.assemble_step(
        step=len(generated), tokens=tokens, embeddings=embeddings,
        attentions=attentions, hiddens=hiddens,
        logits_raw=logits_raw, logits_sampled=logits_proc, sampled=sampled,
        id_to_text=lambda i: tok.decode([i]),
        done=done, head=head, focus_layer=focus_layer,
    )
