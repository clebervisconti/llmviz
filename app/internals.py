"""
internals.py — downsamplers that turn raw model internals into small, JSON-safe
payloads for the browser. numpy-only, so it works for BOTH the real torch path
(tensors converted to numpy) and the scripted DEMO path (numpy generated directly).

Every function here exists to keep per-step payloads tiny (target < ~50KB):
  - attention: seq×seq matrix quantized to uint8 + top-k links per query token
  - hidden states: projected to 2D via PCA (numpy SVD, no sklearn)
  - logits: only the top-k tokens, as probabilities

See docs/ARCHITECTURE.md §3 for the response contract these feed into.
"""
from __future__ import annotations

import numpy as np

MAX_SEQ = 64          # hard cap on prompt+generated tokens (bounds seq^2 growth)
TOP_LOGITS = 20       # how many next-token candidates to send
TOP_LINKS = 4         # strongest attention links kept per query token


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def quantize_u8(matrix: np.ndarray) -> list[list[int]]:
    """Normalize a 2D matrix to 0..255 uint8 for a canvas heatmap."""
    m = np.asarray(matrix, dtype=np.float32)
    lo, hi = float(m.min()), float(m.max())
    if hi - lo < 1e-9:
        norm = np.zeros_like(m)
    else:
        norm = (m - lo) / (hi - lo)
    return (norm * 255.0).round().astype(np.uint8).tolist()


def top_links(attn_row_matrix: np.ndarray, k: int = TOP_LINKS) -> list[dict]:
    """
    Given a seq×seq attention matrix (row = query token, col = key token),
    return the k strongest (from->to) links per query, for the bipartite head view.
    """
    m = np.asarray(attn_row_matrix, dtype=np.float32)
    seq = m.shape[0]
    links: list[dict] = []
    for q in range(seq):
        row = m[q]
        # causal: a token can only attend to itself and earlier tokens
        valid = row[: q + 1]
        if valid.size == 0:
            continue
        kk = min(k, valid.size)
        idx = np.argpartition(valid, -kk)[-kk:]
        idx = idx[np.argsort(valid[idx])[::-1]]
        for j in idx:
            w = float(valid[j])
            if w > 0.001:
                links.append({"from": int(q), "to": int(j), "w": round(w, 3)})
    return links


def pack_attention(attn_layer: np.ndarray, head: int | None = None) -> dict:
    """
    attn_layer: array shaped (num_heads, seq, seq) for ONE transformer layer.
    Default view = mean over heads. If `head` is given, that single head's matrix.
    Returns {matrix_u8, top_links, mode}.
    """
    a = np.asarray(attn_layer, dtype=np.float32)
    if head is None:
        mat = a.mean(axis=0)          # mean over heads → (seq, seq)
        mode = "mean"
    else:
        head = max(0, min(head, a.shape[0] - 1))
        mat = a[head]
        mode = f"head{head}"
    return {
        "matrix_u8": quantize_u8(mat),
        "top_links": top_links(mat),
        "mode": mode,
    }


def pca_2d(hidden: np.ndarray) -> list[list[float]]:
    """
    hidden: (seq, dim) hidden states for one layer (or embeddings).
    Project to 2D with PCA via SVD. Returns one [x, y] per token, scaled to ~[-1, 1].
    """
    h = np.asarray(hidden, dtype=np.float32)
    if h.shape[0] == 1:
        return [[0.0, 0.0]]
    mean = h.mean(axis=0, keepdims=True)
    centered = h - mean
    # economy SVD; rows of Vt are principal directions
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        coords = centered @ vt[:2].T          # (seq, 2)
    except np.linalg.LinAlgError:
        coords = centered[:, :2]
    # scale to a tidy box for the scatter plot
    span = np.max(np.abs(coords)) or 1.0
    coords = coords / span
    return [[round(float(x), 3), round(float(y), 3)] for x, y in coords]


def pack_logits(logits_row: np.ndarray, id_to_text, k: int = TOP_LOGITS) -> list[dict]:
    """
    logits_row: 1D array over the vocab for the next-token position (pre-softmax).
    id_to_text: callable mapping a token id -> display string.
    Returns top-k [{id, text, p}] sorted by probability desc.
    """
    row = np.asarray(logits_row, dtype=np.float32)
    probs = softmax(row)
    kk = min(k, probs.size)
    idx = np.argpartition(probs, -kk)[-kk:]
    idx = idx[np.argsort(probs[idx])[::-1]]
    return [
        {"id": int(i), "text": id_to_text(int(i)), "p": round(float(probs[i]), 4)}
        for i in idx
    ]


def hidden_norm(hidden_layer: np.ndarray) -> float:
    """A single 0..1-ish scalar for a layer block's 'activation' glow."""
    h = np.asarray(hidden_layer, dtype=np.float32)
    # mean L2 norm of token vectors, squashed to a friendly range
    norms = np.linalg.norm(h, axis=-1)
    val = float(norms.mean())
    return round(float(np.tanh(val / 20.0)), 3)


def assemble_step(
    *,
    step: int,
    tokens: list[dict],
    embeddings: np.ndarray,          # (seq, dim) — the embedding layer output
    attentions: list[np.ndarray],    # one (heads, seq, seq) per transformer layer
    hiddens: list[np.ndarray],       # one (seq, dim) per transformer layer
    logits_raw: np.ndarray,          # (vocab,) pre-softmax for the next position
    logits_sampled: np.ndarray,      # (vocab,) after temperature/top-k processing
    sampled: dict,                   # {"id", "text"}
    id_to_text,
    done: bool,
    head: int | None = None,
    focus_layer: int | None = None,
) -> dict:
    """
    Build the /api/generate_step response from numpy internals. Shared by the real
    (torch) inference path and the scripted DEMO path so the frontend is identical.
    `head`/`focus_layer` optionally request a single head's full matrix for one layer
    (advanced attention view); all other layers stay mean-over-heads.
    """
    layers = []
    for li, (attn, hid) in enumerate(zip(attentions, hiddens)):
        want_head = head if (focus_layer is not None and li == focus_layer) else None
        layers.append(
            {
                "index": li,
                "attention": pack_attention(attn, head=want_head),
                "hidden_norm": hidden_norm(hid),
            }
        )
    return {
        "step": step,
        "caps": {"attention": True, "embeddings": True, "layers_static": False},
        "tokens": tokens,
        "embeddings_2d": pca_2d(embeddings),
        "layers": layers,
        "logits_raw": pack_logits(logits_raw, id_to_text),
        "logits_sampled": pack_logits(logits_sampled, id_to_text),
        "sampled": sampled,
        "done": done,
    }
