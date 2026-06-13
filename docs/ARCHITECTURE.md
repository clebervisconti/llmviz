# LLMViz — Technical Architecture

> Decision record + concrete spec for the build. Derived from `RESEARCH.md`. Opus implements this.
> Guiding principle: **clone NeuraNetViz's architecture; swap CNN-image for LLM-text.**

## 1. Decisions (ADR-style summary)

| # | Decision | Why | Alternatives rejected |
|---|---|---|---|
| D1 | **Server-side inference** (FastAPI + PyTorch + HF transformers, CPU) | Matches NeuraNetViz; trivial internals extraction via `output_attentions`/`output_hidden_states`; tiny frontend | Browser (transformers.js) — can't cleanly expose attention; WebLLM — overkill |
| D2 | **Real GPT-2 family** as the models | "Real LLM," recognizable, small enough for 4GB CPU | Toy nanoGPT only — not a "real LLM"; big models — OOM risk |
| D3 | **3 size tiers + Demo mode**: NANO=DistilGPT-2(82M), MICRO=GPT-2(124M, default), SMALL=GPT-2-medium(355M, lazy) | Param selector is the headline teaching feature; Demo guarantees the lesson always runs | Single model — loses the pedagogy |
| D4 | **Vanilla JS + SVG/Canvas, no build step** | Identical to NeuraNetViz; zero toolchain; easy for Opus + future edits | React/Svelte — adds a build step, breaks the family pattern |
| D5 | **Downsample all internals server-side** before JSON | Raw attention is MBs/step; UI needs KBs | Ship raw tensors — too heavy |
| D6 | **Step-based generation API** (one token per call) + autoplay in frontend | Matches the teaching narrative; bounds payload; simple to animate | Server-sent streaming — more complex; fine as v2 |

## 2. Component diagram

```
┌─────────────────────────── Browser (vanilla JS) ───────────────────────────┐
│ index.html · style.css · bgfx.js (bg) · viz.js (pipeline SVG + animation)   │
│ · attention.js (heatmap+head view) · controls.js (model/temp/topk/run/step) │
│        │  fetch JSON                                   ▲ render               │
└────────┼───────────────────────────────────────────────┼────────────────────┘
         ▼                                                 │
┌─────────────────────────── FastAPI (app/main.py) ───────────────────────────┐
│ /api/models  /api/tokenize  /api/generate_step  /api/health                  │
│        │                                                                      │
│        ▼  model_manager.py (lazy load/unload, single-flight lock)            │
│   HF transformers (PyTorch CPU): DistilGPT-2 / GPT-2 / GPT-2-medium          │
│        │ forward(output_attentions, output_hidden_states)                    │
│        ▼ internals.py (downsample: mean-heads, top-k links, uint8, PCA)      │
└──────────────────────────────────────────────────────────────────────────────┘
```

## 3. Backend API contract

> All responses JSON. All heavy arrays downsampled. seq capped at 64 (prompt+generated).

### `GET /api/models`
```json
{ "models": [
  {"id":"nano","label":"NANO","hf":"distilgpt2","params":"82M","layers":6,"heads":12,"dim":768},
  {"id":"micro","label":"MICRO","hf":"gpt2","params":"124M","layers":12,"heads":12,"dim":768,"default":true},
  {"id":"small","label":"SMALL","hf":"gpt2-medium","params":"355M","layers":24,"heads":16,"dim":1024,"lazy":true},
  {"id":"demo","label":"DEMO","params":"scripted"}
] }
```

### `POST /api/tokenize`  → `{prompt, model}`
```json
{ "tokens":[{"id":15496,"text":"Hello","i":0}, ...], "count": 4 }
```

### `POST /api/generate_step`  → `{prompt, model, temperature, top_k, generated:[ids so far]}`
Runs one forward pass over prompt+generated, returns next-token distribution + internals for the
**current last position**. Frontend appends the sampled token and calls again for autoplay.
```json
{
  "step": 3,
  "tokens": [ {"id":..,"text":"..","i":0}, ... ],         // full current sequence
  "embeddings_2d": [[x,y], ...],                            // PCA(2) per token, server-computed
  "layers": [                                               // one per transformer layer
    { "index":0,
      "attention": { "matrix_u8": [[..0-255..], ...],       // mean-over-heads, seq×seq, uint8
                     "top_links":[{"from":3,"to":0,"w":0.42}, ...] },  // top-k per query
      "hidden_norm": 0.83 }                                 // scalar activation magnitude for block glow
  ],
  "logits_raw":   [{"id":314,"text":" I","p":0.21}, ... up to 20],   // softmax(raw logits), top-20
  "logits_sampled":[{"id":314,"text":" I","p":0.34}, ... up to 20],  // after temp+top_k (output_scores)
  "sampled": {"id":314,"text":" I"},
  "done": false
}
```
Optional `GET /api/generate_step?head=H&layer=L` style params (or include in POST body) to fetch
a single head's full matrix for the advanced attention view.

### `GET /api/health` → `{"status":"ok","models_loaded":["micro"],"mem_mb":NNN}`

## 4. Backend module layout (`app/`)
- `main.py` — FastAPI app, static mount, route handlers, request validation, single-flight lock.
- `model_manager.py` — lazy load/cache models; unload SMALL after use; enforce one-at-a-time.
- `inference.py` — run forward pass with `output_attentions=True, output_hidden_states=True`;
  sampling (temperature, top-k) producing both raw and processed distributions.
- `internals.py` — downsamplers: mean-over-heads, top-k links, uint8 quantize, PCA(2/3) via
  numpy (no sklearn needed — `np.linalg.svd`), top-k logits.
- `demo_script.py` — canned step sequence for DEMO mode (a fixed prompt → fixed nice outputs).

## 5. Frontend module layout (`static/`)
- `index.html` — topbar / 3-panel / footer skeleton (from NeuraNetViz).
- `style.css` — start from NeuraNetViz; keep `:root`, topbar, panels, sliders, tooltip, buttons.
- `bgfx.js` — copy verbatim from NeuraNetViz (particle background).
- `viz.js` — center pipeline SVG: token chips → embedding strip → N layer blocks → logit bars →
  sampled token; layer-by-layer animation; status line; tooltips.
- `attention.js` — right-panel attention: seq×seq heatmap (uint8 → canvas, Preto→Verde ramp) +
  bipartite head view (top_links as SVG curves); layer slider + mean/single-head toggle.
- `controls.js` — model-size segmented control, temperature & top-k sliders, GENERATE/STEP
  buttons, sample-prompt buttons; orchestrates the step→append→repeat loop and autoplay timer.

## 6. Performance & safety guardrails
- Single Uvicorn worker; in-process asyncio lock so only one inference runs at a time (prevents
  OOM from concurrent students). Short queue; return 429 if backed up.
- `MemoryMax=2G` in systemd. NANO/MICRO stay resident; SMALL lazy-loads then unloads.
- seq ≤ 64 tokens; logits top-20; attention mean-over-heads by default; full per-head only on
  explicit request for one (layer, head).
- All payloads target < ~50KB/step. Benchmark on the VPS in Phase 4; if MICRO is too slow,
  default to NANO.
- `prefers-reduced-motion` honored in all frontend animation.

## 7. Repo layout (mirrors NeuraNetViz)
```
LLMViz/
├── README.md
├── CLAUDE.md                # handoff/instructions for the dev model
├── PRD.md
├── ROADMAP.md
├── DEPLOYMENT.md
├── requirements.txt         # fastapi, uvicorn, torch(cpu), transformers, numpy
├── docs/ (RESEARCH, ARCHITECTURE, UI-SPEC, REFERENCE-NEURANETVIZ)
├── app/ (main.py, model_manager.py, inference.py, internals.py, demo_script.py)
├── static/ (index.html, style.css, bgfx.js, viz.js, attention.js, controls.js, brand/*.png)
└── deploy/ (llmviz.service)
```
No model files committed — HF downloads & caches on first run on the VPS.
