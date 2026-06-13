# Reference: NeuraNetViz (the sibling app)

> Path: `/Volumes/STORAGE/Users/clebervisconti/Library/Mobile Documents/com~apple~CloudDocs/Cyberlabs/Apps/NeuraNetViz`
> LLMViz must look and feel like the same product family. This doc summarizes what to reuse
> and what to adapt, so the development model (Opus) doesn't need to re-explore that repo.

## What NeuraNetViz is

Educational app that visualizes a CNN classifying animal images. FastAPI backend runs real
inference (TensorFlow/Keras: a from-scratch CNN + MobileNetV2), returns per-layer activations
+ heatmaps as JSON; a vanilla-JS frontend draws the network as SVG and animates the forward pass.

## Tech stack (proven pattern — reuse)

- **Backend**: FastAPI + Uvicorn, single `app/main.py` (~400 lines). Endpoints:
  `GET /api/architecture`, `POST /api/predict`, `GET /api/health`.
- **Frontend**: no framework, no build step. `static/index.html` + `style.css` (~490 lines)
  + `viz.js` (~680 lines, all rendering/interaction) + `bgfx.js` (canvas particle background).
- **Visualization**: SVG (960×600 viewBox) for the network; Canvas for heatmaps; CSS for bars.
- **Total**: ~1,775 LOC. LLMViz should land in the same ballpark (small, readable, educational).

## Files worth copying/adapting directly into LLMViz

| NeuraNetViz file | Reuse in LLMViz |
|---|---|
| `static/bgfx.js` | Copy as-is (particle constellation background). |
| `static/style.css` | Start from it: keep `:root` variables, topbar, panels, sliders, tooltip, buttons; replace center-stage classes. |
| `static/index.html` | Same skeleton: topbar / 3-panel main / footer. |
| `app/main.py` | Same shape: FastAPI app, static mount, `/api/architecture`, `/api/health`; replace predict with generate/step endpoints. |
| `deploy/neuralnetviz.service` | Template for `deploy/llmviz.service` (uvicorn on a dedicated localhost port, systemd, Restart=on-failure). |

## Interaction/animation grammar to mirror

- Forward pass animates **layer by layer (~320ms each)**: edges pulse with a green gradient,
  nodes flash and settle; total run a few seconds — long enough to follow, short enough to repeat.
- Node visual encoding: size + color intensity = activation strength; text inside node = value.
- Hover tooltips on every element with shape/stats + a one-sentence concept explanation.
- Right panel shows ranked softmax bars (top-1 highlighted green, 0.4s width transition).
- Heatmap ramp: dark `#1e1e1e` → green `#28d600` per-channel normalized.
- Status line narrates the pipeline ("encoding image…" → "running forward pass…" → result).
- Live controls (sliders) re-render the SVG without re-running inference.

## Deployment pattern (reuse verbatim, new port/subdomain)

- systemd service runs `uvicorn app.main:app --host 127.0.0.1 --port 8801` (NeuraNetViz).
  LLMViz takes the **next free port (e.g. 8802)**.
- OpenLiteSpeed reverse-proxies `neuralnetviz.clebervisconti.com → 127.0.0.1:8801`;
  LLMViz gets `llmviz.clebervisconti.com → 127.0.0.1:8802` (or a cybersphere.com.br subdomain).
- Cloudflare proxied DNS in front; short cache TTL on static, no-cache on index.html.
- Code lives on GitHub; VPS pulls (cron or manual) — see `cs-app-deployment` skill.

## Key differences LLMViz must handle (vs NeuraNetViz)

1. **Input is text, not an image** → no upload/dropzone; prompt textarea + sample prompts.
2. **Generation is iterative** (token by token), not a single forward pass → needs a
   step/auto-play loop and a streaming or per-step API.
3. **Payloads are bigger** (attention matrices per layer/head) → backend must downsample/
   summarize (top-k attention, mean-over-heads options) before sending JSON.
4. **Model size selector is a real feature** (NANO/MICRO/SMALL), not just a mode query param —
   it is the main pedagogical lever ("what do more parameters buy you?").
