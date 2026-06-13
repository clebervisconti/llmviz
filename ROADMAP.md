# LLMViz — Build Roadmap

> Phased plan for the **development model (Opus)**. Check boxes as you go. Each phase ends in a
> verifiable state. Don't start a phase until the previous one's "Done when" is true.
> Read `CLAUDE.md` first, then `docs/ARCHITECTURE.md` + `docs/UI-SPEC.md`.

## Phase 0 — Setup & decisions  *(small)* ✅ DONE
- [x] Open questions resolved (`PRD.md §8`): subdomain `llmviz.cybersphere.com.br` (public, no
      Access gate); ship all three tiers; DEMO is the default landing experience.
- [x] `requirements.txt`: fastapi, uvicorn, numpy (DEMO); torch + transformers commented for VPS.
- [x] Local venv (fastapi/uvicorn/numpy only — torch deferred to VPS). Forward-pass shape
      verification deferred to Phase 4 on the VPS (torch not installed locally; the live path is
      written to the documented HF contract).
- [x] Copied `bgfx.js` + brand PNGs from NeuraNetViz; adapted `style.css` + `index.html`.
- **Done:** uvicorn serves the static page and `/api/health` returns ok. ✅

## Phase 1 — Backend core (real internals)  *(medium)* ✅ DONE
- [x] `model_manager.py`: tier registry, lazy load/cache, RESIDENT set, unload non-resident on
      lazy load; torch optional (DEMO works without it). Single-flight lock lives in `main.py`.
- [x] `inference.py`: real forward pass (`output_attentions`/`output_hidden_states`) + sampling;
      raw + processed next-token distributions.
- [x] `internals.py`: mean-over-heads, top-k links, uint8 quantize, numpy SVD PCA(2), top-20
      logits, shared `assemble_step`. seq ≤ 64 enforced.
- [x] Endpoints `/api/models`, `/api/tokenize`, `/api/generate_step`, `/api/health` per spec.
- [x] `demo_script.py` + DEMO mode through the same response shape; live tiers fall back to
      scripted (with `engine` tag) when torch is absent so the size selector always works.
- **Done:** verified via TestClient — `/api/generate_step` returns ~8KB downsampled payloads;
      all tiers route correctly; full run terminates. ✅

## Phase 2 — Frontend pipeline & branding  *(medium-large)* ✅ DONE
- [x] `index.html` 3-panel layout + `LLM`+green`VIZ` wordmark + model badge.
- [x] `controls.js`: segmented model control, temp/top-k sliders, sample prompts, GENERATE/STEP,
      autoplay loop, status line, DEMO auto-play on load.
- [x] `viz.js`: tokens → embedding scatter → N layer blocks → next-token chip; NeuraNetViz
      animation grammar; per-stage teaching tooltips.
- [x] Next-token probability bars (top-1 green, raw-ghost + sampled-fill, 0.4s transition).
- [x] Brand fidelity: shared CSS variables/fonts, green-as-signal, `prefers-reduced-motion` guards.
- **Done:** headless harness drives a full generate cycle with zero JS errors; pipeline + bars +
      text stream render from real backend data. ✅

## Phase 3 — Attention, embeddings & sampling depth  *(medium)* ✅ DONE
- [x] `attention.js`: seq×seq heatmap (uint8→canvas, Preto→Verde ramp); layer slider; mean-heads
      vs single-head (head 0) toggle; clicking a layer block focuses the attention view.
- [x] Embedding 2D scatter from server-side PCA.
- [x] Sampling "before vs after": raw ghost bar vs temperature/top-k-processed fill; live
      re-preview on slider change.
- [x] STEP + autoplay loop: append sampled token, re-run until done/limit; running text stream.
- [x] Model-size switch changes layer/head count in the diagram (verified: DEMO=6, MICRO=12).
- **Done:** the 9-step narrative is visualized; tiers change the picture; temperature reshapes
      the distribution. ✅

> Phases 0–3 built & verified 2026-06-13. Local run: `.venv/bin/uvicorn app.main:app --port 8810`
> then open http://127.0.0.1:8810 (DEMO auto-plays). Phase 4 (deploy) and Phase 5 (polish) remain.
> NOT YET DONE on a real model: install torch+transformers on the VPS and confirm live tiers +
> attention shapes + latency (Phase 4). Until then live tiers serve the scripted engine.

## Phase 4 — Deploy & harden  *(medium)* — see `DEPLOYMENT.md`
- [ ] `deploy/llmviz.service` (uvicorn :8802, `MemoryMax=2G`, Restart=on-failure).
- [ ] `gh repo create llmviz --public` and push.
- [ ] DNS via `cloudflare-manager`; VPS clone + CPU torch venv via `hostgator-vps-manager`;
      OLS reverse proxy + LE cert (DNS-01); systemd enable.
- [ ] **Benchmark on the VPS**: measure MICRO step latency + RSS; if too slow/heavy, set NANO
      as default and/or drop SMALL.
- [ ] Run `DEPLOYMENT.md` post-deploy checklist; add to AgentOS project registry.
- **Done when:** `https://<subdomain>/api/health` is green and a full lesson runs in the browser
      without stalling or breaching MemoryMax.

## Phase 5 — Polish & teaching pass  *(small)*
- [ ] Tooltip copy review for beginner clarity (align with `docs/RESEARCH.md §e` narrative).
- [ ] Mobile stacked layout check; keyboard/contrast/reduced-motion accessibility pass.
- [ ] README screenshots; short "how to use in a lecture" note.
- [ ] Optional: Claude Code Stop-hook auto-sync (AgentOS pattern) or cron git-pull on the VPS.
- **Done when:** Cleber signs off after a dry-run lecture.

## Backlog / v2 ideas
- In-browser inference mode (transformers.js) for zero server load at scale.
- Portuguese (pt-BR) UI toggle.
- Compare two model sizes side by side on the same prompt.
- "Guided tour" overlay that walks a first-timer through the 9 steps.
- Save/share a specific prompt+settings as a URL.
