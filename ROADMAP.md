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

## Phase 4 — Deploy & harden  *(medium)* ✅ DONE — see `DEPLOYMENT.md`
- [x] `deploy/llmviz.service` (uvicorn :8802, `MemoryMax=2G`, Restart=on-failure, enabled at boot).
- [x] `gh repo create llmviz --public` and push → https://github.com/clebervisconti/llmviz
- [x] DNS: proxied A record `llmviz.cybersphere.com.br` → 129.121.49.96 (Cloudflare).
- [x] VPS: `/var/lib/llmviz/{repo,venv}`; CPU torch 2.8 + transformers 4.57 installed.
- [x] OLS reverse proxy via CyberPanel `createChild` + `issueSSL` (HTTP-01 cert), then vhost.conf
      rewritten to proxy → 127.0.0.1:8802. **NOT** the manual `httpd_config.conf` edit — CyberPanel
      did the risky config so the box's other 5 sites stayed safe.
- [x] **Benchmarked on the VPS** (in a memory-capped cgroup scope so it couldn't OOM the box):
      DistilGPT-2 (NANO) **654MB / 46ms**, GPT-2 (MICRO) **634MB / 452ms** per forward pass with
      attentions. Both enabled live; **SMALL (gpt2-medium) left scripted** — box too tight
      (only ~750MB free, swap full). `MAX_RESIDENT=1` (one model at a time) keeps LLMViz ≤~770MB.
- [x] Post-deploy checks: public `/api/health` green (`live:true`), live NANO+MICRO return real
      next-token distributions, DEMO + scripted SMALL work, NeuraNetViz + WordPress unaffected.
- **Done:** https://llmviz.cybersphere.com.br is live (DEMO default, real GPT-2 on NANO/MICRO). ✅

> ⚠ Operational notes for future edits:
> - The proxy lives in `/usr/local/lsws/conf/vhosts/llmviz.cybersphere.com.br/vhost.conf`
>   (backed up under `/home/backups/llmviz/`). If you re-issue SSL via the CyberPanel UI it may
>   rewrite this file — re-apply the proxy context/extprocessor afterward.
> - Cert auto-renews via CyberPanel/acme (acme-challenge context is in the vhost).
> - To change live tiers: edit `LLMVIZ_LIVE` / `LLMVIZ_LIVE_TIERS` in the systemd unit, `daemon-reload`, restart.
> - Update pipeline: `cd /var/lib/llmviz/repo && git pull && systemctl restart llmviz`.

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
