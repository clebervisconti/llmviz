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

## Phase 5 — Polish & teaching pass  *(in progress)*
- [x] **Embeddings stage made teachable** (2026-06-13): each token shown as a labeled point in
      "meaning space" with cross-hair axes + caption; tooltip explains "token → vector of N
      numbers". Honest caption: "illustrative (DEMO)" vs "the model's learned meaning space" on
      live tiers. Verified live (DistilGPT-2 shows real clustering, e.g. cat≈sat).
- [x] **Cache-control fix**: `index.html` now `no-store` so asset-version bumps deploy instantly
      (was being cached 7 days by OLS/CF).
- [x] **Model-fit re-verified** for the micro VPS: 8-step live NANO load test held at ~760MB
      (peak 839MB ≪ 2G cap), 0 OOM restarts, other sites unaffected. NANO is the safe default;
      MICRO also fits; SMALL stays scripted.
- [x] reduced-motion guards in place; controls are native (keyboard-accessible); green-on-dark
      is high-contrast. Responsive grid stacks at ≤1100px.
- [ ] README screenshots; short "how to use in a lecture" note.
- [ ] Optional: cron git-pull on the VPS or Stop-hook auto-sync (AgentOS pattern).
- **Done when:** Cleber signs off after a dry-run lecture.

## Backlog / v2 ideas
- In-browser inference mode (transformers.js) for zero server load at scale.
- Portuguese (pt-BR) UI toggle.
- Compare two model sizes side by side on the same prompt.
- "Guided tour" overlay that walks a first-timer through the 9 steps.
- Save/share a specific prompt+settings as a URL.

## Phase 6 — Transformer-Explainer "inside one block" view  *(in progress, started 2026-06-27)*
Brings the visual language of poloclub.github.io/transformer-explainer into LLMViz (user request).
- [x] **Backend: real Q/K/V.** Forward hook on the focus layer's fused `c_attn` extracts genuine
      per-token Q/K/V; `internals.pack_qkv` downsamples to per-token norms + 8-bucket mini-vectors
      (uint8, focus layer only). `GenerateReq.want_qkv` gates it; DEMO synthesizes a matching shape
      (`demo_script._synth_qkv`). Verified shape + payload < 50KB across all tiers (demo/nano/micro/small).
- [x] **Frontend: `block_viz.js`** — expanded single block: token column → Q/K/V color strips →
      attention grid (Query·Key→softmax) → Out → MLP → residual, connected by Sankey ribbons; head
      selector, block navigator, "+N−1 more identical blocks" stacked-card motif. Overview⇄Block toggle.
      Plus continuously-flowing comet ribbons + hover path-tracing (reduced-motion-guarded).
- [x] **Frontend: probabilities polish + ribbon animation** (sampled-token tag, animated bars, inline
      temperature; all motion reduced-motion-guarded).
- **Deviation from frozen spec:** UI-SPEC §6 amended (new §6a) to allow a scoped Q/K/V hue triad
      (`--cv-q` violet / `--cv-k` amber / `--cv-v` green) **only** inside the attention sub-view —
      user-approved 2026-06-27. Rest of the app keeps the no-off-brand-colors rule.

## Phase 6.5 — Migrate to Mac mini + Gemma 3  *(done 2026-06-28)*
- [x] Real white-box models run locally (torch/MPS); the "static" feeling was the missing model
      (no torch → scripted DEMO). NANO/MICRO/SMALL (distilgpt2/gpt2/gpt2-medium) all live.
- [x] **Migrated off the VPS to the Mac mini**, served via the existing `agentos` cloudflared tunnel
      (`llmviz.cybersphere.com.br` → `localhost:8810`; DNS repointed from VPS A-record to tunnel CNAME).
      VPS `llmviz` systemd service stopped + disabled (port 8802 freed). See `deploy/MAC-MINI.md`.
- [x] **GEMMA tier = Gemma 3 4B** (`mlx-community/gemma-3-4b-it-4bit`) via a dedicated Python 3.11
      MLX venv (`mlx_lm 0.31.3`; Py3.9 capped at the buggy `mlx 0.29.x`). `mlx_backend.py` handles the
      new OpenAI logprobs API. Gemma 2 9B removed from cache. Black-box (tokens+probs, no attention).
- ⚠️ **Boot-persistence pending one manual step:** launchd is TCC-denied on `/Volumes/STORAGE/Cyberlabs`,
      so the installed launchd agents need a **Full Disk Access** grant on the Python binary to auto-start
      (steps in `deploy/MAC-MINI.md`). Until then the app + MLX run in-session.

## Phase 7 — Security hardening  *(planned — next release; from cs-red-team assessment 2026-06-28)*
Source: `outputs/cs-red-team/2026-06-28-110229/.../SECURITY-ASSESSMENT.md`. No injection/auth vulns found;
risk is operational (least-privilege + DoS + headers). **Context: the app now runs on the Mac mini, not the
VPS — this changes H1/I1 (see notes); the app-level fixes M1–L3 are host-independent and still apply.**

**P1 — host / least-privilege**
- [ ] **H1 (was High, now reduced):** the VPS root service is decommissioned. On the Mac mini the app runs
      as the `clebervisconti` user (not root) under launchd — already much better. Remaining: run under a
      dedicated low-priv user if feasible, and confirm the launchd job has no unnecessary privileges. The
      systemd hardening directives (NoNewPrivileges, ProtectSystem=strict, PrivateTmp, SystemCallFilter=
      @system-service, etc.) no longer apply (no systemd); the macOS equivalent is the FDA-scoped agent.

**P2 — app-level DoS & input (host-independent, do in the new build)**
- [x] **M1:** capped `GenerateReq.generated` with `max_length=MAX_SEQ` (now 422, not silent truncate) +
      an ASGI `/api/*` request-body cap of 64 KB (`MAX_API_BODY` → 413). *(done 2026-06-28; verified 422/413.)*
- [ ] **M2:** add per-IP rate limiting (slowapi/Starlette) on `/api/generate_step` + `/api/tokenize`;
      add a Cloudflare rate-limit rule for `/api/*`. (Origin is now a tunnel — not directly reachable, so
      CF-bypass risk is lower than the old VPS, but app-layer throttle is still defense-in-depth.)

**P3 — headers & info disclosure**
- [x] **M3:** security-headers middleware live (CSP scoped to `cdn.signalfx.com` + Splunk RUM connect-src
      + Google Fonts, `X-Content-Type-Options`, `X-Frame-Options: DENY`, `Referrer-Policy`, HSTS,
      `frame-ancestors 'none'`, `Permissions-Policy`). *(done 2026-06-28; verify CSP in a real browser console
      and tighten away from `'unsafe-inline'` next.)*
- [x] **L1:** 500/502 paths now return generic messages (`internal error` / `generation failed` /
      `tokenizer unavailable`); real exception goes to `logger.exception(...)` → Splunk APM. *(done 2026-06-28.)*
- [x] **L2:** `openapi_url=None` set; `/openapi.json` → 404. *(done 2026-06-28; verified.)*
- [x] **L3:** `/api/health` trimmed to `{"status":"ok"}`. *(done 2026-06-28; verified.)*

**P4 — monitoring / supply-chain (mostly accept + confirm)**
- [ ] **L4 (accepted):** Splunk RUM ingest token is browser-public by design; monitor `app=llmviz-browser`
      ingest volume for anomalies; rotate if abused.
- [ ] **I1 (changed):** the VPS root auto-deploy chain is gone (VPS decommissioned). The Mac mini serves
      from the local repo; re-evaluate the new deploy trust model (the `.claude` sync hook + how the Mac
      picks up changes). Keep branch protection + 2FA on the public GitHub repo.
- [ ] **deps:** pin torch/transformers (currently `>=`); add `pip-audit`/Dependabot. (Note: transformers
      was bumped to 4.53 + a Py3.11 `mlx_lm 0.31.3` venv added during the Gemma 3 switch — re-pin both.)
- **I2 (info, no action):** LLM prompt-injection impact is negligible — base GPT-2, no tools/agency; the
      GEMMA tier returns display-only text. (The separate Hermes-agent MLX proxy is where injection should
      be assessed — out of scope for LLMViz.)
