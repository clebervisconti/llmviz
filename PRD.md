# LLMViz — Product Requirements (PRD)

> Sibling product to **NeuraNetViz**. Where NeuraNetViz shows how a neural network *classifies
> an image*, LLMViz shows how a **Large Language Model generates text**, end to end.
> Owner: Cleber Visconti. Audience: his students/alumni (non-experts).

## 1. Problem & goal
Students hear "LLM," "tokens," "attention," "parameters" but have no mental model of what
happens between typing a prompt and getting a reply. **Goal:** a single-screen, visual,
interactive web app that animates the whole pipeline on a *real* small LLM, so a beginner can
watch — and poke at — every stage.

## 2. Target user & context
- **Primary:** Cleber's students/alumni, little-to-no ML background.
- **Use:** in a lecture (projected, Cleber driving) and self-serve afterward (link shared).
- **Device:** desktop browser primarily; must degrade gracefully on mobile.
- **Language:** English UI (consistent with NeuraNetViz).

## 3. The core experience (happy path)
1. Student lands on a dark CYBERSPHERE-branded page; particle background; a prompt box.
2. Picks a **model size** (NANO / MICRO / SMALL) — sees param/layer/head counts change.
3. Types or picks a sample prompt, hits **GENERATE**.
4. The center pipeline animates the 9-step narrative (tokenize → embed → attention across
   layers → logits → sample → append → repeat), token by token.
5. Right panel shows live **attention heatmap/links** and the **next-token probability bars**.
6. Student drags **temperature** / **top-k** and re-runs; watches the distribution and the
   output change.
7. Hovers any stage for a one-sentence plain-English explanation.

## 4. Functional requirements
- **FR1** Model-size selector with ≥3 real tiers + a Demo (scripted) fallback. Switching tiers
  visibly changes the diagram (number of layer blocks/heads) and the output quality.
- **FR2** Prompt input + 4–6 preset sample prompts.
- **FR3** Tokenization view: token chips with IDs and a count.
- **FR4** Embedding view: 2D projection of token vectors.
- **FR5** Layer-by-layer animated flow through transformer blocks.
- **FR6** Attention visualization: heatmap (seq×seq) + bipartite head view; layer selector;
  mean-over-heads default with single-head option.
- **FR7** Next-token probability bars (top-k), top-1 highlighted.
- **FR8** Sampling controls: temperature + top-k sliders, showing the distribution *before and
  after* the knobs apply.
- **FR9** Generation: token-by-token **STEP** mode + **autoplay**; chosen token appends and the
  pipeline re-runs.
- **FR10** Tooltips on every stage with beginner explanations.
- **FR11** Status line narrating the pipeline; clear loading/working states.
- **FR12** Honors `prefers-reduced-motion`.

## 5. Non-functional requirements
- **NFR1** No build step; vanilla JS + SVG/Canvas; FastAPI backend (match NeuraNetViz).
- **NFR2** Runs within the 4GB-RAM CPU VPS budget (`MemoryMax=2G`, one inference at a time).
- **NFR3** A generation step on MICRO returns in a few seconds or less on the VPS; if not,
  default to NANO.
- **NFR4** CYBERSPHERE brand fidelity per `docs/UI-SPEC.md` (enforced at review).
- **NFR5** Per-step payload < ~50KB.
- **NFR6** Deployed at a subdomain behind Cloudflare + OLS, public, no login.
- **NFR7** Accessible: keyboard-operable controls, sufficient contrast, reduced-motion path.

## 6. Explicitly out of scope (v1)
- Chat/multi-turn conversation; system prompts; RAG/tools.
- Training visualization (NeuraNetViz territory) — LLMViz is inference-only.
- User accounts, history, saving sessions.
- Models > ~355M; GPU; fine-tuning.
- In-browser inference mode (candidate for v2).
- i18n / Portuguese UI (candidate for v2).

## 7. Success criteria
- A non-technical student can, unaided, explain in their own words what tokens, attention, and
  temperature do after 5 minutes with the app.
- Cleber can run the whole pipeline live in a lecture without it stalling or crashing the VPS.
- Visually indistinguishable as a sibling of NeuraNetViz (same brand family).
- The model-size selector makes "more parameters" tangible.

## 8. Resolved decisions (Cleber, 2026-06-13)
- **Subdomain:** `llmviz.cybersphere.com.br` (CYBERSPHERE brand domain; zone
  `cbb665dcd1447268f6fe8b0437d1fcc6`). Public, **no login** — note this zone also hosts AgentOS
  behind Cloudflare Access, but LLMViz must stay open (do NOT apply the Access gate to it).
- **Model tiers:** ship **all three** at launch — NANO (DistilGPT-2 82M), MICRO (GPT-2 124M,
  default live model), SMALL (GPT-2-medium 355M, lazy-load + unload). Benchmark SMALL on the VPS
  in Phase 4; if it breaches `MemoryMax`, keep it lazy and serialized.
- **DEMO is the default landing experience** — instant scripted animation on load; students opt
  into the live model. Chosen for lecture reliability.
