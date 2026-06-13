# LLMViz — Start here (handoff to the development model)

You are picking up **LLMViz**, an educational web app that visually demonstrates how a Large
Language Model turns a prompt into a response. It is the **sibling of NeuraNetViz** and must
look like the same product family. The research & design are **done** (by Fable 5); your job is
to **build, deploy, and polish** it, cheaply and faithfully.

## Read in this order
1. **`PRD.md`** — what we're building and why; scope boundaries.
2. **`docs/RESEARCH.md`** — the research that justifies every decision (read at least the TL;DR
   + sections b, c, f).
3. **`docs/ARCHITECTURE.md`** — the concrete spec: API contract, modules, guardrails. *This is
   your blueprint.*
4. **`docs/UI-SPEC.md`** — frozen CYBERSPHERE visual spec. Do not invent colors/fonts.
5. **`docs/REFERENCE-NEURANETVIZ.md`** — what to copy/adapt from the sibling app.
6. **`ROADMAP.md`** — your phased checklist. Work the phases in order.
7. **`DEPLOYMENT.md`** — Phase 4 runbook.

## The 5 things that must stay true
1. **Real model, honest data.** Use real HuggingFace GPT-2-family inference on CPU and show its
   *real* attentions/logits. Simplify the *explanation*, never fake the *numbers*. (DEMO mode is
   the one allowed exception — it's clearly a scripted fallback.)
2. **No build step.** Vanilla JS + SVG/Canvas + FastAPI. No React/Vue/bundlers. Match NeuraNetViz.
3. **Brand fidelity.** CYBERSPHERE dark-first; green is a signal, never a fill; data in JetBrains
   Mono; every animation guarded by `prefers-reduced-motion`. See `docs/UI-SPEC.md`.
4. **The param selector is the headline feature.** Switching NANO/MICRO/SMALL must visibly change
   the diagram and the output. That's the whole pedagogical point.
5. **Respect the 4GB VPS.** Downsample every payload server-side; one inference at a time;
   `MemoryMax=2G`; lazy-load + unload SMALL; cap seq ≤ 64. Benchmark before trusting MICRO.

## Tools & infra you have (from the global CLAUDE.md credential map)
- **GitHub**: `gh` CLI authed as `clebervisconti`. Create the public repo `llmviz`.
- **VPS**: HostGator `129.121.49.96:22022`, key in Keychain `hostgator-vps-ssh-key`. Use the
  `hostgator-vps-manager` skill for server work, `cs-app-deployment` for the pipeline.
- **Cloudflare**: token `CF_API_TOKEN_ACCESS` in `~/.env.cloudflare`. Use `cloudflare-manager`
  for DNS. Zone `clebervisconti.com` id is in the global CLAUDE.md.
- **Branding**: the `cs-branding` skill is canonical; logos under
  `…/Cyberlabs/ai-workspace/shared/cs-branding/`. `docs/UI-SPEC.md` already distilled it.

## Conventions
- Keep it small and readable (~NeuraNetViz's ~1,800 LOC ballpark).
- No secrets in code or commits. No model files committed (HF caches on the VPS).
- Confirm destructive VPS/Cloudflare ops before running; snapshot before changing server files.
- Update `ROADMAP.md` checkboxes as you complete items; note any deviations from
  `docs/ARCHITECTURE.md` inline in that file so the design stays the source of truth.
- Port **8802** (8801 is NeuraNetViz). Subdomain **`llmviz.cybersphere.com.br`**, public/no
  login (that zone hosts AgentOS behind Access — do NOT gate LLMViz). Ship all three model tiers;
  DEMO is the default landing mode. (Decisions resolved 2026-06-13; see `PRD.md §8`.)

## First action
Start **Phase 0** in `ROADMAP.md`: confirm the open questions, set up `requirements.txt` + venv,
verify a real forward pass returns the expected attention/hidden-state shapes, and stand up the
static-serving FastAPI stub. Then proceed phase by phase.
