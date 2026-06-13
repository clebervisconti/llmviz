# LLMViz

**Watch a Large Language Model think.** LLMViz is an educational web app that animates the full
journey of a prompt through a real (small) LLM: tokenization → embeddings → attention across
transformer layers → next-token probabilities → sampling → the response, generated one token at
a time. Built for students, in plain English, on the CYBERSPHERE brand.

Sibling app to **[NeuraNetViz](../NeuraNetViz)** (which visualizes a neural network classifying
images). Same look, same stack — different lesson.

## What makes it teach well
- **Pick a model size** (NANO / MICRO / SMALL) and *see* what more parameters buy you — more
  layers, more attention heads, better output.
- **Real model, honest internals.** It runs an actual GPT-2-family model and shows its real
  attention weights and token probabilities — not a cartoon.
- **Poke at it.** Drag temperature and top-k and watch the next-token distribution reshape.
- **Step or autoplay** through generation, token by token.
- **Every stage explains itself** with a one-sentence, beginner-friendly tooltip.

## Stack
FastAPI + PyTorch (CPU) + HuggingFace `transformers` backend · vanilla JS + SVG/Canvas frontend
(no build step) · deployed behind OpenLiteSpeed + Cloudflare on the HostGator VPS. Mirrors
NeuraNetViz exactly.

## Status
📐 **Design complete (research + PM docs). Development not started.**
Built by Cleber Visconti with Claude Code — researched/designed with Fable 5, developed with Opus.

## Docs map
| Doc | Purpose |
|---|---|
| [CLAUDE.md](CLAUDE.md) | **Start here** — handoff for the development model |
| [PRD.md](PRD.md) | Product requirements & scope |
| [ROADMAP.md](ROADMAP.md) | Phased build checklist |
| [DEPLOYMENT.md](DEPLOYMENT.md) | VPS + Cloudflare deploy runbook |
| [docs/RESEARCH.md](docs/RESEARCH.md) | The research behind every decision |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Technical blueprint (API, modules, guardrails) |
| [docs/UI-SPEC.md](docs/UI-SPEC.md) | Frozen CYBERSPHERE visual spec |
| [docs/REFERENCE-NEURANETVIZ.md](docs/REFERENCE-NEURANETVIZ.md) | What to reuse from the sibling app |

## Local development (once built)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload   # http://localhost:8000
```

---
© Cleber Visconti · CYBERSPHERE
