# LLMViz — Research Report

> Author: Fable 5 (research & design phase). Date: 2026-06-13.
> Audience: the development model (Opus) + Cleber.
> Method note: a deep-research fan-out (5 angles, 17 sources, 84 extracted claims) was run.
> The adversarial-verification pass could not complete (account session limit reset 11:10pm
> BRT — all verifier votes abstained, shown as "0-0", which is *not* a refutation). The claims
> below come from **primary sources** (project GitHub repos, HuggingFace official docs, the
> Transformer Explainer arXiv paper) and are independently consistent with established
> knowledge as of early 2026. Sources are cited inline. Treat license/version specifics as
> "verify at implementation time" (flagged ⚠).

---

## TL;DR — recommendation for v1

**Build a server-side FastAPI app running a real GPT-2 small (124M) on CPU, exposing real
attention + hidden states, with a vanilla-JS/SVG frontend — exactly the NeuraNetViz stack.**
Make the "model size selector" switch between **DistilGPT-2 (82M, 6 layers)**, **GPT-2 small
(124M, 12 layers)**, and optionally **GPT-2 medium (355M, 24 layers)** so students literally
see "more parameters = more layers/heads = different output." Keep a **fake/scripted demo
mode** as a fallback for offline classroom use and to guarantee the animation always plays.

Why server-side over browser-only: it matches the proven NeuraNetViz deployment, keeps the
frontend tiny and framework-free, and—critically—HuggingFace `transformers` makes real
attention/hidden-state extraction trivial (`output_attentions=True`), whereas transformers.js
does not cleanly expose those internals. The cost (CPU inference latency, ~hundreds of ms to a
few seconds per generation on a small prompt) is acceptable for a step-through teaching tool.

---

## (a) Prior art — what exists and what to borrow

| Tool | What it does | Stack | License | What to steal |
|---|---|---|---|---|
| **Transformer Explainer** (Georgia Tech Polo Club) [1][2][3] | Runs **real GPT-2 small (124M) live in the browser**; visualizes the full pipeline: tokenization → embeddings → positional encoding → Q/K/V → multi-head masked self-attention → MLP → softmax → sampling. Interactive **temperature, top-k, top-p**. | Svelte + D3 + ONNX Runtime Web; model = nanoGPT GPT-2 → ONNX. **Has a build step.** | MIT ⚠ (verify) | The *narrative ordering* and the temperature/top-k widgets. Best single reference for "what to show." |
| **Brendan Bycroft LLM Visualization** (bbycroft.net/llm) [4][5][6] | Gorgeous **3D walk-through** of GPT layer geometry. Primary live demo is a **tiny char-level model that sorts letters A,B,C** (from Karpathy minGPT), not a full LLM. Can render GPT-2/GPT-3 *geometry* without running the weights. | TypeScript + WebGL | **No license (all rights reserved)** [6] — ideas only, do **not** copy code. | The "data flowing through stacked layer blocks" spatial metaphor; the idea of a tiny toy model for a guaranteed-fast demo. |
| **Tiktokenizer** (dqbd) [10][11] | Live tokenization playground; exact token counts via `tiktoken`. | Next.js, WASM tiktoken | MIT ⚠ | The tokenization-step UI: colored token chips + IDs + count. Note: GPT-2 uses BPE; our tokenizer comes from the model. |
| **BertViz** (jessevig) [13] | Attention visualizer for HF models. Three views: **head view** (attention between tokens for selected heads), **model view** (all layers×heads grid), **neuron view** (Q·K detail). Jupyter-based. | Python + JS, in-notebook | Apache-2.0 ⚠ | The three canonical attention views. "Head view" (bipartite token-to-token curves) is the most intuitive for beginners. |
| **3Blue1Brown** transformer videos | Conceptual animation: embeddings as directions in space, attention as "tokens passing information," the unembedding/softmax. | — | © (inspiration only) | The *explanatory metaphors* and pacing for narration text. |

Aggregator of more tools: Awesome-Transformer-Visualization [7].

**Conclusion:** No existing tool both (a) matches your no-build vanilla stack and (b) is
freely licensed for direct code reuse. **Build fresh, borrow concepts.** Transformer Explainer
is the gold-standard reference for *what* to show; BertViz for *how* to show attention.

---

## (b) How to actually run the model — three options

### Option A — Server-side Python inference (RECOMMENDED for v1)
- PyTorch + HuggingFace `transformers`, CPU-only build, on the VPS.
- Real model, real internals. `output_attentions=True` / `output_hidden_states=True` give you
  everything the visualization needs in **one forward pass** (see section c).
- **Pros:** matches NeuraNetViz exactly; trivial internals extraction; tiny dependency-free
  frontend; model files cached server-side (no client download).
- **Cons:** CPU latency; server RAM/CPU load per request; needs concurrency guarding.
- **Verdict:** Best fit. The teaching tool is step-through, not real-time chat, so latency is fine.

### Option B — In-browser inference (transformers.js / ONNX Runtime Web / WebLLM)
- transformers.js runs HF models in-browser via **ONNX Runtime**; **WASM/CPU by default**,
  experimental **WebGPU** for speed [15][16]. Models must be **converted to ONNX** (via
  Optimum) [15].
- **Critical limitation:** the transformers.js public API does **not** cleanly expose
  `output_attentions`/`output_hidden_states` [17] — the exact data this app is built around.
  You'd be fighting the library to get attention matrices out.
- **Pros:** zero server inference cost; scales to any number of students; offline once loaded.
- **Cons:** ~hundreds of MB model download per student; attention/hidden-state extraction is
  not first-class; WebGPU support uneven; harder to debug.
- **Verdict:** Tempting for scale, but the internals-extraction gap makes it a poor v1 fit.
  Revisit for a v2 "runs entirely in your browser" mode if server load becomes a problem.

### Option C — Fake / scripted / toy model
- Precomputed outputs for a handful of canned prompts, OR a genuinely tiny char-level nanoGPT
  (Karpathy) trained on a toy task (like bbycroft's A,B,C sort), OR fully hand-authored fake
  attention matrices.
- **Pros:** instant, zero load, animation always works, perfect for offline classroom.
- **Cons:** not "real" — students aren't watching an actual LLM.
- **Verdict:** Ship this as a **fallback "Demo mode"** alongside Option A, not as the only mode.
  It guarantees the lesson runs even if the model server is down or slow, and lets you script a
  clean, didactic example. **Recommended as a complement, not a replacement.**

**Final model strategy:** Option A as primary, Option C as a built-in fallback toggle.

---

## (c) Extracting & serving internals from HuggingFace (the heart of the backend)

All from official HF docs [18][19][20]:

- **Attentions** (`output_attentions=True`): tuple, **one tensor per layer**, each shape
  `(batch, num_heads, seq_len, seq_len)`, **post-softmax probabilities** → directly usable as
  heatmaps. Payload scales as `num_layers × num_heads × seq_len²`.
  Example: GPT-2 small (12 layers × 12 heads) at 50 tokens ≈ **360k floats** (~1.4MB raw f32).
- **Hidden states** (`output_hidden_states=True`): tuple of **N+1 tensors** (embedding output
  + one per layer), each `(batch, seq_len, hidden_size)`. Gives both initial embeddings and
  every layer's activations from one pass.
- **Logits**: `(batch, seq_len, vocab_size)`, **pre-softmax**. Slice the last position →
  next-token distribution; **apply softmax yourself** to show probabilities.
- **During `generate()`** with `return_dict_in_generate=True`: per-step attentions/hidden
  states, one tuple element per generated token [20]. Also distinguishes `output_scores`
  (after temperature/top-k processors) from `output_logits` (raw) — **so you can show both the
  raw distribution and the effect of the sampling knobs.** This is pedagogically gold.

### Payload management (do NOT ship raw tensors to the browser)
The full attention payload is heavy. The backend must **downsample before serializing**:
1. **Mean-over-heads** option (1 matrix/layer) for the default view; full per-head on demand.
2. **Top-k attention** per query token (keep the few strongest links) for the bipartite view.
3. **Quantize** to uint8 (0–255) for heatmaps — exactly NeuraNetViz's heatmap approach.
4. **Project hidden states** to 2–3D (PCA, computed server-side) for the embedding view; ship
   only the projected coords, not the 768-dim vectors.
5. **Cap sequence length** (e.g. prompt + generated ≤ 64 tokens) to bound seq² growth.
6. Send **top-k logits** (e.g. top 20), not the full 50k vocab.

With these, per-step payloads drop to tens of KB — fine for a responsive UI.

---

## (d) Visualization techniques (maps to UI-SPEC.md)

- **Token flow through layers**: SVG columns (one per stage), green gradient edges pulsing
  layer-by-layer — reuse NeuraNetViz's forward-pass animation grammar.
- **Attention**: two views, à la BertViz —
  (1) **bipartite "head view"**: tokens listed left & right, curved links weighted by attention
  (top-k only); head/layer selectors.
  (2) **grid heatmap**: seq×seq matrix, Preto→Verde ramp, with a layer slider and
  mean-over-heads vs single-head toggle.
- **Embeddings**: 2D scatter of PCA-projected token vectors (server-computed); optionally show
  how they move between layers.
- **Logits / next token**: horizontal **probability bar race** (top-k), top-1 highlighted green
  — directly reuse NeuraNetViz's predictions panel.
- **Sampling widgets**: temperature slider + top-k slider; show the distribution **before and
  after** the knobs reshape it (using `output_logits` vs `output_scores`). This makes
  temperature *visceral*.
- **Generation**: token-by-token "step" button + autoplay; each accepted token flies back into
  the sequence and the pipeline re-runs.

## (e) Pedagogy — recommended teaching sequence

Beginner-friendly narrative (each step = one animated stage + 1–2 sentence tooltip):
1. **You type a prompt** → it's just text.
2. **Tokenization** → text is chopped into tokens (sub-words), each with an ID. *Show chips.*
3. **Embeddings** → each token becomes a vector (a point in "meaning space"). *Show vectors.*
4. **Positional info** → the model also encodes *where* each token is.
5. **Attention** → every token "looks at" other tokens to gather context. *Show links.* This is
   the key idea; spend the most time here.
6. **Layers stack** → this look-and-mix repeats across many layers; deeper = more abstract.
   *This is where the parameter-count selector matters: more layers/heads/width.*
7. **Prediction** → the final layer scores every possible next token (logits). *Show bars.*
8. **Sampling** → temperature/top-k pick one token from the distribution. *Show the knobs.*
9. **Repeat** → the chosen token is appended and the whole thing runs again — that's how text
   is generated one token at a time. *Loop the animation.*

**Acceptable simplifications for beginners:** treat positional encoding as "adds position info"
without the sinusoid math; show attention as mean-over-heads by default (per-head is advanced);
describe embeddings as "meaning space" via 2D projection; skip layernorm/residual details in
the main flow (mention in advanced tooltips); don't show the full 50k vocab. **Do NOT fake the
numbers** — use the real model's real outputs so the bars/heatmaps are honest; simplify the
*explanation*, not the *data*.

## (f) Sizing on a 4GB-RAM CPU VPS

The VPS (HostGator, ~4GB RAM, no GPU, shared with WordPress + NeuraNetViz) constrains model choice.

| Model | Params | Layers×Heads | Approx RAM (fp32 + activations) | Suitability |
|---|---|---|---|---|
| **DistilGPT-2** | 82M | 6 × 12 | ~0.5–1GB | ✅ "NANO" — fast, fewest layers, cleanest viz |
| **GPT-2 small** | 124M | 12 × 12 | ~0.8–1.5GB | ✅ "MICRO" — the sweet spot / default |
| **GPT-2 medium** | 355M | 24 × 16 | ~2–3GB | ⚠ "SMALL" — tight on 4GB; load on demand, unload after |
| GPT-2 large / TinyLlama / SmolLM-360M+ | 355M–1.1B | — | >3GB | ❌ risky on this box for v1 |

- TinyStories models (1M–33M) are an option for an even smaller/faster toy tier, but their
  vocab/quality make the logits view less recognizable to students — DistilGPT-2 is a better
  "small real model."
- **Latency**: CPU GPT-2-small generation of a short continuation is typically sub-second to a
  few seconds per request (HF/Optimum CPU benchmarks for gpt2 [22]); enabling
  `output_attentions` adds overhead but is fine for step-through use. ⚠ Benchmark on the actual
  VPS during Phase 4.
- **Operational guardrails** (also in DEPLOYMENT.md): `systemd MemoryMax=2G`; **load one model
  at a time**, lazy-load GPT-2 medium and free it after; serialize requests (single worker,
  small queue) so two students don't run inference simultaneously and OOM the box; cap tokens.

---

## Concrete v1 recommendation

1. **Stack:** FastAPI + Uvicorn backend, vanilla JS + SVG/Canvas frontend, no build step
   (clone NeuraNetViz's structure). Deploy Shape-B behind OLS + Cloudflare on port 8802.
2. **Model:** real HuggingFace GPT-2 family on CPU. Selector = **NANO (DistilGPT-2 82M) /
   MICRO (GPT-2 124M, default) / SMALL (GPT-2 medium 355M, lazy)**. Plus a **Demo (scripted)**
   fallback mode.
3. **Backend exposes:** `/api/models`, `/api/tokenize`, `/api/generate_step` (one token at a
   time, returns downsampled attentions + hidden-state PCA + top-k logits before/after
   sampling), `/api/health`. Downsample every payload (mean-over-heads default, top-k links,
   uint8 heatmaps, server-side PCA, top-20 logits, seq ≤ 64).
4. **Frontend teaches the 9-step narrative** with the CYBERSPHERE look (UI-SPEC.md), reusing
   NeuraNetViz's animation grammar, bgfx background, sliders, tooltips, and bar panel.
5. **Parameter selector is the headline feature** — switching tiers visibly changes the number
   of layer blocks/heads in the diagram and the quality of the output, answering "what do more
   parameters buy?"

---

## Sources

[1] https://github.com/poloclub/transformer-explainer
[2] https://poloclub.github.io/transformer-explainer/
[3] https://arxiv.org/abs/2408.04619 (Transformer Explainer paper) · https://arxiv.org/html/2408.04619v1
[4] https://bbycroft.net/llm · [5] https://github.com/bbycroft/llm-viz · [6] (repo license field = null → all rights reserved)
[7] https://github.com/Ki-Seki/Awesome-Transformer-Visualization
[10] https://github.com/dqbd/tiktokenizer · [11] (MIT ⚠)
[13] https://github.com/jessevig/bertviz
[15][16] https://huggingface.co/docs/transformers.js/index
[17] https://github.com/huggingface/transformers/issues/33996
[18] https://huggingface.co/docs/transformers/en/main_classes/output
[19] https://huggingface.co/docs/transformers/en/internal/generation_utils
[20] https://discuss.huggingface.co/t/.../61543 (generate attention output)
[21] https://www.kdnuggets.com/how-to-visualize-model-internals-and-attention-in-hugging-face-transformers
[22] https://huggingface.co/datasets/optimum-benchmark/cpu/.../gpt2/benchmark.json
[23] https://amanvir.com/gpt-2-attention

⚠ = verify exact license/version/latency at implementation time (Phase 0/4).
