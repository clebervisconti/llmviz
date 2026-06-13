# LLMViz — UI & Branding Specification

> Visual identity: **CYBERSPHERE** (`cs-branding` skill is the canonical source).
> Visual sibling: **NeuraNetViz** (`../NeuraNetViz`) — LLMViz must feel like the same family of apps.
> Status: FROZEN for v1. Opus should implement exactly this; do not invent new colors or fonts.

## 1. Brand foundation

Dark-first. Background is Preto Neural, text is white/muted, and Verde Ascensão is a
*signal* (glow, stroke, edge, particle) — never a large solid fill.

### CSS variables (copy verbatim into `static/style.css`)

```css
:root {
  /* Surfaces */
  --cv-bg:           #1e1e1e;   /* Preto Neural — page background */
  --cv-bg-elevated:  #262626;   /* panels */
  --cv-bg-overlay:   #2f2f2f;   /* tooltips, popovers */

  /* Accent — Verde Ascensão */
  --cv-accent:       #28d600;
  --cv-accent-hover: #34f000;
  --cv-accent-muted: rgba(40, 214, 0, 0.15);
  --cv-glow: 0 0 14px rgba(40, 214, 0, 0.55), 0 0 36px rgba(40, 214, 0, 0.25);

  /* Text */
  --cv-text:         #ffffff;
  --cv-text-muted:   rgba(255, 255, 255, 0.70);
  --cv-text-faint:   rgba(255, 255, 255, 0.45);

  /* Lines */
  --cv-border:       rgba(255, 255, 255, 0.12);

  /* State (used sparingly) */
  --cv-warn:         #f0b400;   /* amber — warnings/hypothesis only */
}
```

These are the **same variable names and values as NeuraNetViz** — intentional, so the two
apps are visually interchangeable.

### Typography

| Role | Font | Weights | Notes |
|---|---|---|---|
| Headings / UI labels | Plus Jakarta Sans | 300–800 | uppercase + letter-spacing for section labels |
| Body | Poppins | 300–700 | |
| Data readout (tokens, probabilities, IDs, shapes) | JetBrains Mono | 400, 600 | every numeric/token value on screen uses mono |

```css
@import url("https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Poppins:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap");
```

Token chips, probability percentages, layer shapes, token IDs → **always JetBrains Mono**.
This is the "sensor readout" aesthetic; it also visually separates *data* from *explanation*.

## 2. Page layout (mirror NeuraNetViz)

```
┌────────────────────────────────────────────────────────────────────┐
│ Topbar (sticky, 60px): logo · LLM**VIZ** wordmark · nav · model badge │
├──────────────┬───────────────────────────────────┬─────────────────┤
│ Left panel   │ Center stage                      │ Right panel     │
│ 320px        │ 1fr — the transformer pipeline    │ 340px           │
│              │ (SVG, 16:10 min 360px)            │                 │
│ · prompt box │                                   │ · next-token    │
│ · sample     │ tokens → embeddings → N layer     │   probability   │
│   prompts    │ blocks → logits → sampled token   │   bars          │
│ · model size │                                   │ · attention     │
│   selector   │                                   │   heatmap       │
│ · temperature│                                   │ · generated     │
│ · top-k      │                                   │   text stream   │
│ · run/step   │                                   │                 │
├──────────────┴───────────────────────────────────┴─────────────────┤
│ Footer: © Cleber Visconti · CYBERSPHERE · GitHub link              │
└────────────────────────────────────────────────────────────────────┘
```

- Wordmark in topbar: `LLM` in white + `VIZ` in `--cv-accent` (NeuraNetViz uses NEURA**NET**VIZ).
- Model badge top-right shows active model (e.g. `GPT-2 · 124M params`), with tooltip.
- Background: reuse `bgfx.js` particle constellation from NeuraNetViz **as-is**
  (~30–50 nodes, green lines `rgba(40,214,0,0.22)`, dots `rgba(40,214,0,0.65)`, z-index −1).
- Mobile (<900px): stack panels vertically — prompt, pipeline, results.

## 3. Center-stage visualization rules

The pipeline diagram is **SVG** (like NeuraNetViz's network diagram), one column per stage:

| Stage | Visual | Color |
|---|---|---|
| Input text | text chip | white |
| Tokens | rounded chips, one per token, mono font, alternating tints | white chips, green border on hover |
| Embeddings | small vector bars / mini-heat strip per token | green ramp |
| Transformer layer blocks (×N) | rounded-rect blocks with "ATTN + MLP" sublabel | `--cv-accent` at varying opacity by activation |
| Logits / softmax | top-k bar chart | top-1 in solid green text, bars in green fill `--cv-accent-muted` |
| Sampled token | chip that flies back to the sequence | green glow pulse |

Animation grammar (same as NeuraNetViz forward pass):
- Edges between stages pulse with a green linearGradient, ~300–350ms per stage.
- Active block scales 1.0→1.06 with `--cv-glow`, then settles.
- Heatmap color ramp: **Preto Neural → Verde Ascensão** interpolation, exactly NeuraNetViz's
  ramp: `R 30→40, G 30→214, B 30→0`.
- Every animated element respects `prefers-reduced-motion: reduce` (fall back to instant state).

## 4. Component inventory

- **Prompt box**: textarea, mono, dark `--cv-bg-elevated`, green focus ring + inset glow
  (same treatment as NeuraNetViz dropzone hover).
- **Sample prompt buttons**: 4–6 preset prompts (like the 6 sample-animal buttons).
- **Model size selector**: segmented control (e.g. `NANO · MICRO · SMALL`), shows param count
  + layer/head/dim breakdown underneath in mono.
- **Sliders**: temperature (0.1–2.0) and top-k (1–50) — same slider styling as NeuraNetViz's
  node-size/contrast sliders.
- **Run / Step controls**: `GENERATE` primary button (pill, green bg `#28d600`, dark text
  `#06200a`, uppercase, glow on hover) + `STEP` secondary (outline) for token-by-token mode.
- **Tooltips**: dark overlay `--cv-bg-overlay`, arrow, clamped to viewport; every node/block
  hover explains the concept in 1–2 sentences (this is the core teaching device).
- **Status line**: `awaiting prompt…` → `tokenizing…` → `running layer 3/12…` →
  `sampling next token…` → `done · 14 tokens · 1.2s`.

## 5. Logo & assets

Logos live at:
`/Volumes/STORAGE/Users/clebervisconti/Library/Mobile Documents/com~apple~CloudDocs/Cyberlabs/ai-workspace/shared/cs-branding/`
- Topbar: `Icone fundo preto.png` (icon-only, ≥40px) + CSS wordmark.
- Favicon: derive from icon PNG.
- Never recolor/distort; min padding = ¼ logo width.
- NeuraNetViz also ships brand PNGs in its `static/` — copy the same ones for consistency.

## 6. Dos & don'ts (enforced at review)

- ✅ Green = glow/stroke/edge/text accent. ❌ Never large solid green surfaces.
- ✅ All data values in JetBrains Mono. ❌ No prose in mono.
- ✅ `prefers-reduced-motion` guard on every animation. ❌ No unguarded animations.
- ✅ Vanilla JS + SVG/Canvas, no build step. ❌ No React/Vue/bundlers.
- ✅ Border radius ≥10px. ❌ No off-brand colors (blues/purples/reds) anywhere.
- ✅ English UI text (matching NeuraNetViz). Keep sentences short — audience is students.
