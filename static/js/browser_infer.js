/* browser_infer.js — in-browser inference via transformers.js (ONNX Runtime Web).
   Runs a tiny GPT-2 (distilgpt2) ENTIRELY in the browser — no server, no API.
   Loaded from CDN as an ES module (dynamic import, so no build step).

   Like the GEMMA tier, an inference runtime doesn't expose attention/hidden states,
   so caps.attention/embeddings = false (the white-box GPT-2 server tiers keep those).
   Exposes window.LLMViz.browser. */
(function () {
  const MODEL = "Xenova/distilgpt2";   // small, quantized ONNX — quick first download
  const CDN = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3";
  const LAYERS = 6;                     // distilgpt2 geometry (for the static layer blocks)

  let tjs = null, tok = null, model = null, loadingP = null;

  function softmaxTopK(row, k) {
    let mx = -Infinity;
    for (let i = 0; i < row.length; i++) if (row[i] > mx) mx = row[i];
    let sum = 0;
    const exp = new Float64Array(row.length);
    for (let i = 0; i < row.length; i++) { const e = Math.exp(row[i] - mx); exp[i] = e; sum += e; }
    // partial top-k by scanning (vocab ~50k, fine)
    const idx = Array.from({ length: row.length }, (_, i) => i);
    idx.sort((a, b) => exp[b] - exp[a]);
    const top = idx.slice(0, k);
    return top.map((i) => ({ id: i, p: exp[i] / sum }));
  }

  async function ensureLoaded(onProgress) {
    if (model) return;
    if (!loadingP) {
      loadingP = (async () => {
        const m = await import(CDN);
        m.env.allowLocalModels = false;          // always fetch from HF CDN
        m.env.useBrowserCache = true;            // cache the model in the browser after first load
        // Single-threaded WASM: threaded ORT needs cross-origin isolation (COOP/COEP) which
        // this page doesn't set; the worker's wasm fetch otherwise fails with "Failed to fetch".
        try {
          m.env.backends.onnx.wasm.numThreads = 1;
          m.env.backends.onnx.wasm.proxy = false;
        } catch (_) {}
        tok = await m.AutoTokenizer.from_pretrained(MODEL, { progress_callback: onProgress });
        model = await m.AutoModelForCausalLM.from_pretrained(MODEL, { dtype: "q8", progress_callback: onProgress });
        tjs = m;
      })();
    }
    await loadingP;
  }

  async function tokenize(text) {
    await ensureLoaded();
    const ids = tok.encode(text);
    return ids.slice(0, 64).map((id, i) => ({ id, text: tok.decode([id]), i }));
  }

  async function generateStep(prompt, generatedText, temperature, topk) {
    await ensureLoaded();
    const full = (prompt || "") + (generatedText || "");
    const enc = await tok(full);
    const out = await model(enc);
    const logits = out.logits;                   // Tensor [1, seq, vocab]
    const dims = logits.dims, vocab = dims[dims.length - 1], seq = dims[dims.length - 2];
    const data = logits.data;                    // flat Float32Array
    const last = new Float32Array(vocab);
    const base = (seq - 1) * vocab;
    const t = Math.max(0.05, temperature || 1);
    for (let i = 0; i < vocab; i++) last[i] = data[base + i] / t;

    const k = Math.max(5, Math.min(topk || 10, 20));
    const top = softmaxTopK(last, k);
    // renormalize over the visible top-k and decode
    const s = top.reduce((a, d) => a + d.p, 0) || 1;
    const dist = top.map((d) => ({ id: d.id, text: tok.decode([d.id]), p: +(d.p / s).toFixed(4) }));

    // sample from the top-k (temperature already applied)
    let r = Math.random() * s, pick = top[0].id;
    for (const d of top) { r -= d.p; if (r <= 0) { pick = d.id; break; } }
    const sampled = { id: pick, text: tok.decode([pick]) };

    // token chips for the current sequence
    const ids = tok.encode(full).slice(0, 64);
    const tokens = ids.map((id, i) => ({ id, text: tok.decode([id]), i }));

    const done = ids.length >= 64 || pick === tok.eos_token_id;
    return {
      step: tok.encode(generatedText || "").length,
      engine: "browser",
      caps: { attention: false, embeddings: false, layers_static: true },
      tokens,
      embeddings_2d: [],
      layers: Array.from({ length: LAYERS }, (_, i) => ({ index: i, hidden_norm: null })),
      logits_raw: dist,
      logits_sampled: dist,
      sampled,
      done,
    };
  }

  window.LLMViz = window.LLMViz || {};
  window.LLMViz.browser = { ensureLoaded, tokenize, generateStep, MODEL, LAYERS };
})();
