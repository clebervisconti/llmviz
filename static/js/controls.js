/* controls.js — orchestration: model selector, sampling sliders, sample prompts,
   Generate/Step loop, and rendering of pipeline + next-token bars + attention + text.
   Talks to the FastAPI backend; viz.js and attention.js do the drawing. */
(function () {
  const $ = (id) => document.getElementById(id);
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const SAMPLES = [
    "The cat sat on the",
    "Once upon a time, in a land",
    "The capital of France is",
    "To make a good cup of coffee, you",
    "In the year 2050, computers will",
  ];

  const state = {
    models: [], model: "demo", temp: 0.8, topk: 10,
    prompt: "The cat sat on the",
    generated: [], lastStep: null, running: false, runToken: 0,
    attnLayer: 0, attnMode: "mean", // "mean" | "single"
  };

  // ---------- API ----------
  async function api(path, body, _tries) {
    const res = await fetch(path, {
      method: body ? "POST" : "GET",
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
    // transient "busy/warming up" — wait a beat and retry rather than erroring at the user
    if ((res.status === 429 || res.status === 503) && (_tries || 0) < 3) {
      status("warming up the model…", "busy");
      await sleep(1500);
      return api(path, body, (_tries || 0) + 1);
    }
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
    return res.json();
  }

  function stepBody(extra) {
    return Object.assign({
      prompt: state.prompt, model: state.model,
      temperature: state.temp, top_k: state.topk,
      generated: state.generated.slice(),
      generated_text: genText.join(""),   // used by the GEMMA/MLX tier
    }, extra || {});
  }

  // ---------- status ----------
  function status(msg, cls) {
    const s = $("status"); s.textContent = msg;
    s.className = "status" + (cls ? " " + cls : "");
  }

  // ---------- model selector ----------
  function buildModels(models) {
    state.models = models;
    const seg = $("model-seg"); seg.innerHTML = "";
    models.forEach((m) => {
      const b = document.createElement("button");
      b.textContent = m.label;
      // Never disable: live tiers fall back to the scripted engine when torch is absent,
      // so the size selector (the headline teaching feature) always changes the diagram.
      b.title = m.available ? m.blurb : m.blurb + " — runs as a scripted preview until the live model server is enabled";
      if (m.id === state.model) b.classList.add("active");
      b.addEventListener("click", () => selectModel(m.id));
      seg.appendChild(b);
    });
    selectModel(state.model);
  }

  function selectModel(id) {
    state.model = id;
    [...$("model-seg").children].forEach((b, i) =>
      b.classList.toggle("active", state.models[i] && state.models[i].id === id));
    const m = state.models.find((x) => x.id === id);
    if (m) {
      $("model-meta").innerHTML =
        `<b>${m.label}</b> · ${m.params} params<br>${m.layers} layers · ${m.heads} heads · dim ${m.dim}`;
      const badge = $("model-badge");
      badge.textContent = `${m.label} · ${m.params}`;
      badge.title = m.blurb;
      badge.classList.toggle("scripted", m.id === "demo" || !m.available);
    }
    // reset run for the new geometry
    resetRun();
  }

  // ---------- rendering ----------
  function renderPredictions(data) {
    const ul = $("predictions"); ul.innerHTML = "";
    const sampled = data.logits_sampled || [];
    const rawMap = {}; (data.logits_raw || []).forEach((d) => (rawMap[d.id] = d.p));
    if (!sampled.length) { ul.innerHTML = '<li class="empty">run to see the next-token distribution</li>'; return; }
    sampled.forEach((d, i) => {
      const li = document.createElement("li");
      if (i === 0) li.className = "top";
      const raw = rawMap[d.id] || 0;
      li.innerHTML =
        `<span class="label">${escapeHtml(d.text)}</span>` +
        `<span class="bar"><span class="raw" style="width:${(raw * 100).toFixed(1)}%"></span>` +
        `<span class="samp" style="width:${(d.p * 100).toFixed(1)}%"></span></span>` +
        `<span class="pct">${(d.p * 100).toFixed(0)}%</span>`;
      ul.appendChild(li);
    });
  }

  function renderAttentionControls(data) {
    const n = (data.layers || []).length;
    const sl = $("attn-layer");
    sl.max = Math.max(0, n - 1);
    if (state.attnLayer > n - 1) state.attnLayer = n - 1;
    sl.value = state.attnLayer;
    $("attn-layer-out").textContent = state.attnLayer;
    const layer = (data.layers || [])[state.attnLayer];
    if (layer) window.LLMViz.attention.render($("attn-heatmap"), layer);
  }

  function renderTextStream() {
    const el = $("text-stream");
    const gen = state.lastStep ? collectGenText() : "";
    el.innerHTML = `<span class="prompt-part">${escapeHtml(state.prompt)}</span>` +
      `<span class="gen-part">${escapeHtml(gen)}</span>` +
      (state.running ? '<span class="cursor">▌</span>' : "");
  }

  // accumulate generated token texts (we stored them during the run)
  const genText = [];
  function collectGenText() { return genText.join(""); }

  function renderAll(data, animate) {
    const svg = $("pipeline");
    const m = state.models.find((x) => x.id === state.model);
    if (m) data.dim = m.dim;   // let the embeddings tooltip name the real vector size
    const groups = window.LLMViz.pipeline.render(svg, data);
    renderPredictions(data);
    // some backends (GEMMA/MLX) can't expose attention — hide the panel, show the note
    const caps = data.caps || { attention: true, embeddings: true };
    $("attn-section").style.display = caps.attention ? "" : "none";
    $("attn-note").style.display = caps.attention ? "none" : "";
    if (caps.attention) renderAttentionControls(data);
    renderTextStream();
    $("step-counter").textContent = data.tokens ? `${data.tokens.length} tokens · step ${data.step + 1}` : "";
    if (animate) return window.LLMViz.pipeline.animate(groups);
    return Promise.resolve();
  }

  // ---------- generation ----------
  function resetRun() {
    state.generated = []; state.lastStep = null; genText.length = 0; state.running = false;
    $("predictions").innerHTML = '<li class="empty">run to see the next-token distribution</li>';
    $("step-counter").textContent = "";
    renderTextStream();
    // draw an initial (step-0) frame so the pipeline isn't empty
    doStep(false, true).catch(() => {});
  }

  async function doStep(animate, silent) {
    try {
      if (!silent) status("running forward pass…", "busy");
      const data = await api("/api/generate_step", stepBody(
        state.attnMode === "single" ? { head: 0, focus_layer: state.attnLayer } : {}));
      state.lastStep = data;
      // record generated token text (only when actually advancing, not the silent preview)
      if (!silent) { genText.push(data.sampled.text); state.generated.push(data.sampled.id); }
      await renderAll(data, animate);
      if (!silent) status(data.done ? `done · ${data.tokens.length} tokens` : "token added — step again or keep going", "ok");
      return data;
    } catch (e) {
      status("error: " + e.message, "err");
      throw e;
    }
  }

  async function runAll() {
    if (state.running) { state.running = false; return; } // toggle stop
    resetRunSoft();
    state.running = true; const myToken = ++state.runToken;
    $("run").textContent = "■ Stop";
    const MAX = 16;
    try {
      for (let i = 0; i < MAX; i++) {
        if (!state.running || myToken !== state.runToken) break;
        const data = await doStep(true, false);
        if (data.done) break;
        if (!reduced) await sleep(450);
      }
    } finally {
      if (myToken === state.runToken) { state.running = false; $("run").textContent = "▶ Generate"; renderTextStream(); }
    }
  }
  function resetRunSoft() { state.generated = []; state.lastStep = null; genText.length = 0; }

  // ---------- helpers ----------
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  function escapeHtml(s) { return (s || "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

  // ---------- wire up ----------
  function init() {
    // sample prompts
    const sr = $("samples");
    SAMPLES.forEach((p) => {
      const b = document.createElement("button");
      b.textContent = p.length > 22 ? p.slice(0, 22) + "…" : p;
      b.title = p;
      b.addEventListener("click", () => { $("prompt").value = p; state.prompt = p; resetRun(); });
      sr.appendChild(b);
    });

    $("prompt").addEventListener("input", (e) => { state.prompt = e.target.value; });
    $("prompt").addEventListener("change", () => resetRun());

    $("temp").addEventListener("input", (e) => {
      state.temp = parseFloat(e.target.value); $("temp-out").textContent = state.temp.toFixed(2);
      debouncedPreview();
    });
    $("topk").addEventListener("input", (e) => {
      state.topk = parseInt(e.target.value, 10); $("topk-out").textContent = state.topk;
      debouncedPreview();
    });

    $("attn-layer").addEventListener("input", (e) => {
      state.attnLayer = parseInt(e.target.value, 10); $("attn-layer-out").textContent = state.attnLayer;
      if (state.attnMode === "single") doStep(false, true).catch(() => {});
      else if (state.lastStep) renderAttentionControls(state.lastStep);
    });
    $("attn-mode").addEventListener("click", () => {
      state.attnMode = state.attnMode === "mean" ? "single" : "mean";
      const btn = $("attn-mode");
      btn.textContent = state.attnMode === "mean" ? "mean heads" : "head 0";
      btn.classList.toggle("active", state.attnMode === "mean");
      doStep(false, true).catch(() => {});
    });

    $("run").addEventListener("click", runAll);
    $("step").addEventListener("click", () => { state.running = false; doStep(true, false); });

    // clicking a layer block in the pipeline focuses the attention view on it
    window.LLMViz.onLayerClick = (i) => {
      state.attnLayer = i; $("attn-layer").value = i; $("attn-layer-out").textContent = i;
      if (state.attnMode === "single") doStep(false, true).catch(() => {});
      else if (state.lastStep) renderAttentionControls(state.lastStep);
    };

    // debounced re-preview of the CURRENT step so slider changes reshape the bars live
    let dt; function debouncedPreview() { clearTimeout(dt); dt = setTimeout(() => { if (!state.running) doStep(false, true); }, 160); }

    // boot: load models, then auto-play the DEMO landing experience
    api("/api/models").then((r) => {
      buildModels(r.models);
      const def = r.models.find((m) => m.default) || r.models[0];
      if (def) selectModel(def.id);
      status("ready — press Generate", "");
      if (!reduced) setTimeout(runAll, 600); // DEMO auto-plays on load
    }).catch((e) => status("could not load models: " + e.message, "err"));
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
