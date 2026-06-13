/* viz.js — the center pipeline: tokens → embeddings → transformer layers → next token.
   Renders an SVG (960×600 viewBox) and animates the flow left-to-right, mirroring
   NeuraNetViz's forward-pass grammar. Exposes window.LLMViz.pipeline. */
(function () {
  const SVGNS = "http://www.w3.org/2000/svg";
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function el(tag, attrs, kids) {
    const n = document.createElementNS(SVGNS, tag);
    if (attrs) for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (kids) (Array.isArray(kids) ? kids : [kids]).forEach((c) => c && n.appendChild(c));
    return n;
  }
  const clear = (svg) => { while (svg.firstChild) svg.removeChild(svg.firstChild); };

  // layout columns within the 960×600 viewBox
  const COL = { tokens: 110, embed: 360, layers: 600, out: 850 };
  const TOP = 80, BOT = 540;

  // concept copy shown in tooltips (the teaching layer)
  const TIP = {
    tokens: ["Tokens", "Your text is split into tokens (sub-words). Each has an ID the model understands."],
    embed: ["Embeddings", "Every token becomes a vector — a point in “meaning space”. Nearby points mean similar things. Shown projected to 2D."],
    layer: ["Transformer layer", "Each layer mixes information between tokens (attention) then transforms it (MLP). Stacking many layers builds deeper understanding."],
    out: ["Next token", "The final layer scores every possible next token. One is picked and added to the text — then it all runs again."],
  };

  let tip;
  function showTip(target, title, body) {
    if (!tip) tip = document.getElementById("viz-tooltip");
    if (!tip) return;
    const stage = target.ownerSVGElement.parentElement.getBoundingClientRect();
    const r = target.getBoundingClientRect();
    tip.innerHTML = `<div class="tt-title">${title}</div><div>${body}</div>`;
    tip.style.left = (r.left - stage.left + r.width / 2) + "px";
    tip.style.top = (r.top - stage.top) + "px";
    tip.setAttribute("data-visible", "true");
  }
  function hideTip() { if (tip) tip.setAttribute("data-visible", "false"); }

  function hoverable(node, title, body) {
    node.addEventListener("mouseenter", () => showTip(node, title, body));
    node.addEventListener("mouseleave", hideTip);
    return node;
  }

  function stageLabel(x, text) {
    const t = el("text", { x, y: 46, class: "stage-label", "text-anchor": "middle" });
    t.textContent = text;
    return t;
  }

  // returns {groups} so the animator can pulse them in order
  function render(svg, data) {
    clear(svg);
    const tokens = data.tokens || [];
    const layers = data.layers || [];
    const groups = { tokenG: [], embedG: null, layerG: [], outG: null, edges: [] };

    // ---- defs: pulsing edge gradient ----
    const defs = el("defs");
    const grad = el("linearGradient", { id: "flow", x1: "0", y1: "0", x2: "1", y2: "0" });
    grad.appendChild(el("stop", { offset: "0%", "stop-color": "rgba(40,214,0,0.05)" }));
    grad.appendChild(el("stop", { offset: "50%", "stop-color": "rgba(40,214,0,0.65)" }));
    grad.appendChild(el("stop", { offset: "100%", "stop-color": "rgba(40,214,0,0.05)" }));
    defs.appendChild(grad);
    svg.appendChild(defs);

    svg.appendChild(stageLabel(COL.tokens, "tokens"));
    svg.appendChild(stageLabel(COL.embed, "embeddings"));
    svg.appendChild(stageLabel(COL.layers, "transformer layers"));
    svg.appendChild(stageLabel(COL.out, "next token"));

    // ---- tokens column (vertical chips; show the most recent if many) ----
    const maxTok = 16;
    const shown = tokens.length > maxTok ? tokens.slice(tokens.length - maxTok) : tokens;
    const th = Math.min(30, (BOT - TOP) / Math.max(shown.length, 1));
    const tokY = (i) => TOP + i * th + th / 2;
    shown.forEach((tok, i) => {
      const y = tokY(i);
      const w = 120, x = COL.tokens - w / 2;
      const g = el("g", { class: "tok-chip" });
      g.appendChild(el("rect", { x, y: y - th / 2 + 2, width: w, height: th - 4, rx: 7,
        fill: "rgba(255,255,255,0.06)", stroke: "rgba(255,255,255,0.25)" }));
      const t = el("text", { x: COL.tokens, y: y + 4, class: "tok-text", "text-anchor": "middle" });
      t.textContent = (tok.text || "").replace(/\n/g, "⏎").slice(0, 14) || "·";
      g.appendChild(t);
      hoverable(g, TIP.tokens[0], `“${(tok.text || "").trim()}” · id ${tok.id}`);
      svg.appendChild(g);
      groups.tokenG.push(g);
    });
    if (tokens.length > maxTok) {
      const more = el("text", { x: COL.tokens, y: TOP - 6, class: "stage-label", "text-anchor": "middle" });
      more.textContent = `(+${tokens.length - maxTok} earlier)`;
      svg.appendChild(more);
    }

    // ---- embeddings scatter: each token becomes a vector, shown as a point in 2D ----
    const bs = 200, bx = COL.embed - bs / 2, by = (TOP + BOT) / 2 - bs / 2;  // vertically centered
    const embBox = el("g", { class: "emb-box" });
    embBox.appendChild(el("rect", { x: bx, y: by, width: bs, height: bs, rx: 10,
      fill: "rgba(40,214,0,0.04)", stroke: "rgba(40,214,0,0.25)" }));
    // faint cross-hair axes to signal "this is a 2D space"
    embBox.appendChild(el("line", { x1: bx + 12, y1: by + bs / 2, x2: bx + bs - 12, y2: by + bs / 2, stroke: "rgba(255,255,255,0.07)" }));
    embBox.appendChild(el("line", { x1: bx + bs / 2, y1: by + 12, x2: bx + bs / 2, y2: by + bs - 12, stroke: "rgba(255,255,255,0.07)" }));
    const emb = data.embeddings_2d || [];
    const labelAll = emb.length <= 12;       // avoid clutter on long prompts
    emb.forEach((p, i) => {
      const cx = bx + bs / 2 + p[0] * (bs / 2 - 26);
      const cy = by + bs / 2 + p[1] * (bs / 2 - 26);
      embBox.appendChild(el("circle", { class: "emb-dot", cx, cy, r: 4, opacity: 0.9 }));
      const txt = ((tokens[i] && tokens[i].text) || "").trim();
      if (labelAll && txt) {
        const lab = el("text", { x: cx + 7, y: cy + 3, class: "emb-label" });
        lab.textContent = txt.slice(0, 8);
        embBox.appendChild(lab);
      }
    });
    // honest caption: real "meaning space" only with a live model; illustrative in DEMO
    const live = data.engine === "live";
    const cap = el("text", { x: COL.embed, y: by + bs + 18, "text-anchor": "middle", class: "emb-caption" });
    cap.textContent = live ? "the model's learned meaning space (2D)" : "positions are illustrative (DEMO)";
    embBox.appendChild(cap);
    const cap2 = el("text", { x: COL.embed, y: by - 10, "text-anchor": "middle", class: "emb-caption" });
    cap2.textContent = "each token → a vector → a point";
    embBox.appendChild(cap2);
    hoverable(embBox, TIP.embed[0],
      live ? "Each token is turned into a vector of " + (data.dim || "hundreds of") + " numbers the model learned during training. Similar words sit near each other. Shown here squeezed down to 2D."
           : "Each token becomes a vector of numbers (its embedding). With a real model (NANO/MICRO) similar words cluster; in DEMO these positions are just illustrative.");
    svg.appendChild(embBox);
    groups.embedG = embBox;

    // ---- transformer layer stack ----
    const n = layers.length || 1;
    const lh = Math.min(34, (BOT - TOP) / n);
    const lw = 150, lx = COL.layers - lw / 2;
    const layY = (i) => TOP + i * lh;
    layers.forEach((L, i) => {
      const g = el("g", { class: "layer-block" });
      const strength = L.hidden_norm == null ? 0.5 : L.hidden_norm;
      const fill = `rgba(40,214,0,${0.12 + 0.55 * strength})`;
      g.appendChild(el("rect", { x: lx, y: layY(i) + 1.5, width: lw, height: lh - 3, rx: 6,
        fill, stroke: "rgba(40,214,0,0.45)" }));
      if (lh >= 18) {
        const t = el("text", { x: COL.layers, y: layY(i) + lh / 2 + 4, "text-anchor": "middle",
          class: "tok-text", "font-size": Math.min(11, lh - 8) });
        t.textContent = n <= 14 ? `L${i} · attn+mlp` : `L${i}`;
        t.setAttribute("fill", "rgba(255,255,255,0.85)");
        g.appendChild(t);
      }
      hoverable(g, `${TIP.layer[0]} ${i}`, TIP.layer[1] + ` (activation ${strength.toFixed(2)})`);
      g.addEventListener("click", () => window.LLMViz && window.LLMViz.onLayerClick && window.LLMViz.onLayerClick(i));
      svg.appendChild(g);
      groups.layerG.push(g);
    });

    // ---- next-token chip ----
    const outY = (TOP + BOT) / 2;
    const og = el("g", { class: "out-chip" });
    const ow = 130;
    og.appendChild(el("rect", { x: COL.out - ow / 2, y: outY - 22, width: ow, height: 44, rx: 10 }));
    const ot = el("text", { x: COL.out, y: outY + 6, "text-anchor": "middle" });
    ot.textContent = (data.sampled && data.sampled.text || "…").slice(0, 12);
    og.appendChild(ot);
    hoverable(og, TIP.out[0], TIP.out[1]);
    svg.appendChild(og);
    groups.outG = og;

    // ---- flow edges (drawn under nodes would be nicer, but fine on top faintly) ----
    function edge(x1, y1, x2, y2) {
      const p = el("path", { class: "flow-edge",
        d: `M ${x1} ${y1} C ${(x1 + x2) / 2} ${y1}, ${(x1 + x2) / 2} ${y2}, ${x2} ${y2}`,
        fill: "none", stroke: "rgba(255,255,255,0.10)", "stroke-width": 1.5 });
      svg.insertBefore(p, svg.firstChild.nextSibling); // above defs, below nodes
      groups.edges.push(p);
      return p;
    }
    // a few representative edges (not all-to-all, to stay readable)
    edge(COL.tokens + 60, outY, bx, by + bs / 2);
    edge(bx + bs, by + bs / 2, lx, outY);
    edge(lx + lw, outY, COL.out - ow / 2, outY);

    return groups;
  }

  function pulse(node) {
    if (reduced || !node) return;
    node.classList.remove("pulse");
    void node.getBoundingClientRect();
    node.classList.add("pulse");
  }

  // animate the flow; returns a promise that resolves when done
  function animate(groups) {
    return new Promise((resolve) => {
      if (reduced) { resolve(); return; }
      const seq = [];
      groups.tokenG.forEach((g) => seq.push(g));
      seq.push(groups.embedG);
      groups.layerG.forEach((g) => seq.push(g));
      seq.push(groups.outG);
      let i = 0;
      const tokStep = Math.max(12, 220 / Math.max(groups.tokenG.length, 1));
      function tick() {
        if (i >= seq.length) { resolve(); return; }
        pulse(seq[i]);
        // light up the edge as we cross stages
        groups.edges.forEach((e) => {
          e.setAttribute("stroke", "url(#flow)"); e.setAttribute("stroke-width", "2.5");
          setTimeout(() => { e.setAttribute("stroke", "rgba(255,255,255,0.10)"); e.setAttribute("stroke-width", "1.5"); }, 320);
        });
        i++;
        const delay = i <= groups.tokenG.length ? tokStep : 90;
        setTimeout(tick, delay);
      }
      tick();
    });
  }

  window.LLMViz = window.LLMViz || {};
  window.LLMViz.pipeline = { render, animate };
})();
