/* block_viz.js — "inside one transformer block" view, in the style of
   poloclub.github.io/transformer-explainer. Renders ONE block expanded into the same
   #pipeline SVG (960×600 viewBox): token column → color-coded Q/K/V strips → multi-head
   self-attention grid (Query·Key → softmax) → Out → MLP → residual, joined by Sankey
   ribbons. Driven by the focus layer's real Q/K/V (data.layers[L].qkv) + single-head
   attention matrix. Exposes window.LLMViz.block. */
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

  // single source of truth for the scoped Q/K/V triad (CSS vars, see UI-SPEC §6a)
  function colors() {
    const css = getComputedStyle(document.documentElement);
    const get = (v, fb) => (css.getPropertyValue(v).trim() || fb);
    return {
      q: get("--cv-q", "#7c6cff"),
      k: get("--cv-k", "#ff8a3d"),
      v: get("--cv-v", "#28d600"),
    };
  }

  // layout columns within the 960×600 viewBox
  const COL = { tok: 86, q: 214, k: 292, v: 370, grid: 470, out: 700, mlp: 800, next: 900 };
  const LANE_W = 60, GRID_W = 150;
  const TOP = 96, BOT = 540;

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
  function stageLabel(x, text, fill) {
    const t = el("text", { x, y: 62, class: "stage-label", "text-anchor": "middle" });
    if (fill) t.setAttribute("fill", fill);
    t.textContent = text;
    return t;
  }
  // green ramp for the attention grid (matches attention.js)
  function gridFill(t) {
    const r = Math.round(0x1e + (0x28 - 0x1e) * t);
    const g = Math.round(0x1e + (0xd6 - 0x1e) * t);
    const b = Math.round(0x1e + (0x00 - 0x1e) * t);
    return `rgb(${r},${g},${b})`;
  }

  // filled cubic-bezier "ribbon" band between two vertical segments
  function ribbon(x1, yt1, yb1, x2, yt2, yb2, fill, opacity, cls) {
    const mid = (x1 + x2) / 2;
    const d = `M ${x1} ${yt1} C ${mid} ${yt1}, ${mid} ${yt2}, ${x2} ${yt2}` +
              ` L ${x2} ${yb2} C ${mid} ${yb2}, ${mid} ${yb1}, ${x1} ${yb1} Z`;
    return el("path", { d, fill, opacity, class: cls || "sankey" });
  }

  // a thin curved centerline with a CSS-animated dash → "particles flowing" along the path.
  // Reduced-motion: the global `animation:none` rule freezes it to a static dashed line.
  function flowLine(x1, y1, x2, y2, color, opacity, width, idx) {
    const mid = (x1 + x2) / 2;
    const a = { class: "flow-line",
      d: `M ${x1} ${y1} C ${mid} ${y1}, ${mid} ${y2}, ${x2} ${y2}`,
      fill: "none", stroke: color, "stroke-width": width || 1.4,
      "stroke-opacity": opacity, "stroke-linecap": "round" };
    if (idx != null) a["data-tok"] = idx;
    return el("path", a);
  }

  function render(svg, data, sel) {
    clear(svg);
    const C = colors();
    const tokens = data.tokens || [];
    const layers = data.layers || [];
    const L = layers[sel.layer] || layers[layers.length - 1] || {};
    const qkv = L.qkv;
    const nLayers = layers.length || 1;
    const midY = (TOP + BOT) / 2;
    const groups = { tokenG: [], qkvG: [], gridG: null, outG: null, mlpG: null, ribbons: [] };

    // ---- defs: directional flow gradients per channel ----
    const defs = el("defs");
    function grad(id, c0) {
      const g = el("linearGradient", { id, x1: "0", y1: "0", x2: "1", y2: "0" });
      g.appendChild(el("stop", { offset: "0%", "stop-color": c0, "stop-opacity": "0.05" }));
      g.appendChild(el("stop", { offset: "50%", "stop-color": c0, "stop-opacity": "0.55" }));
      g.appendChild(el("stop", { offset: "100%", "stop-color": c0, "stop-opacity": "0.05" }));
      defs.appendChild(g);
    }
    grad("flowQ", C.q); grad("flowK", C.k); grad("flowV", C.v); grad("flowG", C.v);
    svg.appendChild(defs);

    // ---- stacked-card motif: "+N−1 more identical blocks" behind the block ----
    const stackN = Math.min(3, Math.max(0, nLayers - 1));
    for (let s = stackN; s >= 1; s--) {
      const off = s * 7;
      svg.appendChild(el("rect", {
        x: 150 + off, y: TOP - 14 + off, width: 740 - off, height: (BOT - TOP) + 30,
        rx: 14, fill: "rgba(255,255,255,0.018)", stroke: "rgba(255,255,255,0.06)",
      }));
    }
    if (nLayers > 1) {
      const more = el("text", { x: 890, y: TOP - 22, class: "block-more", "text-anchor": "end" });
      more.textContent = `+ ${nLayers - 1} more identical block${nLayers - 1 === 1 ? "" : "s"}`;
      svg.appendChild(more);
    }
    // the focused-block frame
    svg.appendChild(el("rect", {
      x: 150, y: TOP - 14, width: 740, height: (BOT - TOP) + 30, rx: 14,
      fill: "rgba(40,214,0,0.02)", stroke: "rgba(40,214,0,0.22)", class: "block-frame",
    }));

    svg.appendChild(stageLabel(COL.tok, "tokens"));
    svg.appendChild(stageLabel(COL.q, "Q", C.q));
    svg.appendChild(stageLabel(COL.k, "K", C.k));
    svg.appendChild(stageLabel(COL.v, "V", C.v));
    svg.appendChild(stageLabel(COL.grid + GRID_W / 2, "self-attention"));
    svg.appendChild(stageLabel(COL.out, "out"));
    svg.appendChild(stageLabel(COL.mlp, "MLP"));
    svg.appendChild(stageLabel(COL.next, "next", C.v));

    // ---- token column (most-recent if many) ----
    const maxTok = 22;
    const shown = tokens.length > maxTok ? tokens.slice(tokens.length - maxTok) : tokens;
    const seqShown = shown.length;
    const th = Math.min(26, (BOT - TOP) / Math.max(seqShown, 1));
    const tokY = (i) => TOP + i * th + th / 2;

    // Q/K/V strips need the same per-token rows; map shown-index → norm/bars
    const offset = tokens.length - seqShown;  // align qkv arrays (full seq) to shown rows
    const u8 = (arr, i) => (arr && arr[offset + i] != null ? arr[offset + i] / 255 : 0);
    const bars = (arr, i) => (arr && arr[offset + i]) || null;

    shown.forEach((tok, i) => {
      const y = tokY(i);
      const w = 104, x = COL.tok - w / 2;
      const g = el("g", { class: "tok-chip", "data-tok": i });
      g.appendChild(el("rect", { x, y: y - th / 2 + 2, width: w, height: th - 4, rx: 7,
        fill: "rgba(255,255,255,0.11)", stroke: "rgba(255,255,255,0.32)" }));
      const t = el("text", { x: COL.tok, y: y + 4, class: "tok-text", "text-anchor": "middle" });
      t.textContent = (tok.text || "").replace(/\n/g, "⏎").slice(0, 12) || "·";
      g.appendChild(t);
      hoverable(g, "Token", `“${(tok.text || "").trim()}” · id ${tok.id}`);
      svg.appendChild(g);
      groups.tokenG.push(g);
    });
    if (tokens.length > maxTok) {
      const more = el("text", { x: COL.tok, y: TOP - 6, class: "stage-label", "text-anchor": "middle" });
      more.textContent = `(+${tokens.length - maxTok} earlier)`;
      svg.appendChild(more);
    }

    // ---- Q/K/V strips: per-token mini-vector (8 buckets) colored by channel ----
    function lane(colx, color, normArr, barArr, name, gradId) {
      const lg = el("g", { class: "qkv-lane" });
      shown.forEach((tok, i) => {
        const y = tokY(i);
        const cw = LANE_W, cx = colx - cw / 2, cy = y - th / 2 + 3, ch = th - 6;
        const nrm = u8(normArr, i);
        const cell = el("g", { class: "qkv-cell", "data-tok": i });
        // backing chip whose opacity reads the vector's overall magnitude (norm)
        cell.appendChild(el("rect", { x: cx, y: cy, width: cw, height: ch, rx: 4,
          fill: color, opacity: (0.12 + 0.32 * nrm).toFixed(3),
          stroke: color, "stroke-opacity": 0.5 }));
        // 8-bucket mini-vector
        const b = bars(barArr, i);
        if (b && ch > 5) {
          const n = b.length, bw = (cw - 6) / n;
          for (let j = 0; j < n; j++) {
            const h = Math.max(1, (b[j] / 255) * (ch - 4));
            cell.appendChild(el("rect", {
              x: cx + 3 + j * bw + 0.5, y: cy + ch - 2 - h, width: Math.max(1, bw - 1), height: h,
              fill: color, opacity: 0.85, rx: 0.6,
            }));
          }
        }
        hoverable(cell, name + " vector",
          `token “${(tok.text || "").trim()}” · ${name} magnitude ${(nrm).toFixed(2)} (head ${sel.head})`);
        lg.appendChild(cell);
        // thin flowing line token → this lane cell (thickness/opacity read the norm)
        const fl = flowLine(COL.tok + 52, y, colx - LANE_W / 2, y,
          color, (0.2 + 0.55 * nrm).toFixed(3), 1 + 1.6 * nrm, i);
        svg.insertBefore(fl, svg.firstChild.nextSibling);
        groups.ribbons.push(fl);
      });
      svg.appendChild(lg);
      groups.qkvG.push(lg);
    }
    if (qkv) {
      lane(COL.q, C.q, qkv.q_norm, qkv.q_bars, "Query", "flowQ");
      lane(COL.k, C.k, qkv.k_norm, qkv.k_bars, "Key", "flowK");
      lane(COL.v, C.v, qkv.v_norm, qkv.v_bars, "Value", "flowV");
    } else {
      const na = el("text", { x: (COL.q + COL.v) / 2, y: (TOP + BOT) / 2, "text-anchor": "middle", class: "emb-caption" });
      na.textContent = "Q/K/V not available — switch to a white-box model";
      svg.appendChild(na);
    }

    // ---- attention grid (selected head): Query·Key dot products → softmax ----
    const mat = (L.attention && L.attention.matrix_u8) || [];
    const gseq = mat.length;
    const gx = COL.grid, gy = (TOP + BOT) / 2 - GRID_W / 2;
    const gridG = el("g", { class: "attn-grid" });
    gridG.appendChild(el("rect", { x: gx - 4, y: gy - 4, width: GRID_W + 8, height: GRID_W + 8, rx: 8,
      fill: "rgba(0,0,0,0.35)", stroke: "rgba(40,214,0,0.25)" }));
    if (gseq) {
      const cs = GRID_W / gseq;
      for (let r = 0; r < gseq; r++) {
        for (let c = 0; c <= r; c++) {           // causal: upper triangle is ~0
          const tval = (mat[r][c] || 0) / 255;
          if (tval < 0.02) continue;
          gridG.appendChild(el("rect", {
            x: gx + c * cs, y: gy + r * cs, width: Math.ceil(cs), height: Math.ceil(cs),
            fill: gridFill(tval), opacity: 0.95,
          }));
        }
      }
    }
    hoverable(gridG, "Self-attention", "Each row is a query token; each column a key token. Brighter = the query attends more to that key. Causal: a token can only look back.");
    svg.appendChild(gridG);
    groups.gridG = gridG;
    const gcap = el("text", { x: gx + GRID_W / 2, y: gy + GRID_W + 16, "text-anchor": "middle", class: "emb-caption" });
    gcap.textContent = "Query · Key → softmax";
    svg.appendChild(gcap);

    // Q and K ribbons feeding the grid (the dot-product inputs) + flowing comets
    const gmidT = gy, gmidB = gy + GRID_W, gMid = gy + GRID_W / 2;
    groups.ribbons.push(svg.insertBefore(
      ribbon(COL.q + LANE_W / 2, TOP, BOT, gx, gmidT, gmidB, "url(#flowQ)", 0.18, "sankey"), svg.firstChild.nextSibling));
    groups.ribbons.push(svg.insertBefore(
      ribbon(COL.k + LANE_W / 2, TOP, BOT, gx, gmidT, gmidB, "url(#flowK)", 0.18, "sankey"), svg.firstChild.nextSibling));
    svg.appendChild(flowLine(COL.q + LANE_W / 2, midY, gx, gMid - 8, C.q, 0.5, 2));
    svg.appendChild(flowLine(COL.k + LANE_W / 2, midY, gx, gMid + 8, C.k, 0.5, 2));

    // ---- Out node (attention output) ----
    const outG = el("g", { class: "out-node" });
    outG.appendChild(el("rect", { x: COL.out - 30, y: midY - 30, width: 60, height: 60, rx: 10,
      fill: "rgba(40,214,0,0.10)", stroke: "rgba(40,214,0,0.55)" }));
    const ot = el("text", { x: COL.out, y: midY + 4, "text-anchor": "middle", class: "node-label" });
    ot.textContent = "Out";
    outG.appendChild(ot);
    hoverable(outG, "Attention output", "The attended Values are combined and projected — this token's updated representation.");
    svg.appendChild(outG);
    groups.outG = outG;
    // V → Out ribbon, and grid → Out
    groups.ribbons.push(svg.insertBefore(
      ribbon(COL.v + LANE_W / 2, TOP, BOT, COL.out - 30, midY - 28, midY + 28, "url(#flowV)", 0.22, "sankey"), svg.firstChild.nextSibling));
    groups.ribbons.push(svg.insertBefore(
      ribbon(gx + GRID_W, gmidT, gmidB, COL.out - 30, midY - 26, midY + 26, "url(#flowG)", 0.2, "sankey"), svg.firstChild.nextSibling));
    svg.appendChild(flowLine(COL.v + LANE_W / 2, midY, COL.out - 30, midY - 6, C.v, 0.5, 2));
    svg.appendChild(flowLine(gx + GRID_W, gMid, COL.out - 30, midY + 6, C.v, 0.45, 2));

    // ---- MLP block ----
    const mlpG = el("g", { class: "mlp-node" });
    mlpG.appendChild(el("rect", { x: COL.mlp - 34, y: midY - 40, width: 68, height: 80, rx: 10,
      fill: "rgba(40,214,0,0.08)", stroke: "rgba(40,214,0,0.5)" }));
    [-22, 0, 22].forEach((dy) => mlpG.appendChild(el("circle", { cx: COL.mlp - 14, cy: midY + dy, r: 4, fill: "rgba(40,214,0,0.6)" })));
    [-11, 11].forEach((dy) => mlpG.appendChild(el("circle", { cx: COL.mlp + 14, cy: midY + dy, r: 4, fill: "rgba(40,214,0,0.6)" })));
    const mt = el("text", { x: COL.mlp, y: midY + 58, "text-anchor": "middle", class: "node-label" });
    mt.textContent = "MLP";
    mlpG.appendChild(mt);
    hoverable(mlpG, "MLP", "A small feed-forward network transforms each token's vector independently — the block's 'thinking' step.");
    svg.appendChild(mlpG);
    groups.mlpG = mlpG;
    groups.ribbons.push(svg.insertBefore(
      ribbon(COL.out + 30, midY - 26, midY + 26, COL.mlp - 34, midY - 30, midY + 30, "url(#flowG)", 0.22, "sankey"), svg.firstChild.nextSibling));
    svg.appendChild(flowLine(COL.out + 30, midY, COL.mlp - 34, midY, C.v, 0.5, 2.2));

    // ---- next-token chip ----
    const ng = el("g", { class: "out-chip" });
    ng.appendChild(el("rect", { x: COL.next - 36, y: midY - 18, width: 72, height: 36, rx: 9 }));
    const nt = el("text", { x: COL.next, y: midY + 5, "text-anchor": "middle" });
    nt.textContent = (data.sampled && data.sampled.text || "…").slice(0, 9);
    ng.appendChild(nt);
    hoverable(ng, "Next token", "After all blocks, the final scores pick one token — added to the text, then it runs again.");
    svg.appendChild(ng);
    groups.ribbons.push(svg.insertBefore(
      ribbon(COL.mlp + 34, midY - 30, midY + 30, COL.next - 36, midY - 16, midY + 16, "url(#flowG)", 0.22, "sankey"), svg.firstChild.nextSibling));
    svg.appendChild(flowLine(COL.mlp + 34, midY, COL.next - 36, midY, C.v, 0.55, 2.2));

    // ---- residual skip connection (token column → after MLP) ----
    const resPath = el("path", {
      class: "residual",
      d: `M ${COL.tok} ${TOP - 26} C 360 ${TOP - 70}, 640 ${TOP - 70}, ${COL.mlp} ${midY - 42}`,
      fill: "none", stroke: "rgba(255,255,255,0.22)", "stroke-width": 1.5, "stroke-dasharray": "5 4",
    });
    svg.appendChild(resPath);
    const rlab = el("text", { x: 500, y: TOP - 56, "text-anchor": "middle", class: "emb-caption" });
    rlab.textContent = "+ residual (skip connection)";
    svg.appendChild(rlab);

    // ---- hover path-tracing: highlight one token's Q/K/V + its attention row ----
    const cs = gseq ? GRID_W / gseq : 0;
    const rowHi = el("rect", { class: "grid-row-hi", x: gx, y: gy, width: GRID_W,
      height: Math.max(2, cs), rx: 1, visibility: "hidden" });
    svg.appendChild(rowHi);
    let lastHot = -1;
    function setHot(i) {
      if (i === lastHot) return;
      lastHot = i;
      if (i == null) {
        svg.removeAttribute("data-hot");
        svg.querySelectorAll(".hot").forEach((n) => n.classList.remove("hot"));
        rowHi.setAttribute("visibility", "hidden");
        return;
      }
      svg.setAttribute("data-hot", "1");
      svg.querySelectorAll("[data-tok]").forEach((n) =>
        n.classList.toggle("hot", n.getAttribute("data-tok") === String(i)));
      if (cs && i < gseq - offset) {
        rowHi.setAttribute("y", gy + (i) * cs);
        rowHi.setAttribute("visibility", "visible");
      } else { rowHi.setAttribute("visibility", "hidden"); }
    }
    svg.addEventListener("mousemove", (e) => {
      const t = e.target && e.target.closest ? e.target.closest("[data-tok]") : null;
      setHot(t ? parseInt(t.getAttribute("data-tok"), 10) : null);
    });
    svg.addEventListener("mouseleave", () => setHot(null));

    return groups;
  }

  function pulse(node) {
    if (reduced || !node) return;
    node.classList.remove("pulse");
    void node.getBoundingClientRect();
    node.classList.add("pulse");
  }

  function animate(groups) {
    return new Promise((resolve) => {
      if (reduced) { resolve(); return; }
      const seq = [];
      groups.tokenG.forEach((g) => seq.push(g));
      groups.qkvG.forEach((g) => seq.push(g));
      seq.push(groups.gridG, groups.outG, groups.mlpG);
      let i = 0;
      function tick() {
        if (i >= seq.length) { resolve(); return; }
        pulse(seq[i]);
        groups.ribbons.forEach((r) => {
          r.classList.add("flowing");
          setTimeout(() => r.classList.remove("flowing"), 360);
        });
        i++;
        setTimeout(tick, 70);
      }
      tick();
    });
  }

  window.LLMViz = window.LLMViz || {};
  window.LLMViz.block = { render, animate };
})();
