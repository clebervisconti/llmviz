/* attention.js — renders the seq×seq attention matrix as a heatmap on a canvas,
   using the NeuraNetViz Preto Neural → Verde Ascensão ramp (R 30→40, G 30→214, B 30→0).
   One pixel per cell; CSS scales it up with image-rendering: pixelated.
   Exposes window.LLMViz.attention. */
(function () {
  function ramp(t) {
    // t in 0..1 → [r,g,b]
    return [
      Math.round(0x1e + (0x28 - 0x1e) * t),
      Math.round(0x1e + (0xd6 - 0x1e) * t),
      Math.round(0x1e + (0x00 - 0x1e) * t),
    ];
  }

  function render(canvas, layer) {
    if (!canvas || !layer || !layer.attention) return;
    const m = layer.attention.matrix_u8;
    const seq = m.length;
    canvas.width = seq;
    canvas.height = seq;
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(seq, seq);
    for (let r = 0; r < seq; r++) {
      for (let c = 0; c < seq; c++) {
        const t = (m[r][c] || 0) / 255;
        const [R, G, B] = ramp(t);
        const idx = (r * seq + c) * 4;
        img.data[idx] = R; img.data[idx + 1] = G; img.data[idx + 2] = B; img.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }

  window.LLMViz = window.LLMViz || {};
  window.LLMViz.attention = { render };
})();
