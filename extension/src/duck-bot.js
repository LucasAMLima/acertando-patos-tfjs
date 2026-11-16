// Content script (isolated world) que roda dentro do iframe do jogo.
//
// Pipeline a cada frame:
//   canvas do jogo -> tensor -> YOLOv5n (TF.js) -> caixas duck / red_duck / egg
//   -> estima a velocidade de cada prateleira -> prevê onde cada objeto estará
//   -> atira só nos patos amarelos cuja posição prevista não encosta em
//      pato vermelho/ovo -> dispara o clique.
//
// Regras do jogo (lidas do data.js):
//   pato amarelo = +1 ponto e +2 s | pato vermelho = -1 ponto | ovo = GAME OVER
//
// Atalho: tecla "B" liga/desliga o bot. A sobreposição mostra o que a IA vê.
(async () => {
  const CONFIG = {
    // dentro da extensão usa o arquivo empacotado; fora dela (testes) aceita uma URL global
    modelUrl: globalThis.DUCK_BOT_MODEL_URL ?? chrome.runtime.getURL('models/ducks_web_model/model.json'),
    labels: ['duck', 'red_duck', 'egg'],
    targetClass: 0,          // pato amarelo com alvo: o ÚNICO que deve ser atingido
    avoidClasses: [1, 2],    // pato vermelho e ovos: nunca atirar
    minScore: 0.5,
    shotMemoryMs: 500,       // não atira de novo no mesmo pato enquanto ele "morre"
    laneTolerance: 0.06,     // objetos com a base nessa faixa de altura (fração) = mesma prateleira
    safetyMargin: 0.35,      // margem extra ao redor de ovos/patos vermelhos (fração da largura deles)
    clickDelayMs: 8,         // tempo estimado entre decidir e o jogo processar o clique
  };

  const canvas = document.getElementById('c2canvas');
  if (!canvas) return;

  // ---------- Overlay / HUD ----------
  const overlay = document.createElement('canvas');
  Object.assign(overlay.style, { position: 'fixed', pointerEvents: 'none', zIndex: 2147483647 });
  document.body.appendChild(overlay);
  const octx = overlay.getContext('2d');
  let enabled = true;
  let status = 'carregando modelo...';
  document.addEventListener('keydown', e => {
    if (e.key === 'b' || e.key === 'B') enabled = !enabled;
  });

  // ---------- Modelo ----------
  await tf.ready();
  const model = await tf.loadGraphModel(CONFIG.modelUrl);
  const [, inH, inW] = model.inputs[0].shape;
  tf.dispose(await model.executeAsync(tf.zeros([1, inH, inW, 3])));
  status = `pronto (${tf.getBackend()})`;

  // Captura: reduz o canvas WebGL (1920x1080) para 640 px de largura na própria
  // GPU com drawImage, e só então lê os pixels -> ~9x menos dados por frame.
  const small = document.createElement('canvas');
  const sctx = small.getContext('2d', { willReadFrequently: false });

  // Pré-processamento igual ao do YOLOv5: completa até virar quadrado
  // (padding embaixo/à direita), redimensiona para 640x640 e normaliza [0,1].
  function preprocess() {
    const scale = inW / Math.max(canvas.width, canvas.height);
    const w = Math.round(canvas.width * scale), h = Math.round(canvas.height * scale);
    if (small.width !== w || small.height !== h) { small.width = w; small.height = h; }
    sctx.drawImage(canvas, 0, 0, w, h);
    return tf.tidy(() => {
      const img = tf.browser.fromPixels(small);
      return img.pad([[0, inH - h], [0, inW - w], [0, 0]]).div(255).expandDims(0);
    });
  }

  // O export do YOLOv5 para TF.js já inclui NMS e devolve 4 tensores:
  // boxes [1,N,4] (x1,y1,x2,y2 normalizados), scores [1,N], classes [1,N], count [1].
  async function detect() {
    const input = preprocess();
    const outputs = await model.executeAsync(input);
    input.dispose();
    const data = await Promise.all(outputs.map(t => t.data()));
    const shapes = outputs.map(t => t.shape);
    tf.dispose(outputs);

    const boxesIdx = shapes.findIndex(s => s.length === 3 && s[2] === 4);
    const countIdx = shapes.findIndex(s => s.length === 1);
    const rest = [0, 1, 2, 3].filter(i => i !== boxesIdx && i !== countIdx);
    const isInt = arr => Array.from(arr.slice(0, 20)).every(v => Number.isInteger(v));
    const classesIdx = isInt(data[rest[0]]) && !isInt(data[rest[1]]) ? rest[0] : rest[1];
    const scoresIdx = rest.find(i => i !== classesIdx);

    const side = Math.max(canvas.width, canvas.height);
    const n = data[countIdx][0];
    const dets = [];
    for (let i = 0; i < n; i++) {
      const score = data[scoresIdx][i];
      if (score < CONFIG.minScore) continue;
      const [x1, y1, x2, y2] = data[boxesIdx].slice(i * 4, i * 4 + 4).map(v => v * side);
      dets.push({ cls: data[classesIdx][i], score, x1, y1, x2, y2, cx: (x1 + x2) / 2, cy: (y1 + y2) / 2 });
    }
    return dets;
  }

  // ---------- Rastreamento por prateleira ----------
  // No jogo todos os PATOS de uma prateleira andam juntos (mesma DuckSpeed),
  // enquanto os ovos quase não saem do lugar (só balançam). Por isso estimamos
  // uma velocidade por (prateleira, grupo) — mediana dos pares casados entre
  // frames, suavizada — o que é bem mais robusto do que por objeto.
  const groupOf = cls => (cls === 2 ? 'egg' : 'duck'); // amarelo e vermelho = mesmo movimento
  const lanes = []; // { y, group, vx, n }
  function laneOf(d) {
    const tol = CONFIG.laneTolerance * canvas.height;
    const group = groupOf(d.cls);
    let lane = lanes.find(l => l.group === group && Math.abs(l.y - d.y2) < tol);
    if (!lane) { lane = { y: d.y2, group, vx: 0, n: 0 }; lanes.push(lane); }
    return lane;
  }

  let prev = [], prevTime = 0;
  function updateLanes(dets, time) {
    const dt = time - prevTime;
    const samples = new Map();
    if (prev.length && dt > 0 && dt < 1000) {
      for (const d of dets) {
        const lane = laneOf(d);
        let best = null, bestDx = canvas.width * 0.2;
        for (const p of prev) {
          if (laneOf(p) !== lane) continue;
          const dx = d.cx - p.cx;
          // depois que a direção da prateleira é conhecida, descarta pares no sentido oposto
          if (lane.n >= 3 && Math.abs(dx) > 4 && Math.sign(dx) !== Math.sign(lane.vx)) continue;
          if (Math.abs(dx) < Math.abs(bestDx)) { bestDx = dx; best = p; }
        }
        if (best) {
          if (!samples.has(lane)) samples.set(lane, []);
          samples.get(lane).push(bestDx / dt);
        }
      }
    }
    for (const [lane, vs] of samples) {
      vs.sort((a, b) => a - b);
      const med = vs[Math.floor(vs.length / 2)];
      lane.vx = lane.n ? 0.6 * lane.vx + 0.4 * med : med;
      lane.n++;
    }
    for (const d of dets) d.lane = laneOf(d);
    prev = dets;
    prevTime = time;
  }

  // ---------- Mira e disparo ----------
  // O jogo (Construct 2) lê a posição no "mousemove" e o tiro no "mousedown",
  // ambos escutados no document. Eventos sintéticos funcionam normalmente.
  function shoot(px, py) {
    const r = canvas.getBoundingClientRect();
    const clientX = r.left + px * (r.width / canvas.width);
    const clientY = r.top + py * (r.height / canvas.height);
    const base = { bubbles: true, cancelable: true, view: window, clientX, clientY, button: 0 };
    canvas.dispatchEvent(new MouseEvent('mousemove', base));
    canvas.dispatchEvent(new MouseEvent('mousedown', { ...base, buttons: 1 }));
    canvas.dispatchEvent(new MouseEvent('mouseup', base));
    canvas.dispatchEvent(new MouseEvent('click', base));
  }

  let recentShots = [];

  function chooseTargets(dets, captureTime) {
    const now = performance.now();
    const ahead = now - captureTime + CONFIG.clickDelayMs; // quanto o mundo andou desde a captura
    recentShots = recentShots.filter(s => now - s.t < CONFIG.shotMemoryMs);

    for (const d of dets) {
      d.dx = d.lane.vx * ahead;
      d.px1 = d.x1 + d.dx; d.px2 = d.x2 + d.dx;
    }
    const forbidden = dets.filter(d => CONFIG.avoidClasses.includes(d.cls));
    const targets = [];
    for (const d of dets) {
      if (d.cls !== CONFIG.targetClass || d.lane.n < 3) continue; // velocidade ainda desconhecida
      // ponto de mira: o alvo vermelho na barriga, na posição PREVISTA
      d.aimX = (d.px1 + d.px2) / 2;
      d.aimY = d.y1 + (d.y2 - d.y1) * 0.62;
      if (d.aimX < 0 || d.aimX > canvas.width) continue;
      const blocked = forbidden.some(f => {
        const m = CONFIG.safetyMargin * (f.x2 - f.x1) + Math.abs(f.lane.vx) * 40;
        return d.aimX > f.px1 - m && d.aimX < f.px2 + m && d.aimY > f.y1 - 10 && d.aimY < f.y2 + 10;
      });
      const alreadyShot = recentShots.some(s =>
        Math.abs(s.y - d.aimY) < 40 &&
        Math.abs(s.x + s.vx * (now - s.t) - d.aimX) < (d.x2 - d.x1) * 0.6);
      if (!blocked && !alreadyShot) targets.push(d);
    }
    return targets;
  }

  // ---------- Desenho ----------
  const COLORS = ['#00e676', '#d500f9', '#ff1744'];
  function draw(dets, targets, fps) {
    const r = canvas.getBoundingClientRect();
    Object.assign(overlay.style, { left: r.left + 'px', top: r.top + 'px', width: r.width + 'px', height: r.height + 'px' });
    if (overlay.width !== canvas.width) overlay.width = canvas.width;
    if (overlay.height !== canvas.height) overlay.height = canvas.height;
    octx.clearRect(0, 0, overlay.width, overlay.height);
    octx.lineWidth = 4;
    octx.font = 'bold 22px sans-serif';
    for (const d of dets) {
      octx.strokeStyle = octx.fillStyle = COLORS[d.cls] ?? '#fff';
      const x1 = d.px1 ?? d.x1, x2 = d.px2 ?? d.x2; // desenha na posição prevista
      octx.strokeRect(x1, d.y1, x2 - x1, d.y2 - d.y1);
      octx.fillText(`${CONFIG.labels[d.cls] ?? d.cls} ${(d.score * 100).toFixed(0)}%`, x1, d.y1 - 6);
    }
    octx.strokeStyle = '#ffeb3b';
    for (const t of targets) {
      octx.beginPath(); octx.arc(t.aimX, t.aimY, 14, 0, Math.PI * 2); octx.stroke();
    }
    octx.fillStyle = 'rgba(0,0,0,.6)';
    octx.fillRect(10, 10, 560, 40);
    octx.fillStyle = '#fff';
    octx.fillText(`YOLOv5n TF.js | bot ${enabled ? 'ON' : 'OFF'} (B) | ${status} | ${fps.toFixed(0)} fps`, 20, 38);
  }

  // ---------- Loop ----------
  const stats = globalThis.__duckBotStats = { frames: 0, shots: 0, detections: {} };
  let fps = 0;

  async function loop() {
    const t0 = performance.now();
    try {
      const dets = await detect();
      updateLanes(dets, t0);
      const targets = chooseTargets(dets, t0);
      if (enabled) {
        for (const t of targets) {
          shoot(t.aimX, t.aimY);
          stats.shots++;
          recentShots.push({ x: t.aimX, y: t.aimY, vx: t.lane.vx, t: performance.now() });
        }
      }
      fps = fps * 0.9 + (1000 / Math.max(1, performance.now() - t0)) * 0.1;
      stats.frames++;
      stats.lanes = lanes.filter(l => l.n > 2).map(l => `${l.group}@${Math.round(l.y)}:${(l.vx * 1000).toFixed(0)}px/s`);
      for (const d of dets) {
        const name = CONFIG.labels[d.cls];
        stats.detections[name] = (stats.detections[name] || 0) + 1;
      }
      draw(dets, enabled ? targets : [], fps);
      // Content scripts rodam num "mundo isolado": a página não enxerga as
      // variáveis deles. Publicamos as estatísticas num atributo do DOM para o
      // main.py / teste E2E lerem (a cada ~15 frames, para não pesar).
      if (stats.frames % 15 === 1) {
        document.documentElement.dataset.duckBot = JSON.stringify({ ...stats, enabled, fps: Math.round(fps), status });
      }
    } catch (err) {
      status = 'erro: ' + err.message;
      console.error('[duck-bot]', err);
    }
    requestAnimationFrame(loop);
  }
  console.log('[duck-bot] modelo carregado, backend:', tf.getBackend());
  loop();
})();
