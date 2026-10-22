// "A posteriori" tab of density_viewer.html : the 1/4° NEMO runs started from the median member of each generation of
// the optimal sweep, one view per experiment (10 years of monthly means, 2 months of daily means, ...). Data in
// density_viewer_aposteriori.js (time series, loaded with the tab) and density_viewer_aposteriori_<experiment>_<constraint>.js
// (snapshots of one run, loaded when the run is shown), both written by build_aposteriori_viewer.py. Frame 0 is the
// initial state, frame i >= 1 the i-th output mean (month or day).
// Uses from the main script : Plotly, D, stretch, DEPTH_MAX, DEPTH_TICKS, bytes, sections, meanGrid, draw (σ₀ sections in
// the style of the Density tab and of paper/fig1_density_profile.py).
(() => {
  const SERIES = [
    ['toce', 'Volume mean T [°C]'], ['soce', 'Volume mean S [g kg⁻¹]'], ['ke', 'Volume mean KE of the output-mean flow [cm² s⁻²]'],
    ['acc', 'Zonal transport through the channel [Sv]'], ['rmse', 'RMSE of the zonal σ₀ section to the data mean'],
    ['sst', 'Area mean SST [°C]'], ['mld', 'Area mean mixed layer depth [m]'],
  ];
  const FIELDS = {
    sst: ['maps', 'SST [°C]'], sss: ['maps', 'SSS [g kg⁻¹]'], ssh: ['maps', 'SSH [m]'], speed: ['maps', 'surface speed [m s⁻¹]'],
    mld: ['maps', 'mixed layer depth [m] (σ₀ > σ₀(10 m) + 0.03)'],
    ke_column: ['maps', 'log₁₀ KE averaged over the water column [cm² s⁻²]'],
    sigma: ['sections', 'zonal mean σ₀ and profile (fig. 1 style)'], toce: ['sections', 'zonal mean T [°C]'], soce: ['sections', 'zonal mean S [g kg⁻¹]'],
  };
  const COLORS = ['#d95f02', '#1b9e77', '#7570b3', '#e7298a', '#66a61e', '#e6ab02'];
  const REF_COLOR = '#333';
  const LABELS = { data: 'Data (training state)', none: 'No constraint', none_border0: 'No constraint · Border 0', zero_mean_border0: 'TS · Border 0',
                   mean_density_border0: 'C1 · Border 0', negative_grad_density_huber_border0: 'C7 (Huber) · Border 0',
                   negative_grad_density_huber_border0_30: 'C7 (Huber, η 30) · Border 0',
                   negative_grad_density_huber_border0_80: 'C7 (Huber, η 80) · Border 0' };
  const NOTES = {
    series: 'One line per run, time 0 is the initial state (the converted median member, instantaneous), then the output means (monthly or daily). Dashed : the 1/4° spin-up restart. RMSE : the σ₀ section interpolated onto the 1° grid of the Density tab, against the mean of its 8 data states ; grey band : the RMSE of each of these data states (min–max). KE from the output-mean velocities (eddies averaged out, less so for daily means), on T points. Transport : thickness-weighted u through the meridional line of the first interior column.',
    hovmoller: 'Horizontal volume mean of each level minus that of the initial state, over time.',
    snapshots: 'Each run : its initial state, the selected output mean and their difference, and when the experiment has the run of training states (data), the selected output mean minus that of the data run at the same time. Last row : the 1/4° spin-up restart, and for σ₀ the mean of the 8 data states of the Density tab (1°). KE map : log₁₀ of the kinetic energy of the output-mean flow averaged over the water column (thickness-weighted), so its difference is a log ratio. MLD colour range capped at the 90th percentile (convection reaches the bottom in a few columns). Shared colour range per field : 1st–99th percentile of the reference ; difference : ±99th percentile of |last frame − initial| over the runs of the experiment.',
  };
  const st = { exp: null, shown: null, frame: 0, field: 'sigma', hov: null, timer: null };
  let P = null, built = false, promise = null;
  const root = document.getElementById('ap');

  // ---------- loading and decoding ----------
  const gunzip = b64 => new Response(new Blob([bytes(b64)]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
  const script = (src, missing) => new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = src; s.onload = resolve; s.onerror = () => reject(new Error(`${src} not found, ${missing}`));
    document.body.appendChild(s);
  });
  async function decodeSnapshots(snap) {
    const out = { frames: snap.frames, maps: {}, sections: {} };
    for (const group of ['maps', 'sections'])
      for (const [k, b] of Object.entries(snap[group])) out[group][k] = new Uint16Array(await gunzip(b));
    return out;
  }
  function load() {
    if (!promise) promise = script('density_viewer_aposteriori.js', 'run build_aposteriori_viewer.py').then(async () => {
      P = window.APOSTERIORI_DATA;
      for (const e of Object.values(P.experiments))
        for (const r of Object.values(e.runs))
          for (const k of Object.keys(r.profiles)) r.profiles[k] = new Float32Array(await gunzip(r.profiles[k]));
      P.reference.snap = await decodeSnapshots(P.reference);
    }).catch(e => { promise = null; throw e; });
    return promise;
  }
  const snaps = new Map();
  function snapshotsOf(run) {
    const key = `${st.exp}/${run}`;
    if (!snaps.has(key)) snaps.set(key, script(`density_viewer_aposteriori_${st.exp}_${run}.js`, 'run build_aposteriori_viewer.py')
      .then(() => decodeSnapshots(window.APOSTERIORI_SNAPSHOTS[key]))
      .catch(e => { snaps.delete(key); throw e; }));
    return snaps.get(key);
  }

  // ---------- helpers ----------
  const E = () => P.experiments[st.exp];
  const runs = () => Object.keys(E().runs);
  const allRuns = () => [...new Set(Object.values(P.experiments).flatMap(e => Object.keys(e.runs)))];
  const color = run => COLORS[allRuns().indexOf(run) % COLORS.length];
  const label = run => LABELS[run] || run;
  const longest = () => Object.values(E().runs).reduce((a, r) => r.days.length > a.days.length ? r : a);
  const nFrames = () => longest().days.length;
  // Time axis of the experiment : years beyond two years of run, days otherwise (360-day calendar)
  const unit = () => longest().days.at(-1) > 720 ? 'years' : 'days';
  const times = run => run.days.map(d => unit() === 'years' ? d / 360 : d);
  const frameTime = i => times(longest())[i];
  const frameText = i => {
    if (i === 0) return 'initial state';
    const freq = longest().freq;
    return freq === '1m' ? `year ${Math.floor((i - 1) / 12) + 1}, month ${(i - 1) % 12 + 1}` : freq === '1d' ? `day ${i}` : `output ${i}`;
  };
  const mapsShape = () => [Math.ceil(P.lat.length / P.map_stride), Math.ceil(P.lon.length / P.map_stride)];
  const strided = a => a.filter((_, i) => i % P.map_stride === 0);
  const depthAxis = (show = true) => ({ range: [stretch(DEPTH_MAX), 0], tickvals: DEPTH_TICKS.map(stretch), ticktext: DEPTH_TICKS.map(String),
                                        showticklabels: show, tickfont: { size: 9 }, zeroline: false, showgrid: false });
  // Frame `f` of field `k` of decoded snapshots as rows of a heatmap, null on land
  function frameGrid(snap, k, f) {
    const group = FIELDS[k][0], [lo, hi] = P[group === 'maps' ? 'map_ranges' : 'section_ranges'][k];
    const [ny, nx] = group === 'maps' ? mapsShape() : [P.depth.length, P.lat.length];
    const q = snap[group][k], off = f * ny * nx, z = [];
    for (let j = 0; j < ny; j++) {
      const row = new Array(nx);
      for (let i = 0; i < nx; i++) { const v = q[off + j * nx + i]; row[i] = v === 0 ? null : lo + (v - 1) / 65534 * (hi - lo); }
      z.push(row);
    }
    return z;
  }
  const diffGrid = (a, b) => a.map((row, j) => row.map((v, i) => v === null || b[j][i] === null ? null : v - b[j][i]));

  // ---------- time series ----------
  const COLS = 4;
  function seriesAxes(n) {
    const row = Math.floor(n / COLS), col = n % COLS, rows = Math.ceil(SERIES.length / COLS);
    const h = 1 / rows, w = 1 / COLS;
    return { x: [col * w + 0.035, (col + 1) * w - 0.015], y: [1 - (row + 1) * h + 0.09, 1 - row * h - 0.07] };
  }
  const axisId = n => n === 0 ? '' : String(n + 1);
  function frameLines(nAxes) {
    const t = frameTime(st.frame);
    return Array.from({ length: nAxes }, (_, n) => ({ type: 'line', xref: `x${axisId(n)}`, yref: `y${axisId(n)} domain`,
      x0: t, x1: t, y0: 0, y1: 1, line: { color: '#999', width: 1, dash: 'dot' } }));
  }
  function drawSeries(div) {
    const traces = [], layout = { margin: { l: 10, r: 10, t: 10, b: 10 }, showlegend: true, paper_bgcolor: 'rgba(0,0,0,0)',
      legend: { orientation: 'v', x: 0.77, y: 0.42, xanchor: 'left', yanchor: 'top', font: { size: 11 } }, annotations: [] };
    const tEnd = frameTime(nFrames() - 1);
    SERIES.forEach(([k, title], n) => {
      const a = seriesAxes(n), id = axisId(n);
      layout[`xaxis${id}`] = { domain: a.x, anchor: `y${id}`, tickfont: { size: 9 }, range: [0, tEnd], title: n >= SERIES.length - COLS ? { text: unit(), font: { size: 10 } } : undefined };
      layout[`yaxis${id}`] = { domain: a.y, anchor: `x${id}`, tickfont: { size: 9 }, zeroline: false };
      layout.annotations.push({ text: title, xref: 'paper', yref: 'paper', x: (a.x[0] + a.x[1]) / 2, y: a.y[1] + 0.01,
                                xanchor: 'center', yanchor: 'bottom', showarrow: false, font: { size: 11 } });
      if (k === 'rmse') {
        const lo = Math.min(...P.data_rmse), hi = Math.max(...P.data_rmse);
        traces.push({ type: 'scatter', x: [0, tEnd, tEnd, 0], y: [lo, lo, hi, hi], fill: 'toself', fillcolor: 'rgba(120,120,120,0.2)',
                      line: { width: 0 }, xaxis: `x${id}`, yaxis: `y${id}`, name: 'data states (min–max)', showlegend: true, hoverinfo: 'skip' });
      }
      const ref = P.reference.series[k];
      if (ref !== null && ref !== undefined)
        traces.push({ type: 'scatter', x: [0, tEnd], y: [ref, ref], mode: 'lines', line: { color: REF_COLOR, dash: 'dash', width: 1.2 },
                      xaxis: `x${id}`, yaxis: `y${id}`, name: '1/4° spin-up restart', showlegend: n === 0, legendgroup: 'ref',
                      hovertemplate: `spin-up %{y:.4g}<extra></extra>` });
      for (const run of st.shown) {
        const r = E().runs[run], y = r.series[k];
        traces.push({ type: 'scatter', x: times(r), y, mode: 'lines', line: { color: color(run), width: 1.5 },
                      xaxis: `x${id}`, yaxis: `y${id}`, name: label(run), legendgroup: run, showlegend: n === 0,
                      hovertemplate: `${label(run)}<br>%{x:.2f} ${unit()} : %{y:.4g}<extra></extra>` });
      }
    });
    layout.shapes = frameLines(SERIES.length);
    Plotly.react(div, traces, layout, { displayModeBar: false, responsive: true });
  }

  // ---------- Hovmöller ----------
  function drawHovmoller(div) {
    const r = E().runs[st.hov], nz = P.depth.length, Y = P.depth.map(stretch), traces = [], layout = {
      margin: { l: 48, r: 10, t: 22, b: 34 }, paper_bgcolor: 'rgba(0,0,0,0)', annotations: [] };
    ['toce', 'soce'].forEach((f, n) => {
      const prof = r.profiles[f], nt = prof.length / nz, t = times(r), id = axisId(n);
      const z = Array.from({ length: nz }, (_, k) => Array.from({ length: nt }, (_, i) => {
        const v = prof[i * nz + k] - prof[k]; return Number.isFinite(v) ? v : null; }));
      const m = Math.max(1e-6, ...z.flat().filter(v => v !== null).map(Math.abs));
      const x0 = n * 0.5, x1 = x0 + 0.42;
      traces.push({ type: 'heatmap', x: t, y: Y, z, xaxis: `x${id}`, yaxis: `y${id}`, colorscale: P.colorscales.diff, zmin: -m, zmax: m,
                    customdata: P.depth.map(d => t.map(() => d)),
                    colorbar: { x: x1 + 0.005, len: 0.9, thickness: 9, tickfont: { size: 9 } },
                    hovertemplate: `%{x:.2f} ${unit()}, %{customdata:.0f} m : %{z:.3g}<extra></extra>` });
      layout[`xaxis${id}`] = { domain: [x0, x1], anchor: `y${id}`, tickfont: { size: 9 }, title: { text: unit(), font: { size: 10 } } };
      layout[`yaxis${id}`] = { ...depthAxis(n === 0), anchor: `x${id}`, title: n === 0 ? { text: 'depth [m]', font: { size: 10 } } : undefined };
      layout.annotations.push({ text: `${f === 'toce' ? 'T [°C]' : 'S [g kg⁻¹]'} minus initial, ${label(st.hov)}`, xref: 'paper', yref: 'paper',
                                x: (x0 + x1) / 2, y: 1.02, xanchor: 'center', yanchor: 'bottom', showarrow: false, font: { size: 11 } });
    });
    layout.shapes = frameLines(2);
    Plotly.react(div, traces, layout, { displayModeBar: false, responsive: true });
  }

  // ---------- snapshots ----------
  // σ₀ states go through the Density tab `draw` (contours, colour scale and BAND profile inset of figure 1)
  const GRID = () => ({ latitude: P.lat, depth: P.depth });
  function heatmap(div, z, k, diff, title, x, y) {
    if (k === 'sigma' && !diff) { draw(div, z, true, x ? { latitude: x, depth: y } : GRID(), title); return; }
    const group = FIELDS[k][0], d = E().display[k];
    const scale = diff ? { colorscale: P.colorscales.diff, zmin: -d.diff, zmax: d.diff }
                       : { colorscale: P.colorscales[k], zmin: d.range[0], zmax: d.range[1] };
    const isMap = group === 'maps';
    x ??= isMap ? strided(P.lon) : P.lat;
    y ??= isMap ? strided(P.lat) : P.depth.map(stretch);
    const trace = { type: 'heatmap', x, y, z, ...scale, colorbar: { len: 0.9, thickness: 8, tickfont: { size: 9 } },
                    hovertemplate: `${isMap ? 'lon %{x:.1f}°, lat %{y:.1f}°' : 'lat %{x:.1f}°'} : %{z:.3f}<extra></extra>` };
    const layout = { margin: { l: isMap ? 34 : 44, r: 4, t: 20, b: 26 }, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'lightgray',
                     title: { text: title, font: { size: 11 }, x: 0.02, y: 0.98 },
                     xaxis: { tickfont: { size: 9 }, zeroline: false, showgrid: false, ...(isMap ? { scaleanchor: 'y', constrain: 'domain' } : {}) },
                     yaxis: isMap ? { tickfont: { size: 9 }, zeroline: false, showgrid: false } : depthAxis() };
    Plotly.react(div, [trace], layout, { displayModeBar: false, responsive: true });
  }
  function buildSnapshotGrid() {
    const g = root.querySelector('#ap-snaps');
    const vsData = hasData();
    const rows = [...st.shown].map(run => `<div class="row-label" style="color:${color(run)}">${label(run)} · member ${E().runs[run].member + 1}</div>
      <div class="ap-row${vsData ? ' four' : ''}" data-run="${run}"><div class="ap-p" data-k="initial"></div><div class="ap-p" data-k="frame"></div><div class="ap-p" data-k="diff"></div>${vsData && run !== DATA ? '<div class="ap-p" data-k="vsdata"></div>' : ''}</div>`);
    rows.push(`<div class="row-label">References</div><div class="ap-row" data-run="reference"><div class="ap-p" data-k="spinup"></div><div class="ap-p" data-k="data"></div><div></div></div>`);
    g.innerHTML = rows.join('');
    const isMap = FIELDS[st.field][0] === 'maps', h = isMap ? 520 : 240;
    g.querySelectorAll('.ap-row').forEach(r => r.classList.toggle('maps', isMap));
    g.querySelectorAll('.ap-p').forEach(p => { p.style.height = `${h}px`; });
  }
  // Run of training states (a_posteriori step 0) : the other runs are also shown minus it, at the same time
  const DATA = 'data';
  const hasData = () => DATA in E().runs;
  async function drawSnapshots(changedField = false) {
    const g = root.querySelector('#ap-snaps'), k = st.field, f = st.frame;
    if (changedField) {
      buildSnapshotGrid();
      const row = g.querySelector('.ap-row[data-run="reference"]');
      heatmap(row.querySelector('[data-k=spinup]'), frameGrid(P.reference.snap, k, 0), k, false, `1/4° spin-up restart (${P.reference.path.split('/').at(-1)})`);
      const data = row.querySelector('[data-k=data]');
      if (k === 'sigma') heatmap(data, meanGrid(sections(D.data), D.data.density_error.length), k, false,
                                 'data mean, 8 states (1°, Density tab)', D.latitude, D.depth);
      else { Plotly.purge(data); data.innerHTML = ''; }
    }
    for (const run of st.shown) {
      const row = g.querySelector(`.ap-row[data-run="${run}"]`);
      let snap;
      try { snap = await snapshotsOf(run); } catch (e) { row.textContent = e.message; continue; }
      if (k !== st.field || f !== st.frame) return;
      const fr = Math.min(f, snap.frames - 1), z0 = frameGrid(snap, k, 0), zf = frameGrid(snap, k, fr);
      const vs = row.querySelector('[data-k=vsdata]');
      if (vs) {
        let dataSnap;
        try { dataSnap = await snapshotsOf(DATA); } catch (e) { vs.textContent = e.message; }
        if (k !== st.field || f !== st.frame) return;
        if (dataSnap) heatmap(vs, diffGrid(zf, frameGrid(dataSnap, k, Math.min(fr, dataSnap.frames - 1))), k, true, `${frameText(fr)} − data run, same time`);
      }
      if (changedField || !row.dataset.drawn) { heatmap(row.querySelector('[data-k=initial]'), z0, k, false, 'initial state'); row.dataset.drawn = 1; }
      heatmap(row.querySelector('[data-k=frame]'), zf, k, false, frameText(fr));
      heatmap(row.querySelector('[data-k=diff]'), diffGrid(zf, z0), k, true, `${frameText(fr)} − initial`);
    }
  }

  // ---------- controls ----------
  function setFrame(f) {
    st.frame = f;
    root.querySelector('#ap-frame').value = f;
    root.querySelector('#ap-frame-out').textContent = frameText(f);
    ['#ap-series', '#ap-hov'].forEach(id => { const div = root.querySelector(id); if (div.data) Plotly.relayout(div, { shapes: frameLines(div === root.querySelector('#ap-series') ? SERIES.length : 2) }); });
    return drawSnapshots();
  }
  function play() {
    const button = root.querySelector('#ap-play');
    if (st.timer) { clearTimeout(st.timer); st.timer = null; button.textContent = '▶ Play'; return; }
    button.textContent = '❚❚ Pause';
    const step = async () => {
      const next = st.frame + 1 >= nFrames() ? 0 : st.frame + 1;
      await setFrame(next);
      if (st.timer) st.timer = setTimeout(step, 120);
    };
    st.timer = setTimeout(step, 0);
  }
  function build() {
    st.exp ??= Object.keys(P.experiments)[0];
    root.innerHTML = `
      <div class="nc-bar"><label>Experiment <select id="ap-exp">${Object.keys(P.experiments).map(e =>
        `<option value="${e}"${e === st.exp ? ' selected' : ''}>${e}</option>`).join('')}</select></label>
        <span class="row-label">Runs</span><span id="ap-runs"></span></div>
      <div class="nc-note" id="ap-info"></div><div id="ap-body"></div>`;
    root.querySelector('#ap-exp').addEventListener('change', e => {
      if (st.timer) play();
      st.exp = e.target.value;
      buildBody(); render();
    });
    buildBody();
    built = true;
  }
  // Controls and panels of the selected experiment
  function buildBody() {
    st.shown = new Set(runs());
    st.hov = runs()[0];
    st.frame = 0;
    const span = r => { const d = r.days.at(-1) + (r.days[1] ?? 0); return d > 720 ? `${(d / 360).toFixed(0)} years` : `${d.toFixed(0)} days`; };
    root.querySelector('#ap-info').innerHTML = runs().map(run => { const r = E().runs[run];
      return `<span style="color:${color(run)}">■</span> ${label(run)} : member ${r.member + 1}, ${r.generation.split('/').at(-1)}, ${span(r)} (${r.freq} means) in ${r.elapsed}`; }).join(' &nbsp;·&nbsp; ');
    root.querySelector('#ap-runs').innerHTML = runs().map(run => `<label><input type="checkbox" value="${run}" checked>
          <span style="color:${color(run)}">${label(run)}</span></label>`).join(' ');
    root.querySelector('#ap-body').innerHTML = `
      <div class="nc-grid">
        <div class="col wide"><div class="head"><h2>Time series</h2></div><div class="nc-note">${NOTES.series}</div><div id="ap-series" style="height:440px"></div></div>
        <div class="col wide"><div class="head"><h2>Drift of the layers</h2>
          <div class="nc-ctl"><label>Run <select id="ap-hov-run">${runs().map(run => `<option value="${run}">${label(run)}</option>`).join('')}</select></label></div></div>
          <div class="nc-note">${NOTES.hovmoller}</div><div id="ap-hov" style="height:260px"></div></div>
        <div class="col wide"><div class="head"><h2>Snapshots</h2>
          <div class="nc-ctl">
            <label>Field <select id="ap-field">${Object.entries(FIELDS).map(([k, [, t]]) => `<option value="${k}"${k === st.field ? ' selected' : ''}>${t}</option>`).join('')}</select></label>
            <label>Time <input type="range" id="ap-frame" min="0" max="${nFrames() - 1}" value="0" style="width:260px"><output id="ap-frame-out">initial state</output></label>
            <button id="ap-play" class="ap-btn">▶ Play</button>
          </div></div>
          <div class="nc-note">${NOTES.snapshots}</div><div id="ap-snaps" class="ap-snaps"></div></div>
      </div>`;
    const on = (sel, ev, fn) => root.querySelector(sel).addEventListener(ev, fn);
    root.querySelectorAll('#ap-runs input').forEach(i => i.addEventListener('change', () => {
      i.checked ? st.shown.add(i.value) : st.shown.delete(i.value);
      drawSeries(root.querySelector('#ap-series')); drawSnapshots(true);
    }));
    on('#ap-hov-run', 'change', e => { st.hov = e.target.value; drawHovmoller(root.querySelector('#ap-hov')); });
    on('#ap-field', 'change', e => { st.field = e.target.value; drawSnapshots(true); });
    on('#ap-frame', 'input', e => setFrame(+e.target.value));
    on('#ap-play', 'click', play);
  }
  function render() {
    drawSeries(root.querySelector('#ap-series'));
    drawHovmoller(root.querySelector('#ap-hov'));
    drawSnapshots(true);
  }

  // Redraw of the snapshots after a page-wide display option changed (continuous σ₀ colours)
  window.redrawAposteriori = () => { if (built) drawSnapshots(true); };
  window.showAposteriori = () => {
    const status = document.getElementById('loading');
    if (!P) status.textContent = 'Loading a-posteriori data…';
    return load().then(() => { status.textContent = ''; if (!built) { build(); render(); } },
                       e => { status.textContent = e.message; });
  };
})();
