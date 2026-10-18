// "No constraint" tab of density_viewer.html : data against the unconstrained runs, to show what the unconstrained model
// has learned. Data in density_viewer_noconstraint.js (build_noconstraint_viewer.py), loaded with the tab.
// Uses from the main script : Plotly, D (sigma colorscale and range), stretch, DEPTH_MAX, DEPTH_TICKS, CLUSTER_COLORS,
// clusterColorscale, bytes.
(() => {
  const FIGS = {
    layers: ['A · Layers without their mean', 'Top : temperature or salinity minus the basin mean of its level, one data state next to two generated states, shared colour scale per depth (±99th percentile of the data), warm / salty red. Bottom : k-means (k = 5) of the (T, S) of the ocean cells of each level, fit on the 32 data states with their level mean removed, T and S scaled by their data standard deviation at that level, clusters numbered by increasing σ₀ of their centre. Offset removed : every state with its own level mean removed (the maps above). Raw : the data level mean removed, the offset of the generated member kept. Agreement : share of the ocean cells with the most frequent data label, mean over the members (data : each state against the other 31). Sections : zonal mean over the ocean cells of T and S of the same states, raw (the offset of each level of a generated state shows as a horizontal stripe) or with the basin mean of each level of that state removed ; colour scale from the 8 stored data states.'],
    shifts: ['B · Each layer has its own shift', 'Basin mean of each level, one dot per member. Physical : the basin mean itself. σ : its distance to the data mean in spatial standard deviations of the level in the data. One x axis for all levels, members outside it drawn as triangles on its edge (hover for the value). Right of each row : rms distance to the data mean. Right : correlation of the generated basin means between levels, over the members.'],
    offsets: ['1 · Where the error lives', 'Basin mean of T and S of each level, per member (data grey, run orange), and the share of the member-to-member variance it explains.'],
    maps: ['2 · Maps without the offset', 'Anomaly to the basin mean of the level. Run raw : relative to the data basin mean, the offset of the member shows as a uniform shift.'],
    corr: ['3 · Pattern correlation', 'Correlation of each member anomaly with the data mean anomaly, per level. Zonally asymmetric : zonal mean removed from both (gyres, boundary currents), the latitude-only field scores 0. Data : each member against the mean of the others.'],
    recentred: ['4 · Recentred water masses', 'Raw run, and the run with the basin mean of every level of every member moved onto the data one. T–S diagrams (batch average), k-means cluster maps (most frequent cluster, agreement with the data map) and cluster profiles (dashed data, solid run).'],
    mean: ['5 · Ensemble mean', 'Mean over the members : zonal σ₀ section and SST, with the difference run − data.'],
    spectra: ['6 · Small scales', 'Power spectrum along latitude of the anomaly, columns that are ocean from south to north, members averaged. Last panel : ratio run / data.'],
  };
  const RUN_COLOR = '#d95f02', DATA_COLOR = '#555';
  // Plotly's 'RdBu' runs from blue (low) to red (high) : warm anomalies red
  const DIVERGING = 'RdBu';
  const TS_SCALES = { rest: [[0, '#fee5d9'], [0.5, '#fb6a4a'], [1, '#67000d']], acc: [[0, '#deebf7'], [0.5, '#4292c6'], [1, '#08306b']] };
  // Light grey behind the map subplots (land), white elsewhere
  const landBackground = (layout, ax) => (layout.shapes ??= []).push({ type: 'rect', xref: `${ax.xaxis} domain`, yref: `${ax.yaxis} domain`,
    x0: 0, x1: 1, y0: 0, y1: 1, fillcolor: 'lightgray', line: { width: 0 }, layer: 'below' });
  const RUN_LABELS = { none: 'No constraint', none_border0: 'No constraint · Border 0' };
  // Level clusters ordered by increasing σ₀ : light (red) to dense (dark blue)
  const LEVEL_COLORS = ['#d73027', '#fc8d59', '#fee090', '#91bfdb', '#4575b4', '#313695', '#542788', '#1b1b1b'];
  const levelColorscale = n => LEVEL_COLORS.slice(0, n).flatMap((c, k) => [[k / n, c], [(k + 1) / n, c]]);
  const st = { run: 'none', shown: new Set(['layers', 'shifts']), member: 0, level: 0, field: 'toce', metric: 'asym',
               layers: { field: 'toce', data: 0, gen: [0, 1], clusters: 'anom', sections: 'raw' }, shifts: { field: 'toce', units: 'phys' } };
  const FIELD_LABEL = { toce: 'T [°C]', soce: 'S [g kg⁻¹]' };
  let N = null, built = false, promise = null;

  function load() {
    if (!promise) promise = new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = 'density_viewer_noconstraint.js';
      s.onload = () => { N = window.NOCONSTRAINT_DATA; decodeLabels().then(resolve, reject); };
      s.onerror = () => { promise = null; reject(new Error('density_viewer_noconstraint.js not found, run build_noconstraint_viewer.py')); };
      document.body.appendChild(s);
    });
    return promise;
  }

  // ---------- decoding ----------
  const memo = new Map();
  const once = (key, fn) => { if (!memo.has(key)) memo.set(key, fn()); return memo.get(key); };
  const f32 = b64 => once(b64, () => new Float32Array(bytes(b64).buffer));
  const u8 = b64 => once(b64, () => bytes(b64));
  const maps = (entry, f) => once(entry.maps[f], () => {
    const r = N.anomaly_range[f];
    return Float32Array.from(new Uint16Array(bytes(entry.maps[f]).buffer), q => q === 0 ? NaN : -r + (q - 1) / 65534 * 2 * r);
  });
  const run = () => N.runs[st.run];
  // Level cluster labels are gzip compressed (build_noconstraint_viewer.gz64), decoded once at load
  const levelLabels = new Map();
  const gunzip = b64 => new Response(new Blob([bytes(b64)]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer()
    .then(b => new Uint8Array(b));
  const decodeLabels = () => Promise.all([N.data, ...Object.values(N.runs)].flatMap(e => Object.values(e.level_clusters))
    .map(({ labels }) => gunzip(labels).then(a => levelLabels.set(labels, a))));
  const NY = () => N.lat.length, NX = () => N.lon.length;
  const toGrid = (flat, offset, ny, nx, fn = v => v) => {
    const z = [];
    for (let j = 0; j < ny; j++) {
      const row = new Array(nx);
      for (let i = 0; i < nx; i++) { const v = flat[offset + j * nx + i]; row[i] = Number.isNaN(v) ? null : fn(v); }
      z.push(row);
    }
    return z;
  };
  const meanOf = a => a.reduce((s, v) => s + v, 0) / a.length;
  const nanMean = a => { const v = a.filter(Number.isFinite); return v.length ? meanOf(v) : null; };
  const column = (rows, k) => rows.map(r => r[k]);
  const levelMean = (entry, f) => N.depth.map((_, k) => nanMean(column(entry.offset[f], k)));
  const Y = () => N.depth.map(stretch);
  const depthAxis = (show = true) => ({ range: [stretch(DEPTH_MAX), 0], tickvals: DEPTH_TICKS.map(stretch), ticktext: DEPTH_TICKS.map(String),
                                        showticklabels: show, tickfont: { size: 9 }, zeroline: false, showgrid: false });

  // Subplot helper : axes number n (1-based) on the paper domain [x0, x1] x [y0, y1], returns the trace axis ids
  function sub(layout, n, [x0, x1], [y0, y1], xopt = {}, yopt = {}, title = '') {
    const s = n === 1 ? '' : String(n);
    layout[`xaxis${s}`] = { domain: [x0, x1], anchor: `y${s}`, tickfont: { size: 9 }, zeroline: false, ...xopt };
    layout[`yaxis${s}`] = { domain: [y0, y1], anchor: `x${s}`, tickfont: { size: 9 }, zeroline: false, ...yopt };
    if (title) layout.annotations.push({ text: title, x: (x0 + x1) / 2, y: y1, xref: 'paper', yref: 'paper', yanchor: 'bottom',
                                         showarrow: false, font: { size: 11 } });
    return { xaxis: `x${s}`, yaxis: `y${s}` };
  }
  const baseLayout = () => ({ margin: { l: 46, r: 10, t: 22, b: 36 }, showlegend: false, paper_bgcolor: 'rgba(0,0,0,0)',
                              plot_bgcolor: 'white', annotations: [] });
  const axTitle = (text) => ({ text, font: { size: 10 }, standoff: 2 });
  const plot = (div, traces, layout) => Plotly.react(div, traces, layout, { displayModeBar: false, responsive: true });

  // ---------- figures ----------
  function figOffsets(div) {
    const r = run(), layout = baseLayout(), traces = [], y = Y();
    ['toce', 'soce'].forEach((f, i) => {
      const ax = sub(layout, i + 1, [i * 0.34, i * 0.34 + 0.29], [0, 1], { title: axTitle(`basin mean ${FIELD_LABEL[f]}`), nticks: 5 },
                     { ...depthAxis(i === 0), title: i === 0 ? axTitle('depth [m]') : undefined }, FIELD_LABEL[f].split(' ')[0] + ' basin mean');
      for (const [entry, color, width, who] of [[r, RUN_COLOR, 1, 'run'], [N.data, DATA_COLOR, 0.8, 'data']])
        entry.offset[f].forEach((prof, b) => traces.push({ type: 'scatter', mode: 'lines', x: prof, y, customdata: N.depth, ...ax,
          line: { color, width }, opacity: who === 'data' ? 0.6 : 0.8,
          hovertemplate: `${who} member ${b + 1}<br>%{x:.3f}<br>depth %{customdata:.0f} m<extra></extra>` }));
    });
    const ax = sub(layout, 3, [0.71, 1], [0, 1], { title: axTitle('share of member variance in the basin mean'), range: [0, 1.02], nticks: 6, showgrid: true, gridcolor: '#eee' },
                   depthAxis(false), 'Share explained by the offset');
    for (const [entry, color, who] of [[r, RUN_COLOR, 'run'], [N.data, DATA_COLOR, 'data']])
      ['toce', 'soce'].forEach((f, i) => traces.push({ type: 'scatter', mode: 'lines+markers', ...ax, y, customdata: N.depth,
        x: entry.var_raw[f].map((v, k) => v > 0 ? 1 - entry.var_anom[f][k] / v : null),
        line: { color, width: 2, dash: i ? 'dot' : 'solid' }, marker: { size: 4, color },
        hovertemplate: `${who} ${f === 'toce' ? 'T' : 'S'}<br>share %{x:.3f}<br>depth %{customdata:.0f} m<extra></extra>` }));
    layout.annotations.push({ text: 'solid T · dotted S', x: 0.995, y: 0.02, xref: 'paper', yref: 'paper', xanchor: 'right',
                              showarrow: false, font: { size: 10, color: '#777' } });
    plot(div, traces, layout);
  }

  function figMaps(div) {
    const r = run(), f = st.field, l = st.level, ny = NY(), nx = NX(), P = ny * nx, L = N.map_levels.length, k = N.map_levels[l];
    const md = Math.min(st.member, N.data.n_maps - 1), mr = Math.min(st.member, r.n_maps - 1);
    const dm = maps(N.data, f), rm = maps(r, f);
    const dataLevelMean = levelMean(N.data, f)[k], offset = r.offset[f][mr][k] - dataLevelMean;
    let R = 0;
    for (let i = 0; i < P; i++) { const v = Math.abs(dm[(md * L + l) * P + i]); if (v > R) R = v; }
    R = Math.max(R, 1e-6);
    const panels = [
      ['Data member ' + (md + 1), toGrid(dm, (md * L + l) * P, ny, nx), N.data.corr[f][st.metric][md][k]],
      [`Run member ${mr + 1}, offset removed`, toGrid(rm, (mr * L + l) * P, ny, nx), r.corr[f][st.metric][mr][k]],
      [`Run member ${mr + 1}, raw (offset ${offset >= 0 ? '+' : ''}${offset.toFixed(2)})`, toGrid(rm, (mr * L + l) * P, ny, nx, v => v + offset), null],
    ];
    const layout = baseLayout(), traces = [];
    const W = 0.2;
    panels.forEach(([title, z, c], i) => {
      const ax = sub(layout, i + 1, [i * (W + 0.03), i * (W + 0.03) + W], [0, 1], { nticks: 3, showgrid: false },
                     { showticklabels: i === 0, showgrid: false, range: [N.lat[0], N.lat[ny - 1]] }, title);
      landBackground(layout, ax);
      traces.push({ type: 'heatmap', x: N.lon, y: N.lat, z, ...ax, colorscale: DIVERGING, zmin: -R, zmax: R, zsmooth: false,
                    showscale: i === 2, colorbar: { thickness: 9, len: 0.9, x: 3 * (W + 0.03) - 0.02, tickfont: { size: 9 },
                                                    title: { text: FIELD_LABEL[f], side: 'right', font: { size: 10 } } },
                    hovertemplate: `%{z:.3f}<extra></extra>` });
      if (c !== null) layout.annotations.push({ text: `r ${st.metric === 'asym' ? '(zonally asym.)' : '(full)'} = ${c.toFixed(3)}`,
        x: i * (W + 0.03) + W / 2, y: -0.09, xref: 'paper', yref: 'paper', showarrow: false, font: { size: 10 } });
    });
    layout.margin.b = 40;
    // Correlation profiles of the shown members, as a reminder of the depth dependence
    const ax = sub(layout, 4, [0.76, 1], [0, 1], { title: axTitle(`r ${st.metric === 'asym' ? 'zonally asymmetric' : 'full'}`), nticks: 4, showgrid: true, gridcolor: '#eee' },
                   depthAxis(true), 'r of these members');
    traces.push({ type: 'scatter', mode: 'lines', x: N.data.corr[f][st.metric][md], y: Y(), ...ax, line: { color: DATA_COLOR, width: 2 }, hoverinfo: 'x' },
                { type: 'scatter', mode: 'lines', x: r.corr[f][st.metric][mr], y: Y(), ...ax, line: { color: RUN_COLOR, width: 2 }, hoverinfo: 'x' },
                { type: 'scatter', mode: 'markers', x: [r.corr[f][st.metric][mr][k]], y: [stretch(N.depth[k])], ...ax,
                  marker: { color: RUN_COLOR, size: 9, symbol: 'diamond' }, hoverinfo: 'skip' });
    plot(div, traces, layout);
  }

  function figCorr(div) {
    const r = run(), layout = baseLayout(), traces = [], y = Y(), metric = st.metric;
    ['toce', 'soce'].forEach((f, i) => {
      const x0 = i * 0.52, ax = sub(layout, i + 1, [x0, x0 + 0.46], [0, 1],
        { title: axTitle(`pattern correlation (${metric === 'asym' ? 'zonally asymmetric' : 'full'})`), nticks: 6, showgrid: true, gridcolor: '#eee',
          range: metric === 'asym' ? [-0.05, 1] : undefined },
        { ...depthAxis(i === 0), title: i === 0 ? axTitle('depth [m]') : undefined }, f === 'toce' ? 'Temperature' : 'Salinity');
      r.corr[f][metric].forEach((c, b) => traces.push({ type: 'scatter', mode: 'lines', x: c, y, ...ax, line: { color: RUN_COLOR, width: 0.7 },
                                                          opacity: 0.45, hoverinfo: 'skip' }));
      const mean = N.depth.map((_, k) => nanMean(column(r.corr[f][metric], k)));
      const dmean = N.depth.map((_, k) => nanMean(column(N.data.corr[f][metric], k)));
      traces.push({ type: 'scatter', mode: 'lines', x: mean, y, customdata: N.depth, ...ax, line: { color: RUN_COLOR, width: 3 },
                    hovertemplate: 'run mean r %{x:.3f}<br>depth %{customdata:.0f} m<extra></extra>' },
                  { type: 'scatter', mode: 'lines', x: dmean, y, customdata: N.depth, ...ax, line: { color: DATA_COLOR, width: 2, dash: 'dash' },
                    hovertemplate: 'data vs data r %{x:.3f}<br>depth %{customdata:.0f} m<extra></extra>' });
      if (metric === 'full') traces.push({ type: 'scatter', mode: 'lines', x: N.lat_only[f], y, customdata: N.depth, ...ax,
                                           line: { color: '#999', width: 2, dash: 'dot' },
                                           hovertemplate: 'latitude-only field r %{x:.3f}<br>depth %{customdata:.0f} m<extra></extra>' });
    });
    layout.annotations.push({ text: `<span style="color:${RUN_COLOR}">run members / mean</span> · data vs data (dashed)` +
                                    (metric === 'full' ? ' · latitude-only field (dotted)' : ''),
                              x: 1, y: -0.13, xref: 'paper', yref: 'paper', xanchor: 'right', showarrow: false, font: { size: 10, color: '#555' } });
    layout.margin.b = 44;
    plot(div, traces, layout);
  }

  function modeMap(labels, n, K) {
    const ny = NY(), nx = NX(), P = ny * nx, out = new Uint8Array(P), counts = new Uint16Array(K + 1);
    for (let p = 0; p < P; p++) {
      counts.fill(0);
      for (let b = 0; b < n; b++) counts[labels[b * P + p]]++;
      let best = 0;
      for (let k = 1; k <= K; k++) if (counts[k] > counts[best] || (best === 0 && counts[k])) best = k;
      out[p] = best;
    }
    return out;
  }
  function agreement(labels, n, ref) {
    const P = ref.length;
    const per = [];
    for (let b = 0; b < n; b++) {
      let same = 0, ocean = 0;
      for (let p = 0; p < P; p++) if (ref[p]) { ocean++; if (labels[b * P + p] === ref[p]) same++; }
      per.push(same / ocean);
    }
    return meanOf(per);
  }

  function figRecentred(div) {
    const r = run(), K = N.n_clusters, ny = NY(), nx = NX(), P = ny * nx, nt = N.ts_t.length, ns = N.ts_s.length;
    const nData = N.data.offset.toce.length, nRun = r.offset.toce.length;
    const dataLabels = u8(N.data.raw.labels), dataMode = modeMap(dataLabels, nData, K);
    const cols = [['Data', N.data.raw, nData], ['Run raw', r.raw, nRun], ['Run recentred', r.recentred, nRun]];
    const layout = { ...baseLayout(), margin: { l: 46, r: 10, t: 22, b: 30 } }, traces = [];
    const rows = [[0.71, 1], [0.37, 0.6], [0, 0.27]];
    let n = 1;
    cols.forEach(([title, entry, nb], c) => {
      const x = [c * 0.34, c * 0.34 + 0.3];
      // T-S diagram, (region, T, S) volume fractions, region 0 = ACC, 1 = rest
      const hist = f32(entry.ts), ax1 = sub(layout, n++, x, rows[0], { range: [33.5, 37.5], nticks: 5, title: axTitle('S [g kg⁻¹]') },
        { range: [-2, 28], title: c === 0 ? axTitle('T [°C]') : undefined }, `${title} : T–S`);
      [[1, TS_SCALES.rest, 1], [0, TS_SCALES.acc, 0.8]].forEach(([reg, cs, op]) => traces.push({ type: 'heatmap', x: N.ts_s, y: N.ts_t, ...ax1,
        z: toGrid(hist, reg * nt * ns, nt, ns, v => v > 0 ? Math.log10(v) : null), colorscale: cs, zmin: -6, zmax: -1, showscale: false, opacity: op,
        hovertemplate: `${reg ? 'rest' : 'ACC'}<br>S %{x:.2f}<br>T %{y:.1f}<br>log₁₀ vol %{z:.2f}<extra></extra>` }));
      traces.push({ type: 'contour', x: N.ts_s, y: N.ts_t, z: N.ts_sigma, ...ax1, showscale: false, hoverinfo: 'skip',
                    contours: { start: 22, end: 28.5, size: 0.5, coloring: 'none', showlabels: true, labelfont: { size: 8, color: '#888' } },
                    line: { color: '#aaa', width: 0.6 } });
      // Cluster map (most frequent cluster) and its agreement with the data map
      const labels = u8(entry.labels), mode = c === 0 ? dataMode : modeMap(labels, nb, K);
      const agree = agreement(labels, nb, dataMode);
      const ax2 = sub(layout, n++, [x[0] + 0.08, x[1] - 0.08], rows[1], { nticks: 3, showgrid: false }, { showgrid: false, range: [N.lat[0], N.lat[ny - 1]] },
                      `${title} : clusters · agreement ${(100 * agree).toFixed(0)} %`);
      landBackground(layout, ax2);
      traces.push({ type: 'heatmap', x: N.lon, y: N.lat, z: toGrid(Float32Array.from(mode, v => v || NaN), 0, ny, nx), ...ax2,
                    colorscale: clusterColorscale(K), zmin: 0.5, zmax: K + 0.5, zsmooth: false, showscale: false, hovertemplate: 'cluster %{z}<extra></extra>' });
      // Cluster profiles (K, 2, nz) : T on the left half, S on the right half of the column
      const prof = f32(entry.profiles), nz = N.depth.length, dprof = f32(N.data.raw.profiles);
      ['T', 'S'].forEach((name, fi) => {
        const xs = fi ? [x[0] + 0.16, x[1]] : [x[0], x[0] + 0.14];
        const ax3 = sub(layout, n++, xs, rows[2], { nticks: 3, showgrid: true, gridcolor: '#eee', title: axTitle(name === 'T' ? 'T [°C]' : 'S [g kg⁻¹]') },
                        depthAxis(fi === 0 && c === 0), `${name}(z)`);
        for (let k = 0; k < K; k++) {
          const line = (src) => N.depth.map((_, z) => { const v = src[(k * 2 + fi) * nz + z]; return Number.isFinite(v) ? v : null; });
          traces.push({ type: 'scatter', mode: 'lines', x: line(dprof), y: Y(), ...ax3, line: { color: CLUSTER_COLORS[k], width: 1, dash: 'dash' }, hoverinfo: 'skip' });
          if (c) traces.push({ type: 'scatter', mode: 'lines', x: line(prof), y: Y(), customdata: N.depth, ...ax3, line: { color: CLUSTER_COLORS[k], width: 2 },
                               hovertemplate: `cluster ${k + 1}<br>${name} %{x:.2f}<br>depth %{customdata:.0f} m<extra></extra>` });
        }
      });
    });
    plot(div, traces, layout);
  }

  function figMean(div) {
    const r = run(), ny = NY(), nx = NX(), nz = N.depth.length, y = Y();
    const ds = f32(N.data.section_mean), rs = f32(r.section_mean), dsst = f32(N.data.sst_mean), rsst = f32(r.sst_mean);
    const diff = (a, b) => Float32Array.from(a, (v, i) => v - b[i]);
    const layout = baseLayout(), traces = [];
    const W = 0.3;
    const secAxis = (c) => ({ ...depthAxis(c === 0), title: c === 0 ? axTitle('depth [m]') : undefined });
    [['Data mean σ₀', ds], ['Run mean σ₀', rs], ['Run − data σ₀', diff(rs, ds)]].forEach(([title, a], c) => {
      const ax = sub(layout, c + 1, [c * (W + 0.035), c * (W + 0.035) + W], [0.55, 1], { nticks: 5, showgrid: false }, secAxis(c), title);
      const isDiff = c === 2;
      landBackground(layout, ax);
      traces.push({ type: 'heatmap', x: N.lat, y, z: toGrid(a, 0, nz, ny), ...ax, zsmooth: false, showscale: isDiff,
                    colorscale: isDiff ? DIVERGING : D.colorscale, zmin: isDiff ? -1 : 22, zmax: isDiff ? 1 : 28.5,
                    colorbar: { thickness: 8, len: 0.42, y: 0.78, tickfont: { size: 9 } }, hovertemplate: '%{z:.2f}<extra></extra>' });
    });
    [['Data mean SST', dsst], ['Run mean SST', rsst], ['Run − data SST', diff(rsst, dsst)]].forEach(([title, a], c) => {
      const x0 = c * (W + 0.035) + W / 2 - 0.07;
      const ax = sub(layout, c + 4, [x0, x0 + 0.14], [0, 0.45], { nticks: 3, showgrid: false }, { showgrid: false, range: [N.lat[0], N.lat[ny - 1]] }, title);
      const isDiff = c === 2;
      landBackground(layout, ax);
      traces.push({ type: 'heatmap', x: N.lon, y: N.lat, z: toGrid(a, 0, ny, nx), ...ax, zsmooth: false, showscale: isDiff,
                    colorscale: isDiff ? DIVERGING : 'Portland', zmin: isDiff ? -3 : 0, zmax: isDiff ? 3 : 27,
                    colorbar: { thickness: 8, len: 0.42, y: 0.22, tickfont: { size: 9 } }, hovertemplate: '%{z:.2f} °C<extra></extra>' });
    });
    plot(div, traces, layout);
  }

  function figSpectra(div) {
    const r = run(), f = st.field, layout = baseLayout(), traces = [], L = N.map_levels.length;
    const w = 0.22;
    N.map_depths.forEach((d, l) => {
      const ax = sub(layout, l + 1, [l * (w + 0.03), l * (w + 0.03) + w], [0, 1],
        { type: 'log', autorange: 'reversed', title: axTitle('wavelength [grid cells]'), showgrid: true, gridcolor: '#eee' },
        { type: 'log', title: l === 0 ? axTitle('power') : undefined, showgrid: true, gridcolor: '#eee', exponentformat: 'power' }, `${d.toFixed(0)} m`);
      const x = N.spectrum_wavelength[l];
      traces.push({ type: 'scatter', mode: 'lines', x, y: N.data.spectra[f][l], ...ax, line: { color: DATA_COLOR, width: 2 },
                    hovertemplate: 'data<br>λ %{x:.1f}<br>%{y:.2e}<extra></extra>' },
                  { type: 'scatter', mode: 'lines', x, y: r.spectra[f][l], ...ax, line: { color: RUN_COLOR, width: 2 },
                    hovertemplate: 'run<br>λ %{x:.1f}<br>%{y:.2e}<extra></extra>' });
    });
    const ax = sub(layout, L + 1, [L * (w + 0.03), 1], [0, 1],
      { type: 'log', autorange: 'reversed', title: axTitle('wavelength [grid cells]'), showgrid: true, gridcolor: '#eee' },
      { type: 'log', title: axTitle('run / data'), showgrid: true, gridcolor: '#eee' }, 'Ratio run / data');
    const dashes = ['solid', 'dash', 'dot'];
    N.map_depths.forEach((d, l) => traces.push({ type: 'scatter', mode: 'lines', x: N.spectrum_wavelength[l], ...ax,
      y: r.spectra[f][l].map((v, i) => v / N.data.spectra[f][l][i]), line: { color: RUN_COLOR, width: 2, dash: dashes[l] },
      hovertemplate: `${d.toFixed(0)} m<br>λ %{x:.1f}<br>ratio %{y:.2f}<extra></extra>` }));
    traces.push({ type: 'scatter', mode: 'lines', x: [N.spectrum_wavelength[0][0], 2], y: [1, 1], ...ax, line: { color: '#999', width: 1 }, hoverinfo: 'skip' });
    layout.annotations.push({ text: N.map_depths.map((d, l) => `${['—', '– –', '···'][l]} ${d.toFixed(0)} m`).join('  '), x: 1, y: -0.16,
                              xref: 'paper', yref: 'paper', xanchor: 'right', showarrow: false, font: { size: 10, color: '#555' } });
    layout.margin.b = 46;
    plot(div, traces, layout);
  }

  // ---------- curated figures ----------
  // Common x range [°C, g kg⁻¹] of the basin means of figure B in physical units
  const PHYS_RANGE = { toce: [0, 25], soce: [33.5, 36.5] };
  const UNIT = { toce: '°C', soce: 'g kg⁻¹' }, FIELD_NAME = { toce: 'Temperature', soce: 'Salinity' };
  // Symmetric range of the anomaly maps at map level l : 99th percentile of |anomaly| over the stored data states
  const anomalyRange = (f, l) => once(`range-${f}-${l}`, () => {
    const dm = maps(N.data, f), P = NY() * NX(), L = N.map_levels.length, v = [];
    for (let b = 0; b < N.data.n_maps; b++) for (let i = 0; i < P; i++) { const a = dm[(b * L + l) * P + i]; if (!Number.isNaN(a)) v.push(Math.abs(a)); }
    v.sort((a, b) => a - b);
    return v[Math.floor(0.99 * (v.length - 1))];
  });
  const niceRange = R => { const p = 10 ** Math.floor(Math.log10(R)), m = R / p; return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10) * p; };

  // ---------- zonal-mean sections of figure A ----------
  const SEC_ROWS = { toce: [0.17, 0.3], soce: [0, 0.13] };
  const SECTION_CONTOUR = { toce: 2, soce: 0.25 };  // contour interval of the raw sections [°C, g kg⁻¹]
  const zonal = (entry, f) => once(entry.zonal[f], () => {
    const [lo, hi] = N.zonal_range[f];
    return Float32Array.from(new Uint16Array(bytes(entry.zonal[f]).buffer), q => q === 0 ? NaN : lo + (q - 1) / 65534 * (hi - lo));
  });
  // (nz, ny) zonal mean of member m, raw or with the basin mean of each level of that member removed, null on land
  const sectionGrid = (entry, f, m, kind) => {
    const a = zonal(entry, f), nz = N.depth.length, ny = NY();
    return N.depth.map((_, k) => Array.from({ length: ny }, (_, j) => {
      const v = a[(m * nz + k) * ny + j];
      return Number.isNaN(v) ? null : kind === 'raw' ? v : v - entry.offset[f][m][k];
    }));
  };
  // Shared colour range of the sections from the stored data states : [min, max] raw, ±99th percentile of |value| offset removed
  const sectionRange = (f, kind) => once(`secrange-${f}-${kind}`, () => {
    const v = [];
    for (let m = 0; m < N.data.n_maps; m++) sectionGrid(N.data, f, m, kind).forEach(row => row.forEach(x => { if (x !== null) v.push(x); }));
    if (kind === 'raw') return [Math.min(...v), Math.max(...v)];
    const a = v.map(Math.abs).sort((x, y) => x - y), R = niceRange(a[Math.floor(0.99 * (a.length - 1))]);
    return [-R, R];
  });

  function figLayers(div) {
    const r = run(), f = st.layers.field, ny = NY(), nx = NX(), P = ny * nx, L = N.map_levels.length;
    const dm = maps(N.data, f), rm = maps(r, f), md = Math.min(st.layers.data, N.data.n_maps - 1);
    const gen = st.layers.gen.map(g => Math.min(g, r.n_maps - 1));
    const layout = { ...baseLayout(), margin: { l: 44, r: 8, t: 66, b: 48 } }, traces = [];
    const G = 1 / L, gap = 0.04, inner = 0.006, pw = (G - gap - 2 * inner) / 3, rowMap = [0.665, 1], rowCl = [0.385, 0.59];
    const kind = st.layers.clusters, LC = N.level_clusters, K = LC.n, Pl = L * P;
    const lab = e => levelLabels.get(e.level_clusters[kind].labels);
    N.map_levels.forEach((k, l) => {
      const R = niceRange(anomalyRange(f, l)), x0 = l * G + gap / 2, ca = l ? `coloraxis${l + 1}` : 'coloraxis';
      layout[ca] = { colorscale: DIVERGING, cmin: -R, cmax: R,
                     colorbar: { orientation: 'h', x: x0 + (G - gap) / 2, xanchor: 'center', y: rowMap[0] - 0.02, yanchor: 'top', len: (G - gap) * 0.8,
                                 thickness: 9, tickfont: { size: 9 }, nticks: 5, ticksuffix: ` ${UNIT[f]}` } };
      layout.annotations.push({ text: `<b>${N.map_depths[l].toFixed(0)} m</b>`, x: x0 + (G - gap) / 2, y: 1.025, xref: 'paper', yref: 'paper',
                                yanchor: 'bottom', showarrow: false, font: { size: 12 } });
      [['Data', dm, md], [`Generated ${gen[0] + 1}`, rm, gen[0]], [`Generated ${gen[1] + 1}`, rm, gen[1]]].forEach(([title, src, m], c) => {
        const xa = x0 + c * (pw + inner);
        const ax = sub(layout, l * 3 + c + 1, [xa, xa + pw], rowMap, { showticklabels: false, showgrid: false, ticks: '' },
                       { showticklabels: l === 0 && c === 0, showgrid: false, range: [N.lat[0], N.lat[ny - 1]], ticksuffix: '°', nticks: 7 }, title);
        landBackground(layout, ax);
        traces.push({ type: 'heatmap', x: N.lon, y: N.lat, z: toGrid(src, (m * L + l) * P, ny, nx), ...ax, coloraxis: ca, zsmooth: false,
                      hovertemplate: `${title}<br>lat %{y:.1f}°<br>%{z:.3f} ${UNIT[f]}<extra></extra>` });
        // Cluster map of the same member
        const axC = sub(layout, 3 * L + l * 3 + c + 1, [xa, xa + pw], rowCl, { showticklabels: false, showgrid: false, ticks: '' },
                        { showticklabels: l === 0 && c === 0, showgrid: false, range: [N.lat[0], N.lat[ny - 1]], ticksuffix: '°', nticks: 7 });
        landBackground(layout, axC);
        const labels = lab(c ? r : N.data), centre = v => `ΔT ${LC.T[l][v - 1].toFixed(2)} °C · ΔS ${LC.S[l][v - 1].toFixed(2)} · σ₀ ${LC.sigma[l][v - 1].toFixed(2)}`;
        const z = toGrid(Float32Array.from(labels.subarray(m * Pl + l * P, m * Pl + (l + 1) * P), v => v || NaN), 0, ny, nx);
        traces.push({ type: 'heatmap', x: N.lon, y: N.lat, z, ...axC, colorscale: levelColorscale(K), zmin: 0.5, zmax: K + 0.5, zsmooth: false, showscale: false,
                      customdata: z.map(row => row.map(v => v ? centre(v) : '')), hovertemplate: `${title}<br>cluster %{z}<br>%{customdata}<extra></extra>` });
      });
      const pct = e => `${(100 * e.level_clusters[kind].agree[l]).toFixed(0)} %`;
      layout.annotations.push({ text: `<b>clusters, ${kind === 'anom' ? 'offset removed' : 'raw'}</b>`, x: x0 + (G - gap) / 2, y: rowCl[1] + 0.005,
                                xref: 'paper', yref: 'paper', yanchor: 'bottom', showarrow: false, font: { size: 11 } },
                              { text: `agreement : data ${pct(N.data)} · generated ${pct(r)}<br>` +
                                      LC.T[l].map((_, v) => `<span style="color:${LEVEL_COLORS[v]}">■</span> ${v + 1}`).join('  ') + '  (hover : centre)',
                                x: x0 + (G - gap) / 2, y: rowCl[0] - 0.006, xref: 'paper', yref: 'paper', yanchor: 'top', showarrow: false, font: { size: 10, color: '#444' } });
    });
    // Zonal-mean sections of the shown states, T then S
    const sec = st.layers.sections, SW = 0.285, sgap = 0.03;
    layout.annotations.push({ text: `<b>zonal-mean sections, ${sec === 'raw' ? 'raw' : 'offset removed'}</b>`, x: 0.47, y: SEC_ROWS.toce[1] + 0.022,
                              xref: 'paper', yref: 'paper', yanchor: 'bottom', showarrow: false, font: { size: 12 } });
    ['toce', 'soce'].forEach((fs, fi) => {
      const rows = SEC_ROWS[fs], [lo, hi] = sectionRange(fs, sec), ca = `coloraxis${L + 1 + fi}`;
      layout[ca] = { colorscale: sec === 'raw' ? N.colorscales[fs] : DIVERGING, cmin: lo, cmax: hi,
                     colorbar: { x: 3 * (SW + sgap) - sgap + 0.008, xanchor: 'left', y: (rows[0] + rows[1]) / 2, yanchor: 'middle', len: rows[1] - rows[0],
                                 thickness: 9, tickfont: { size: 9 }, nticks: 6, title: { text: UNIT[fs], side: 'right', font: { size: 10 } } } };
      const size = sec === 'raw' ? SECTION_CONTOUR[fs] : hi / 8;
      [[N.data, md, 'Data'], [r, gen[0], `Generated ${gen[0] + 1}`], [r, gen[1], `Generated ${gen[1] + 1}`]].forEach(([e, m, title], c) => {
        const xa = c * (SW + sgap);
        const ax = sub(layout, 6 * L + 1 + fi * 3 + c, [xa, xa + SW], rows, { showticklabels: fi === 1, nticks: 8, ticksuffix: '°', showgrid: false, tickfont: { size: 9 } },
                       { ...depthAxis(c === 0), title: c === 0 ? { text: `<b>${fs === 'toce' ? 'T' : 'S'}</b>`, font: { size: 12 }, standoff: 4 } : undefined }, fi === 0 ? title : '');
        landBackground(layout, ax);
        const z = sectionGrid(e, fs, m, sec);
        traces.push({ type: 'heatmap', x: N.lat, y: Y(), z, ...ax, coloraxis: ca, zsmooth: false, customdata: N.depth.map(d => N.lat.map(() => d)),
                      hovertemplate: `${title}<br>lat %{x:.1f}° · %{customdata:.0f} m<br>%{z:.3f} ${UNIT[fs]}<extra></extra>` },
                    { type: 'contour', x: N.lat, y: Y(), z, ...ax, showscale: false, hoverinfo: 'skip', connectgaps: false,
                      contours: { coloring: 'none', start: Math.ceil(lo / size) * size, end: hi, size }, line: { color: 'rgba(0,0,0,0.35)', width: 0.6 } });
      });
    });
    plot(div, traces, layout);
  }

  const pearson = (a, b) => {
    const ma = meanOf(a), mb = meanOf(b);
    let sab = 0, saa = 0, sbb = 0;
    a.forEach((v, i) => { sab += (v - ma) * (b[i] - mb); saa += (v - ma) ** 2; sbb += (b[i] - mb) ** 2; });
    return sab / Math.sqrt(saa * sbb);
  };
  // Inter-level correlation of the basin mean over the members, (nz, nz), null where a level is constant
  const levelCorrelation = (entry, f) => once(`lcorr-${st.run}-${entry === N.data}-${f}`, () => {
    const cols = N.depth.map((_, k) => column(entry.offset[f], k));
    return cols.map(a => cols.map(b => { const c = pearson(a, b); return Number.isFinite(c) ? c : null; }));
  });

  function figShifts(div) {
    const r = run(), f = st.shifts.field, levels = N.shift_levels, sd = N.spatial_std[f], std = st.shifts.units === 'std';
    const mdata = levelMean(N.data, f), L = levels.length;
    // Physical : absolute basin mean on PHYS_RANGE. σ : (basin mean − data mean) / σ, since an absolute value divided by σ has
    // no meaning of its own. Both share one x range over the levels; points outside are drawn on the edge as outward triangles
    const value = (m, k) => std ? (m - mdata[k]) / sd[k] : m;
    const layout = { ...baseLayout(), showlegend: true, margin: { l: 58, r: 10, t: 26, b: 48 },
                     legend: { orientation: 'h', x: 0, y: 1.0, yanchor: 'bottom', font: { size: 10 } } }, traces = [];
    const entries = [[N.data, DATA_COLOR, -0.2, `data (${N.data.offset[f].length} states)`], [r, RUN_COLOR, 0.2, `generated (${r.offset[f].length} members)`]];
    const all = levels.map(k => entries.flatMap(([e]) => e.offset[f].map(m => value(m[k], k))));
    const shared = Math.max(...all.flat().map(Math.abs)) * 1.05, [x0, x1] = std ? [-shared, shared] : PHYS_RANGE[f];
    const jitter = b => ((b * 0.6180339887) % 1 - 0.5) * 0.24;
    const rowH = 1 / L, gapY = 0.012;
    levels.forEach((k, i) => {
      const y1 = 1 - i * rowH, y0 = y1 - rowH + gapY, last = i === L - 1;
      const ax = sub(layout, 4 + i, [0, 0.5], [y0, y1],
                     { range: [x0, x1], showticklabels: last, nticks: 9, showgrid: true, gridcolor: '#e8e8e8',
                       zeroline: std, zerolinecolor: '#999', title: last ? axTitle(std ? 'basin mean − data mean [spatial σ of the level]' : `basin mean [${UNIT[f]}]`) : undefined },
                     { range: [0.5, -0.5], showticklabels: false, showgrid: false, ticks: '' });
      layout.annotations.push({ text: `${N.depth[k].toFixed(0)} m`, x: -0.005, y: (y0 + y1) / 2, xref: 'paper', yref: 'paper', xanchor: 'right',
                                showarrow: false, font: { size: 10 } });
      entries.forEach(([entry, color, dy, who], e) => traces.push({ type: 'scatter', mode: 'markers', ...ax, name: who, showlegend: i === 0, cliponaxis: false,
        x: entry.offset[f].map(m => Math.min(Math.max(value(m[k], k), x0), x1)), y: entry.offset[f].map((_, b) => dy + jitter(b)),
        customdata: entry.offset[f].map((m, b) => [b + 1, m[k], (m[k] - mdata[k]) / sd[k]]),
        marker: { color, opacity: 0.8, line: { width: 0.5, color: 'white' },
                  symbol: entry.offset[f].map(m => value(m[k], k) < x0 ? 'triangle-left' : value(m[k], k) > x1 ? 'triangle-right' : 'circle'),
                  size: entry.offset[f].map(m => value(m[k], k) < x0 || value(m[k], k) > x1 ? 10 : 6) },
        hovertemplate: `${who.split(' ')[0]} %{customdata[0]}<br>${N.depth[k].toFixed(0)} m<br>basin mean %{customdata[1]:.3f} ${UNIT[f]}<br>` +
                       `%{customdata[2]:+.2f} σ from the data mean<extra></extra>` }));
      // Spread of each group around the data mean, physical and in σ of the level
      const rms = entry => Math.sqrt(meanOf(entry.offset[f].map(m => (m[k] - mdata[k]) ** 2)));
      const clipped = entries.reduce((n, [e]) => n + e.offset[f].filter(m => value(m[k], k) < x0 || value(m[k], k) > x1).length, 0);
      layout.annotations.push({ text: `<span style="color:${RUN_COLOR}">gen ${rms(r).toPrecision(2)} ${UNIT[f]} · ${(rms(r) / sd[k]).toFixed(2)} σ` +
                                      `${clipped ? ` · ${clipped} off axis` : ''}</span><br>` +
                                      `<span style="color:${DATA_COLOR}">data ${rms(N.data).toPrecision(2)} ${UNIT[f]} · ${(rms(N.data) / sd[k]).toFixed(3)} σ</span>`,
                                x: 0.515, y: (y0 + y1) / 2, xref: 'paper', yref: 'paper', xanchor: 'left', align: 'left', showarrow: false, font: { size: 9 } });
    });
    // Inter-level correlation matrix of the generated shifts, and its mean against the level separation
    const C = levelCorrelation(r, f), nz = N.depth.length, ticks = levels;
    const axM = sub(layout, 2, [0.68, 0.86], [0.08, 0.92], { tickvals: ticks, ticktext: ticks.map(k => N.depth[k].toFixed(0)), tickangle: -90, range: [-0.5, nz - 0.5], showgrid: false },
                    { tickvals: ticks, ticktext: ticks.map(k => N.depth[k].toFixed(0)), range: [nz - 0.5, -0.5], showgrid: false }, 'r between levels (generated)');
    traces.push({ type: 'heatmap', z: C, ...axM, colorscale: DIVERGING, zmin: -1, zmax: 1, showlegend: false, zsmooth: false,
                  customdata: C.map((row, i) => row.map((_, j) => [N.depth[i], N.depth[j]])),
                  colorbar: { thickness: 8, len: 0.5, x: 0.865, y: 0.5, tickfont: { size: 9 } },
                  hovertemplate: '%{customdata[0]:.0f} m × %{customdata[1]:.0f} m<br>r %{z:.2f}<extra></extra>' });
    const lags = Array.from({ length: 12 }, (_, i) => i + 1), n = r.offset[f].length;
    const lagMean = (M) => lags.map(l => nanMean(Array.from({ length: nz - l }, (_, k) => M[k][k + l])));
    const axL = sub(layout, 3, [0.915, 1], [0.08, 0.92], { title: axTitle('levels apart'), range: [0.5, 12.5], nticks: 6, showgrid: true, gridcolor: '#eee' },
                    { range: [-0.3, 1], nticks: 6, showgrid: true, gridcolor: '#eee', zeroline: true, zerolinecolor: '#999' }, 'mean r');
    const noise = 1 / Math.sqrt(n - 1);
    traces.push({ type: 'scatter', mode: 'lines', x: [0.5, 12.5, 12.5, 0.5], y: [noise, noise, -noise, -noise], fill: 'toself', ...axL,
                  fillcolor: 'rgba(0,0,0,0.07)', line: { width: 0 }, hoverinfo: 'skip', showlegend: false },
                { type: 'scatter', mode: 'lines+markers', x: lags, y: lagMean(C), ...axL, line: { color: RUN_COLOR, width: 2 }, marker: { size: 5 },
                  showlegend: false, hovertemplate: '%{x} levels apart<br>mean r %{y:.2f}<extra></extra>' });
    layout.annotations.push({ text: `grey : ±1/√(n−1), n = ${n}`, x: 1, y: 0.0, xref: 'paper', yref: 'paper', xanchor: 'right', yanchor: 'top',
                              showarrow: false, font: { size: 9, color: '#777' } });
    plot(div, traces, layout);
  }

  const DRAW = { layers: figLayers, shifts: figShifts, offsets: figOffsets, maps: figMaps, corr: figCorr, recentred: figRecentred, mean: figMean, spectra: figSpectra };
  const HEIGHT = { layers: 1450, shifts: 520, offsets: 330, maps: 360, corr: 360, recentred: 760, mean: 520, spectra: 320 };
  const WIDE = new Set(['layers', 'shifts', 'maps', 'recentred', 'mean']);

  // ---------- controls and cards ----------
  const root = document.getElementById('nc');
  const options = (n, sel = 0, label = i => i + 1) => Array.from({ length: n }, (_, i) => `<option value="${i}"${i === sel ? ' selected' : ''}>${label(i)}</option>`).join('');
  const fieldSelect = id => `<label>Field <select id="${id}"><option value="toce">T</option><option value="soce">S</option></select></label>`;
  // Controls in the head of the curated cards
  const CONTROLS = {
    layers: () => {
      const nr = Math.max(...Object.values(N.runs).map(r => r.n_maps));
      return `${fieldSelect('nc-layers-field')}<label>Data <select id="nc-layers-data">${options(N.data.n_maps)}</select></label>
        <label>Generated <select id="nc-layers-g0">${options(nr, 0)}</select> <select id="nc-layers-g1">${options(nr, 1)}</select></label>
        <label>Clusters <select id="nc-layers-clusters"><option value="anom">offset removed</option><option value="raw">raw</option></select></label>
        <label>Sections <select id="nc-layers-sections"><option value="raw">raw</option><option value="anom">offset removed</option></select></label>`;
    },
    shifts: () => `${fieldSelect('nc-shifts-field')}<label>Units <select id="nc-shifts-units"><option value="phys">physical</option>
        <option value="std">σ of the level, from the data mean</option></select></label>`,
  };
  function build() {
    const runs = Object.keys(N.runs);
    root.innerHTML = `
      <div class="nc-bar">
        <label>Run <select id="nc-run">${runs.map(k => `<option value="${k}">${RUN_LABELS[k] || k} (${N.runs[k].projection})</option>`).join('')}</select></label>
        <label>Field <select id="nc-field"><option value="toce">T</option><option value="soce">S</option></select></label>
        <label>Level <select id="nc-level">${N.map_depths.map((d, l) => `<option value="${l}">${d.toFixed(0)} m</option>`).join('')}</select></label>
        <label>Correlation <select id="nc-metric"><option value="asym">zonally asymmetric</option><option value="full">full field</option></select></label>
        <label class="member">Member <input type="range" id="nc-member" min="0" max="${Math.max(...runs.map(k => N.runs[k].n_maps)) - 1}" value="0">
          <output id="nc-member-out">1</output></label>
      </div>
      <div class="panels nc-figs"><span class="row-label">Show</span>${Object.entries(FIGS).map(([k, [t]]) =>
        `<label><input type="checkbox" value="${k}"${st.shown.has(k) ? ' checked' : ''}> ${t}</label>`).join('')}</div>
      <div class="nc-grid">${Object.entries(FIGS).map(([k, [t, note]]) => `
        <div class="col nc-card${WIDE.has(k) ? ' wide' : ''}" data-fig="${k}"${st.shown.has(k) ? '' : ' hidden'}><div class="head"><h2>${t}</h2>
          ${CONTROLS[k] ? `<div class="nc-ctl">${CONTROLS[k]()}</div>` : ''}</div>
          <div class="nc-note">${note}</div><div class="nc-plot" style="height:${HEIGHT[k]}px"></div></div>`).join('')}</div>`;
    const on = (id, ev, fn) => root.querySelector(id).addEventListener(ev, fn);
    on('#nc-run', 'change', e => { st.run = e.target.value; render(); });
    on('#nc-layers-field', 'change', e => { st.layers.field = e.target.value; render(['layers']); });
    on('#nc-layers-data', 'change', e => { st.layers.data = +e.target.value; render(['layers']); });
    [0, 1].forEach(i => on(`#nc-layers-g${i}`, 'change', e => { st.layers.gen[i] = +e.target.value; render(['layers']); }));
    on('#nc-layers-clusters', 'change', e => { st.layers.clusters = e.target.value; render(['layers']); });
    on('#nc-layers-sections', 'change', e => { st.layers.sections = e.target.value; render(['layers']); });
    on('#nc-shifts-field', 'change', e => { st.shifts.field = e.target.value; render(['shifts']); });
    on('#nc-shifts-units', 'change', e => { st.shifts.units = e.target.value; render(['shifts']); });
    on('#nc-field', 'change', e => { st.field = e.target.value; render(['maps', 'spectra']); });
    on('#nc-level', 'change', e => { st.level = +e.target.value; render(['maps']); });
    on('#nc-metric', 'change', e => { st.metric = e.target.value; render(['maps', 'corr']); });
    on('#nc-member', 'input', e => { st.member = +e.target.value; root.querySelector('#nc-member-out').textContent = st.member + 1; render(['maps']); });
    root.querySelectorAll('.nc-figs input').forEach(i => i.addEventListener('change', () => {
      i.checked ? st.shown.add(i.value) : st.shown.delete(i.value);
      root.querySelector(`.nc-card[data-fig=${i.value}]`).hidden = !i.checked;
      render([i.value]);
    }));
    built = true;
  }
  // Draw the shown figures among `which`, one per animation frame
  let token = 0;
  function render(which = Object.keys(FIGS)) {
    const todo = which.filter(k => st.shown.has(k)), t = ++token;
    const step = i => {
      if (t !== token || i >= todo.length) return;
      DRAW[todo[i]](root.querySelector(`.nc-card[data-fig=${todo[i]}] .nc-plot`));
      requestAnimationFrame(() => step(i + 1));
    };
    step(0);
  }

  window.showNoConstraint = () => {
    const label = document.getElementById('loading');
    if (!N) label.textContent = 'Loading no-constraint data…';
    return load().then(() => { label.textContent = ''; if (!built) { build(); render(); } },
                       e => { label.textContent = e.message; });
  };
})();
