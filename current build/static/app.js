/* FAST client. No framework, no innerHTML with data: every node is built with
   textContent, so a file name or a profile name cannot become markup. */

'use strict';

// ── tiny DOM helpers ────────────────────────────────────────────────────────
function el(tag, attrs, ...children) {
  const node = document.createElement(tag);
  if (attrs) {
    for (const [key, value] of Object.entries(attrs)) {
      if (value === null || value === undefined || value === false) continue;
      if (key === 'class') node.className = value;
      else if (key === 'text') node.textContent = value;
      else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
      else node.setAttribute(key, value === true ? '' : String(value));
    }
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

const ICON = {
  check: () => icon(['M20 6L9 17l-5-5']),
  chevron: () => icon(['M9 6l6 6-6 6']),
  info: () => icon(['M12 8v5', 'M12 16h.01'], [['circle', { cx: 12, cy: 12, r: 9 }]]),
  warn: () => icon(['M12 9v4', 'M12 17h.01'],
    [['path', { d: 'M10.3 3.9 1.9 18a2 2 0 0 0 1.7 3h16.8a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z' }]]),
  empty: () => icon(['M12 7v5l3 2'], [['circle', { cx: 12, cy: 12, r: 9 }]]),
  trash: () => icon(['M4 7h16', 'M10 11v6', 'M14 11v6',
    'M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12',
    'M9 7V5a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v2']),
};

function icon(paths, shapes = []) {
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('width', '16');
  svg.setAttribute('height', '16');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('fill', 'none');
  svg.setAttribute('stroke', 'currentColor');
  svg.setAttribute('stroke-width', '1.75');
  svg.setAttribute('stroke-linecap', 'round');
  svg.setAttribute('stroke-linejoin', 'round');
  svg.setAttribute('aria-hidden', 'true');
  for (const d of paths) {
    const path = document.createElementNS(ns, 'path');
    path.setAttribute('d', d);
    svg.append(path);
  }
  for (const [tag, attrs] of shapes) {
    const node = document.createElementNS(ns, tag);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
    svg.append(node);
  }
  return svg;
}

function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); return node; }

// ── state ───────────────────────────────────────────────────────────────────
const state = {
  screen: 'check',
  stage: 'idle',          // idle | file | progress | results
  upload: null,
  file: null,
  selected: new Set(),
  analysis: null,
  profile: null,         // {id, name}
  history: [], historyAt: 0,
  saved: false,
};

const $ = (sel) => document.querySelector(sel);

// ── api ─────────────────────────────────────────────────────────────────────
async function api(path, options = {}) {
  const response = await fetch(path, { credentials: 'same-origin', ...options });
  let data = null;
  const text = await response.text();
  try { data = text ? JSON.parse(text) : null; } catch (e) { data = { error: 'Unexpected reply from the server' }; }
  if (!response.ok) {
    const message = (data && data.error) || `Request failed (${response.status})`;
    const error = new Error(message);
    error.status = response.status;
    if (response.status === 401 && state.profile) {
      // The token expired. Drop what belongs to the profile, but keep the
      // upload and the result on screen: they are not the profile's, and
      // throwing them away would make the athlete start over.
      state.profile = null;
      state.history = []; state.historyAt = 0;
      clear($('#home-last'));
      if (state.screen === 'profile') renderProfile();
      if (state.screen === 'train') renderTrain();
      if (state.screen === 'home') renderHome();
    }
    throw error;
  }
  return data;
}

function toast(message) {
  const node = $('#toast');
  node.textContent = message;
  node.classList.add('show');
  clearTimeout(toast._t);
  toast._t = setTimeout(() => node.classList.remove('show'), 3200);
}

// ── theme ───────────────────────────────────────────────────────────────────
function currentTheme() {
  const explicit = document.documentElement.dataset.theme;
  if (explicit) return explicit;
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function toggleTheme() {
  const next = currentTheme() === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem('fast_theme', next); } catch (e) { /* ignore */ }
  if (state.stage === 'results') renderResults();
  if (state.screen === 'home') renderHome();
}

// ── navigation ──────────────────────────────────────────────────────────────
function nav(screen) {
  state.screen = screen;
  // Opening Home after a pause replays the sweep: an entrance that only ever
  // fires on the first paint is an entrance nobody sees.
  if (screen === 'home' && Date.now() - dialLastPlay > 20_000) dialNonce += 1;
  for (const name of ['home', 'check', 'train', 'history', 'profile']) {
    $('#screen-' + name).classList.toggle('hidden', name !== screen);
  }
  document.querySelectorAll('.tab').forEach((tab) => {
    tab.setAttribute('aria-selected', String(tab.dataset.nav === screen));
  });
  window.scrollTo({ top: 0 });
  if (screen === 'home') renderHome();
  if (screen === 'train') renderTrain();
  if (screen === 'history') loadHistory();
  if (screen === 'profile') renderProfile();
}

function setStage(stage) {
  state.stage = stage;
  for (const name of ['idle', 'file', 'progress', 'results']) {
    $('#stage-' + name).classList.toggle('hidden', name !== stage);
  }
  if (stage !== 'idle') nav('check');
}

// ── upload ──────────────────────────────────────────────────────────────────
async function uploadFile(file, rate) {
  state.file = file;
  const form = new FormData();
  form.append('file', file);
  if (rate) form.append('fs', String(rate));
  setStage('progress');
  $('#progress-text').textContent = 'Reading the recording…';
  $('#progress-sub').textContent = file.name;
  try {
    const data = await api('/upload', { method: 'POST', body: form });
    state.upload = data;
    state.selected = new Set(data.muscles.filter((m) => m.recognised).map((m) => m.column));
    setStage('file');
    renderFile();
  } catch (error) {
    setStage('idle');
    toast(error.message);
  }
}

function renderFile() {
  const up = state.upload;
  const meta = clear($('#file-meta'));
  const rows = [
    ['Recording', up.filename],
    ['Length', up.duration ? formatDuration(up.duration) : 'unknown'],
    ['Analysed', up.analysed_seconds ? 'first ' + formatDuration(up.analysed_seconds) : '—'],
    ['Channels', String(up.cols.length)],
  ];
  for (const [key, value] of rows) {
    meta.append(el('div', null, el('dt', { text: key }), el('dd', { text: value })));
  }

  const notices = clear($('#file-notices'));
  if (up.truncated) {
    notices.append(el('div', { class: 'notice notice--warn' }, ICON.warn(),
      el('span', { text: `This recording is ${formatDuration(up.duration)} long. FAST Athlete analyses the first 30 seconds.` })));
  }
  for (const dropped of up.dropped || []) {
    notices.append(el('div', { class: 'notice notice--warn' }, ICON.warn(),
      el('span', { text: `${dropped.name} was skipped: ${dropped.reason}.` })));
  }
  for (const note of up.notes || []) {
    notices.append(el('div', { class: 'notice' }, ICON.info(), el('span', { text: note })));
  }

  const rateInput = $('#rate-input');
  const known = up.fs && up.fs_source !== 'none';
  rateInput.value = known ? Math.round(up.fs) : '';
  $('#rate-explain').textContent = known
    ? (up.fs_source === 'provided'
        ? 'Set by you.'
        : up.fs_source === 'time-column'
          ? 'Read from the time column in the file.'
          : 'Read from the file header.')
    : 'This file does not say. FAST Athlete cannot measure frequencies without it.';
  $('#rate-card').classList.toggle('notice--warn', !known);

  renderMuscleChoices();
}

function renderMuscleChoices() {
  const up = state.upload;
  const grid = clear($('#muscle-choices'));
  const recognised = up.muscles.filter((m) => m.recognised);
  const others = up.muscles.filter((m) => !m.recognised);

  for (const muscle of recognised) {
    grid.append(choiceFor(muscle, muscle.id, muscle.name, muscle.desc));
  }
  const otherBox = clear($('#other-channels'));
  if (others.length) {
    otherBox.append(el('h3', { class: 'h-group', text: 'Other channels' }),
      el('p', { class: 'meta', text: 'Channels FAST Athlete did not recognise. Check them only if you know they are EMG.' }));
    const list = el('div', { class: 'choices' });
    for (const channel of others) {
      list.append(choiceFor(channel, channel.column, channel.column, 'Unrecognised channel'));
    }
    otherBox.append(list);
  }
  updateSelection();
}

function choiceFor(muscle, value, title, subtitle) {
  const input = el('input', {
    type: 'checkbox', value,
    onchange: (event) => {
      if (event.target.checked) state.selected.add(value); else state.selected.delete(value);
      updateSelection();
    },
  });
  if (state.selected.has(value)) input.checked = true;
  return el('label', { class: 'choice' }, input, el('span', { class: 'box' }, ICON.check()),
    el('span', { class: 'label' }, el('strong', { text: title }), el('span', { text: subtitle })));
}

function updateSelection() {
  const total = state.upload ? state.upload.muscles.length : 0;
  const n = state.selected.size;
  $('#selection-count').textContent = n
    ? `${n} of ${total} muscles chosen`
    : 'Choose at least one muscle';
  const button = $('#analyse-btn');
  button.disabled = n === 0;
  button.textContent = n ? `Check ${n} muscle${n > 1 ? 's' : ''}` : 'Check fatigue';
}

// ── analysis ────────────────────────────────────────────────────────────────
async function analyse() {
  const up = state.upload;
  if (!up || !state.selected.size) return;
  const muscles = up.muscles
    .filter((m) => state.selected.has(m.column))
    .map((m) => ({ id: m.id, column: m.column }));

  setStage('progress');
  $('#progress-text').textContent = 'Measuring…';
  $('#progress-sub').textContent = `${muscles.length} muscle${muscles.length > 1 ? 's' : ''}`;
  const started = Date.now();
  const ticker = setInterval(() => {
    const seconds = Math.round((Date.now() - started) / 1000);
    $('#progress-sub').textContent = `${muscles.length} muscle${muscles.length > 1 ? 's' : ''} · ${seconds} s`;
  }, 500);

  try {
    const data = await api('/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session: up.session, muscles }),
    });
    if (state.upload && Array.isArray(state.upload.muscles)) {
      for (const r of Object.values(data.results || {})) {
        const match = state.upload.muscles
          .find((m) => m.column === r.column && m.recognised);
        if (match) r.name = match.name;
      }
    }
    state.analysis = { ...data, at: Date.now(), sample: !!state.samplePending };
    state.samplePending = false;
    state.checkedAt = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    state.saved = false;
    setStage('results');
    renderResults();
    saveIfPossible();
  } catch (error) {
    setStage('file');
    if (error.status === 404) {
      toast('The upload expired. Choose the file again.');
      setStage('idle');
    } else {
      toast(error.message);
    }
  } finally {
    clearInterval(ticker);
  }
}

// ── results ─────────────────────────────────────────────────────────────────
const RESULTS_ORDER = { red: 0, amber: 1, green: 2, grey: 3 };
const WORDS = { green: 'Low', amber: 'Medium', red: 'High', grey: 'None' };

function statusRank(status) { return status in RESULTS_ORDER ? RESULTS_ORDER[status] : 4; }

function renderResults() {
  const box = clear($('#stage-results'));
  const data = state.analysis;
  if (!data) return;
  const results = Object.values(data.results).sort((a, b) => statusRank(a.status) - statusRank(b.status));
  const overall = data.overall;
  const up = state.upload;
  const byColumn = new Map((up ? up.muscles : []).map((m) => [m.column, m]));

  // verdict
  const worst = results.find((r) => r.status === 'red') || results.find((r) => r.status === 'amber');
  const counts = { green: 0, amber: 0, red: 0, grey: 0 };
  for (const r of results) counts[r.status] = (counts[r.status] || 0) + 1;
  const nameOf = (r) => (byColumn.get(r.column) || {}).name || r.id;

  const sentence = (names) => names.endsWith('.') ? names : names + '.';
  const title = el('h1');
  if (counts.red) {
    title.append(el('span', { class: 'status--red', text: 'High fatigue' }));
    title.append(': ' + sentence(listNames(results.filter((r) => r.status === 'red').map(nameOf))));
  } else if (counts.amber) {
    title.append(el('span', { class: 'status--amber', text: 'Medium fatigue' }));
    title.append(' in ' + sentence(listNames(results.filter((r) => r.status === 'amber').map(nameOf))));
  } else if (counts.green) {
    title.append(el('span', { class: 'status--green', text: 'Low fatigue' }));
    title.append(counts.green > 1 ? ` — all ${counts.green} muscles.` : ' — ' + sentence(listNames(results.filter(r => r.status === 'green').map(nameOf))));
  } else {
    title.append(el('span', { class: 'status--grey', text: 'Nothing measurable' }));
  }

  const lede = el('p', { class: 'lede', text: advice(counts, overall) });
  const countsBox = el('div', { class: 'counts' });
  for (const key of ['green', 'amber', 'red']) {
    if (!counts[key]) continue;
    countsBox.append(el('span', { class: 'count' },
      el('span', { class: `dot dot--${key}` }),
      el('strong', { text: String(counts[key]) }),
      { green: 'low', amber: 'medium', red: 'high' }[key]));
  }
  box.append(el('div', { class: 'verdict' }, title, lede, countsBox));
  box.append(planeCard(results, 'Where each muscle moved'));

  // rows
  const rows = el('div', { class: 'rows' });
  for (const r of results) {
    const meta = byColumn.get(r.column) || {};
    rows.append(el('div', { class: 'row' },
      el('span', { class: `dot dot--${r.status}` }),
      el('div', { class: 'name' },
        el('strong', { text: meta.id && meta.recognised ? `${meta.id} · ${meta.name}` : r.id }),
        // The two numbers the method reads, with their labels.
        el('span', { text: r.delta_mdf === undefined ? (r.tip || '')
          : `MDF ${signed(r.delta_mdf)}% · RMS ${signed(r.delta_rms)}%` })),
      el('span', { class: `score score--${r.status}`, text: r.status === 'grey' ? '—' : String(r.score) }),
      el('span', { class: 'word', text: WORDS[r.status] || '' })));
  }
  box.append(el('div', { class: 'card' }, el('h2', { text: 'Muscle by muscle' }), rows));

  // map
  const map = buildMap(results, byColumn);
  if (map) box.append(map);

  // how it was measured
  const engineLabel = data.engine || '';
  box.append(el('details', { class: 'disclose' },
    el('summary', null, ICON.chevron(), 'How this was measured'),
    el('dl', null,
      el('dt', { text: 'Signal' }), el('dd', { text: 'Frequency content of each muscle\'s electrical activity, start of the contraction compared with the whole recording.' }),
      el('dt', { text: 'Reads' }), el('dd', { text: 'median frequency and RMS, change from the start of the recording' }),
      el('dt', { text: 'Baseline' }), el('dd', { text: 'First five seconds of the same recording' }),
      el('dt', { text: 'Scale' }), el('dd', { text: '70–100 recovered · 40–69 some fatigue · under 40 needs rest' }),
      el('dt', { text: 'Window' }), el('dd', { text: up && up.analysed_seconds ? `First ${formatDuration(up.analysed_seconds)} of the recording` : 'Up to 30 seconds' }),
      el('dt', { text: 'Engine' }), el('dd', { text: engineLabel }))));

  const notice = el('div', { class: 'notice' }, ICON.info(),
    el('span', { text: 'Research estimate, not a diagnosis. The thresholds are provisional and still under validation.' }));
  box.append(notice);

  const actions = el('div', { class: 'btn-row mt-16' });
  actions.append(el('button', { class: 'btn btn--quiet', type: 'button', onclick: startOver }, 'New check'));
  if (state.profile) {
    actions.append(el('button', { class: 'btn btn--quiet', type: 'button', onclick: () => nav('history') }, 'History'));
  }
  box.append(actions);

  if (!state.profile) {
    box.append(el('div', { class: 'notice mt-16' }, ICON.info(),
      el('span', null,
        'This result is not saved. ',
        el('button', { class: 'btn--text', type: 'button', onclick: () => nav('profile') }, 'Add a profile'),
        ' to keep your history.')));
  }
}

function listNames(names) {
  const unique = [...new Set(names)];
  if (unique.length <= 1) return unique[0] || '';
  return unique.slice(0, -1).join(', ') + ' and ' + unique[unique.length - 1];
}

function advice(counts, overall) {
  if (counts.grey && !counts.green && !counts.amber && !counts.red) {
    return 'Nothing in this recording could be measured. Check the sampling rate and that the channels hold EMG.';
  }
  if (counts.red) return 'Give those muscles a day. Everything else can train as planned.';
  if (counts.amber) return 'Keep the volume down today and stay off heavy sets for the muscles listed.';
  return 'Train as planned.';
}

function startOver() {
  state.upload = null; state.file = null; state.analysis = null;
  state.selected = new Set();
  $('#file-input').value = '';
  setStage('idle');
  renderHomeLast();
}

// ── body map ────────────────────────────────────────────────────────────────
const MM_REGION = {
  VL: ['outer-quad', 'front'], VM: ['inner-quad', 'front'], RF: ['quadriceps', 'front'],
  TA: ['tibialis', 'front'], MD: ['deltoids', 'front'], BB: ['biceps', 'front'],
  BR: ['forearm', 'front'], FCR: ['forearm', 'front'], ECR: ['forearm', 'front'],
  FDS: ['forearm', 'front'], FDI: ['hands', 'front'],
  BF: ['hamstring', 'back'], ST: ['hamstring', 'back'], GM: ['calves', 'back'],
  GL: ['calves', 'back'],
};
const MM_LABEL = {
  'outer-quad': 'Vastus lateralis', 'inner-quad': 'Vastus medialis',
  quadriceps: 'Rectus femoris', tibialis: 'Tibialis anterior', hamstring: 'Hamstrings',
  calves: 'Calves', deltoids: 'Deltoid', biceps: 'Biceps', forearm: 'Forearm', hands: 'Hand',
};
let mapWidget = null;

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/* Status names and colour tokens are not the same words on purpose: the status
   is a verdict, the token is a colour. */
const STATUS_TOKEN = { green: '--ok', amber: '--warn', red: '--bad', grey: '--grey' };
const VERDICT_CLASS = STATUS_TOKEN;

/* The canvas only accepts rgb()/rgba() reliably, so tokens are converted. */
function tokenRgb(name) {
  const hex = cssVar(name).replace('#', '');
  if (hex.length !== 6) return 'rgb(128, 128, 128)';
  const value = parseInt(hex, 16);
  return `rgb(${(value >> 16) & 255}, ${(value >> 8) & 255}, ${value & 255})`;
}

function buildMap(results, byColumn) {
  if (typeof MuscleMapJS === 'undefined') return null;
  const regions = new Map();
  for (const r of results) {
    const meta = byColumn.get(r.column) || {};
    const key = meta.recognised ? r.id : null;
    const region = key ? MM_REGION[key] : null;
    if (!region) continue;
    const [slug, side] = region;
    const existing = regions.get(slug);
    // the worst reading wins, which is what the athlete needs to see
    if (!existing || statusRank(r.status) < statusRank(existing.status)) {
      regions.set(slug, { status: r.status, side });
    }
  }
  if (!regions.size) return null;

  const wrap = el('div', { class: 'card mapcard' });
  wrap.append(el('h2', { text: 'On the body' }));
  const sides = new Set([...regions.values()].map((v) => v.side));
  let side = sides.has('front') ? 'front' : 'back';

  const canvas = el('div', { class: 'mapcanvas' });
  const legend = el('div', { class: 'maplegend' });
  for (const key of ['green', 'amber', 'red']) {
    if (![...regions.values()].some((v) => v.status === key)) continue;
    legend.append(el('span', null, el('span', { class: `dot dot--${key}` }), WORDS[key]));
  }

  if (sides.size > 1) {
    const group = el('div', { class: 'segmented' });
    for (const name of ['front', 'back']) {
      group.append(el('button', {
        type: 'button', 'aria-pressed': String(name === side), text: name === 'front' ? 'Front' : 'Back',
        onclick: () => { side = name; paint(); group.querySelectorAll('button').forEach((b) => b.setAttribute('aria-pressed', String(b.textContent.toLowerCase() === name))); },
      }));
    }
    wrap.append(group);
  }
  wrap.append(canvas, legend);
  requestAnimationFrame(() => { paint(); });

  function paint() {
    if (mapWidget && mapWidget.destroy) { try { mapWidget.destroy(); } catch (e) { /* ignore */ } }
    mapWidget = null;
    clear(canvas);
    mapWidget = new MuscleMapJS.MuscleMapWidget(canvas, {
      gender: 'male', side, interactive: false, multiSelect: false, showSubGroups: true,
    });
    mapWidget.setStyle({
      defaultFillColor: tokenRgb('--line'),
      strokeColor: cssVar('--line-strong'),
      strokeWidth: 1,
      selectionColor: tokenRgb('--accent'),
      selectionStrokeColor: tokenRgb('--accent'),
      selectionStrokeWidth: 2,
      headColor: tokenRgb('--line'),
      hairColor: tokenRgb('--line'),
      shadowColor: 'transparent', shadowRadius: 0,
    });
    if (mapWidget.enableTooltip) {
      mapWidget.enableTooltip((muscle) => MM_LABEL[muscle] || muscle);
    }
    for (const [slug, info] of regions) {
      if (info.side !== side) continue;
      mapWidget.highlightSide(slug, 'right', tokenRgb(STATUS_TOKEN[info.status] || '--grey'), 1);
    }
  }
  return wrap;
}

// ── saving ──────────────────────────────────────────────────────────────────
async function saveIfPossible() {
  if (!state.profile || !state.analysis || state.saved) return;
  if (state.analysis.sample) return;
  const results = Object.values(state.analysis.results).map((r) => ({
    id: r.id, column: r.column, status: r.status, score: r.score,
    nfi: r.nfi, tip: r.tip, level: r.level,
    delta_mdf: r.delta_mdf, delta_rms: r.delta_rms,
  }));
  try {
    await api('/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        profile_id: state.profile.id,
        filename: state.upload ? state.upload.filename : '',
        results,
      }),
    });
    state.saved = true;
    state.history.unshift({
      id: `local-${Date.now()}`,
      ts: new Date().toISOString(),
      status: state.analysis.results ? (Object.values(state.analysis.results)[0] || {}).status : null,
      results,
    });
    state.historyAt = Date.now();
    toast(`Saved to ${state.profile.name}`);
  } catch (error) {
    if (error.status === 401) {
      state.profile = null;
      toast('Signed out. Sign in to save results.');
    } else {
      toast(error.message);
    }
  }
}



// ── the strip ───────────────────────────────────────────────────────────────
// A time-frequency plane of one real recording, coloured through a single-hue
// ramp, with the median-frequency trace and its threshold drawn over it. The
// picture is the measurement: brightness is amplitude, the ridge is frequency,
// and the two together are what the method reads.
const STRIP_RAMP = [
  [0.00, [22, 22, 24]],     // #161618, the card ground
  [0.20, [14, 42, 48]],
  [0.45, [11, 92, 104]],
  [0.70, [18, 169, 184]],
  [0.88, [22, 224, 240]],   // the identity cyan
  [1.00, [217, 252, 255]],
];

function rampColour(t) {
  const v = Math.max(0, Math.min(1, t));
  for (let i = 1; i < STRIP_RAMP.length; i += 1) {
    const [pos, rgb] = STRIP_RAMP[i];
    if (v <= pos) {
      const [prevPos, prevRgb] = STRIP_RAMP[i - 1];
      const k = (v - prevPos) / (pos - prevPos || 1);
      return prevRgb.map((c, n) => Math.round(c + (rgb[n] - c) * k));
    }
  }
  return STRIP_RAMP[STRIP_RAMP.length - 1][1];
}

let stripData = null;
async function loadStrip() {
  if (stripData) return stripData;
  // Revalidates with the ETag: cheap, and a regenerated strip is picked up.
  const response = await fetch('/static/demo-strip.json', { cache: 'no-cache' });
  if (!response.ok) throw new Error('strip unavailable');
  stripData = await response.json();
  stripData.bytes = Uint8Array.from(atob(stripData.grid), (c) => c.charCodeAt(0));
  return stripData;
}

function drawStrip(canvas, data) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const cssW = canvas.parentElement.clientWidth || 900;
  const cssH = Math.round(Math.min(380, Math.max(240, cssW * 0.30)));
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  canvas.style.width = '100%';
  canvas.style.height = `${cssH}px`;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);

  const css = getComputedStyle(document.documentElement);
  const col = (name) => css.getPropertyValue(name).trim() || '#888';
  const mono = '11px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
  const sans = '11px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';

  const pad = { l: cssW < 520 ? 8 : 44, r: 12, t: 10, b: 22 };
  const plot = { x: pad.l, y: pad.t, w: cssW - pad.l - pad.r, h: cssH - pad.t - pad.b };
  const { cols, rows, f0, f1 } = data;

  // The plane, drawn small and scaled up with smoothing.
  const off = document.createElement('canvas');
  off.width = cols; off.height = rows;
  const offCtx = off.getContext('2d');
  const image = offCtx.createImageData(cols, rows);
  for (let i = 0; i < cols * rows; i += 1) {
    const v = data.bytes[i] / 255;
    const [r, g, b] = rampColour(v);
    const o = i * 4;
    image.data[o] = r; image.data[o + 1] = g; image.data[o + 2] = b; image.data[o + 3] = 255;
  }
  offCtx.putImageData(image, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  // Row 0 is the lowest frequency; the canvas draws top down.
  for (let i = 0; i < rows; i += 1) {
    const srcRow = rows - 1 - i;
    const y = plot.y + (i / rows) * plot.h;
    const h = plot.h / rows + 0.5;
    ctx.drawImage(off, 0, srcRow, cols, 1, plot.x, y, plot.w, h);
  }

  const frac = (f) => 1 - (f - f0) / (f1 - f0);
  const xAt = (t) => plot.x + (t / data.seconds) * plot.w;

  // Baseline window.
  const baseW = xAt(data.baseline_s) - plot.x;
  ctx.fillStyle = 'rgba(255,255,255,.05)';
  ctx.fillRect(plot.x, plot.y, baseW, plot.h);
  ctx.fillStyle = col('--accent');
  ctx.globalAlpha = 0.55;
  ctx.fillRect(plot.x, plot.y, baseW, 2);
  ctx.globalAlpha = 1;

  // Frequency gridlines.
  ctx.strokeStyle = 'rgba(255,255,255,.06)';
  ctx.lineWidth = 1;
  for (const hz of [10, 20, 30]) {
    const gy = plot.y + frac(hz) * plot.h;
    ctx.beginPath(); ctx.moveTo(plot.x, gy); ctx.lineTo(plot.x + plot.w, gy); ctx.stroke();
  }

  // The median-frequency trace and its threshold.
  const valid = data.mdf.filter((v) => Number.isFinite(v));
  if (valid.length > 4) {
    const base = valid[0];
    const limit = base * 0.95;
    const limitY = plot.y + frac(limit) * plot.h;
    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = col('--warn');
    ctx.globalAlpha = 0.75;
    ctx.beginPath();
    ctx.moveTo(xAt(data.baseline_s), limitY);
    ctx.lineTo(plot.x + plot.w, limitY);
    ctx.stroke();
    ctx.restore();

    ctx.lineWidth = 3.5;
    ctx.strokeStyle = 'rgba(13,13,15,.65)';
    ctx.beginPath();
    data.mdf.forEach((f, i) => {
      if (!Number.isFinite(f)) return;
      const x = xAt((i + 0.5) * (data.seconds / cols));
      const y = plot.y + frac(f) * plot.h;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.lineWidth = 1.75;
    ctx.strokeStyle = '#F2F4F5';
    ctx.beginPath();
    data.mdf.forEach((f, i) => {
      if (!Number.isFinite(f)) return;
      const x = xAt((i + 0.5) * (data.seconds / cols));
      const y = plot.y + frac(f) * plot.h;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Where the trace first drops under the threshold.
    const cross = data.mdf.findIndex((f) => Number.isFinite(f) && f < limit);
    if (cross > 0) {
      const cx = xAt((cross + 0.5) * (data.seconds / cols));
      const cy = plot.y + frac(data.mdf[cross]) * plot.h;
      ctx.fillStyle = col('--warn');
      ctx.beginPath(); ctx.arc(cx, cy, 4, 0, Math.PI * 2); ctx.fill();
      if (cssW >= 620) {
        ctx.font = '11px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
        ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
        ctx.fillText(`flagged · ${Math.round((cross + 0.5) * (data.seconds / cols))} s`,
                     cx + 9, cy - 3);
      }
    }

    // Where the trace ends: the number the picture is arguing for.
    const last = valid[valid.length - 1];
    const change = (last - valid[0]) / valid[0] * 100;
    if (cssW >= 620) {
      ctx.font = '600 12px ui-monospace, SFMono-Regular, Menlo, monospace';
      ctx.fillStyle = '#F2F4F5';
      ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
      ctx.fillText(`MDF ${change >= 0 ? '+' : ''}${change.toFixed(0)} %`,
                   plot.x + plot.w - 4, plot.y + 2);
    }
  }

  // Ticks.
  ctx.fillStyle = col('--text-3');
  ctx.font = mono;
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  if (cssW >= 520) {
    for (const hz of [10, 20, 30]) {
      ctx.fillText(`${hz}`, plot.x - 8, plot.y + frac(hz) * plot.h);
    }
    ctx.textBaseline = 'bottom';
    ctx.fillText('Hz', plot.x - 8, plot.y + 2);
  }
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  for (const s of [0, 10, 20]) {
    ctx.fillText(`${s}`, xAt(s), plot.y + plot.h + 6);
  }
  ctx.textAlign = 'right';
  ctx.fillText(`${Math.round(data.seconds)} s`, plot.x + plot.w, plot.y + plot.h + 6);

  // The two labels that explain the picture.
  ctx.font = sans;
  ctx.fillStyle = col('--text-3');
  ctx.textAlign = 'left'; ctx.textBaseline = 'top';
  ctx.fillStyle = 'rgba(242,244,245,.55)';
  ctx.font = '10px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
  ctx.fillText('BASELINE', plot.x + 6, plot.y + 6);
}

async function renderStrip() {
  const canvas = $('#home-strip');
  if (!canvas) return;
  try {
    const data = await loadStrip();
    drawStrip(canvas, data);
    const caption = $('#strip-caption');
    if (caption) {
      caption.textContent = `Example recording · ${data.label || data.channel} · ${Math.round(data.seconds)} s hold · 1–35 Hz · brighter = more power`;
    }
    window.__strip = data;
  } catch (e) {
    canvas.replaceWith(el('p', { class: 'meta', text: 'The example strip could not be loaded.' }));
  }
}

// ── the fatigue plane ───────────────────────────────────────────────────────
// Horizontal: change in amplitude. Vertical: change in median frequency, with
// lower values further down. The two thresholds the method uses divide the
// plane, and every muscle is a point that moved from the origin.
const PLANE = { x0: -25, x1: 65, y0: 15, y1: -65, w: 960, h: 440 };

function drawPlane(canvas, results) {
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const { x0, x1, y0, y1, w: W, h: H } = PLANE;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  canvas.style.width = '100%';
  canvas.style.height = 'auto';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);

  const css = getComputedStyle(document.documentElement);
  const col = (name) => css.getPropertyValue(name).trim() || '#888';
  const ink = col('--text-3');
  const line = col('--line');
  const ok = col('--ok'), warn = col('--warn'), bad = col('--bad');

  const pad = { l: 66, r: 22, t: 16, b: 46 };
  const X = (v) => pad.l + (v - x0) / (x1 - x0) * (W - pad.l - pad.r);
  const Y = (v) => pad.t + (v - y0) / (y1 - y0) * (H - pad.t - pad.b);
  const plot = { x: pad.l, y: pad.t, w: W - pad.l - pad.r, h: H - pad.t - pad.b };

  // Regions: the quiet corner and the corner that means fatigue.
  ctx.fillStyle = ok; ctx.globalAlpha = 0.06;
  ctx.fillRect(X(x0), Y(y0), X(10) - X(x0), Y(-5) - Y(y0));
  ctx.fillStyle = bad; ctx.globalAlpha = 0.07;
  ctx.fillRect(X(10), Y(-5), X(x1) - X(10), Y(y1) - Y(-5));
  ctx.globalAlpha = 1;

  // Grid and ticks
  ctx.strokeStyle = line; ctx.lineWidth = 1;
  ctx.fillStyle = ink;
  ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  for (const v of [-20, 0, 20, 40, 60]) {
    ctx.beginPath(); ctx.moveTo(X(v), plot.y); ctx.lineTo(X(v), plot.y + plot.h); ctx.stroke();
    ctx.fillText(`${v}`, X(v), plot.y + plot.h + 8);
  }
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  for (const v of [10, 0, -20, -40, -60]) {
    ctx.beginPath(); ctx.moveTo(plot.x, Y(v)); ctx.lineTo(plot.x + plot.w, Y(v)); ctx.stroke();
    ctx.fillText(`${v}`, plot.x - 8, Y(v));
  }

  // Thresholds
  ctx.save();
  ctx.setLineDash([5, 5]);
  ctx.strokeStyle = col('--line-strong');
  ctx.beginPath(); ctx.moveTo(X(10), plot.y); ctx.lineTo(X(10), plot.y + plot.h); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(plot.x, Y(-5)); ctx.lineTo(plot.x + plot.w, Y(-5)); ctx.stroke();
  ctx.restore();

  // Axis titles, horizontal: rotated text at this size reads badly.
  ctx.fillStyle = ink;
  ctx.font = '11px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
  ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
  ctx.fillText('amplitude change (%)', plot.x + plot.w / 2, H - 6);
  ctx.textAlign = 'right'; ctx.textBaseline = 'top';
  ctx.fillStyle = col('--text-3');
  ctx.fillText('median frequency change (%)', plot.x + plot.w - 4, plot.y + 2);

  // Quiet corner labels
  ctx.textAlign = 'left'; ctx.textBaseline = 'top';
  ctx.fillStyle = ok; ctx.globalAlpha = .8;
  ctx.fillText('low', X(x0) + 10, Y(y0) + 8);
  ctx.fillStyle = bad;
  ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
  ctx.fillText('high', X(x1) - 10, Y(y1) - 8);
  ctx.globalAlpha = 1;

  // Muscles
  const points = (results || []).filter((r) => Number.isFinite(r.delta_mdf) && Number.isFinite(r.delta_rms));
  if (!points.length) {
    ctx.fillStyle = ink;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.font = '13px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
    ctx.fillText('Your muscles appear here after a check', plot.x + plot.w / 2, plot.y + plot.h / 2);
    return;
  }
  ctx.font = '600 12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
  const placed = [];
  const quiet = points.filter((r) => r.status === 'green' || r.status === 'grey');
  for (const r of points) {
    const px = Math.max(X(x0), Math.min(X(x1), X(r.delta_rms)));
    const py = Math.max(Y(y0), Math.min(Y(y1), Y(r.delta_mdf)));
    const colour = r.status === 'red' ? bad : r.status === 'amber' ? warn : r.status === 'green' ? ok : ink;
    ctx.strokeStyle = colour; ctx.globalAlpha = .45; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(X(0), Y(0)); ctx.lineTo(px, py); ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.fillStyle = colour;
    ctx.beginPath(); ctx.arc(px, py, 5.5, 0, Math.PI * 2); ctx.fill();
    if (r.status === 'green' || r.status === 'grey') continue;   // labelled as a group
    // Nudge the label down until it sits clear of the ones already drawn.
    let ly = py;
    while (placed.some((p) => Math.abs(p.x - px) < 64 && Math.abs(p.y - ly) < 15)) ly += 15;
    placed.push({ x: px, y: ly });
    ctx.fillStyle = col('--text');
    ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    ctx.fillText(r.id, px + 9, ly);
  }
  if (quiet.length) {
    ctx.fillStyle = ok;
    ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    ctx.fillText(`${quiet.length} at low`, X(0) + 10, Y(0) - 14);
  }
}

function planeCard(results, title) {
  const canvas = el('canvas', { class: 'plane', id: 'plane-canvas' });
  const card = el('div', { class: 'card plane-card mt-16' },
    el('div', { class: 'plane-head' },
      el('h3', { text: title || 'The fatigue plane' }),
      el('div', { class: 'legend' },
        el('span', null, el('i', { style: 'background:var(--ok)' }), 'low'),
        el('span', null, el('i', { style: 'background:var(--warn)' }), 'medium'),
        el('span', null, el('i', { style: 'background:var(--bad)' }), 'high'))),
    el('div', { class: 'plane-wrap' }, canvas));
  requestAnimationFrame(() => drawPlane(canvas, results));
  return card;
}

// ── home: the dial ─────────────────────────────────────────────────────────
// The traffic light, bent into a circle: three named zones, and a marker where
// the reading landed. Zone edges follow the engine's own cut-offs -- 70 ready,
// 40 rest -- so the colour on screen and the number agree.
const DIAL = { size: 1200, radius: 552 };
const ZONE_EDGES = [0.00, 0.30, 0.60, 1.00];
const ZONE_TOKENS = ['--ok', '--warn', '--bad'];
const ZONE_WORDS = ['READY', 'EASY', 'REST'];
const ZONE_INDEX = { green: 0, amber: 1, red: 2 };
const VERDICT_WORD = { green: 'ready', amber: 'go easy', red: 'rest' };
const DIAL_WORD = { green: 'READY', amber: 'EASY', red: 'REST' };
const RANK = { red: 3, amber: 2, green: 1, grey: 0 };
const FRESH_H = 36;          // hours after which a saved check stops answering for today

const zoneToken = (s) => (s in ZONE_INDEX ? ZONE_TOKENS[ZONE_INDEX[s]] : '--grey');
const zoneWord = (s) => (s in ZONE_INDEX ? ZONE_WORDS[ZONE_INDEX[s]].toLowerCase() : 'no reading');

const REDUCED_QUERY = window.matchMedia
  ? window.matchMedia('(prefers-reduced-motion: reduce)') : null;
let reduced = !!(REDUCED_QUERY && REDUCED_QUERY.matches);
if (REDUCED_QUERY && REDUCED_QUERY.addEventListener) {
  REDUCED_QUERY.addEventListener('change', (e) => { reduced = e.matches; });
}

// ── the palette, read once per draw ───────────────────────────────────────
function palette() {
  const css = getComputedStyle(document.documentElement);
  const get = (name) => css.getPropertyValue(name).trim() || '#fff';
  return {
    text: get('--text'), text3: get('--text-3'), line: get('--line'), bg: get('--bg'),
    ok: get('--ok'), warn: get('--warn'), bad: get('--bad'), grey: get('--grey'),
  };
}

// ── the engraving ─────────────────────────────────────────────────────────
// Straight from the reference the client approved (001-etched-disc): one hair
// per instant, radius from its dominant frequency, a gradient that fades at
// both ends so the marks read as engraving rather than as bars, and a second
// pass of half-scale ticks that gives the face its texture.
const RING = 520;
const ART_IN = 140;
const ART_SPAN = 380;
const ART_STEP = (t, from, to) => from + Math.max(0, Math.min(1, t)) * (to - from);

let engraveCache = { key: '', canvas: null };

function dominating(bytes, cols, rows) {
  const f = new Float32Array(cols);
  const power = new Float32Array(cols);
  for (let n = 0; n < cols; n += 1) {
    let best = 0, row = 0;
    for (let i = 0; i < rows; i += 1) {
      const v = bytes[i * cols + n];
      if (v > best) { best = v; row = i; }
    }
    f[n] = rows > 1 ? row / (rows - 1) : 0;
    power[n] = best / 255;
  }
  return { f, power };
}

function engravingFor(strip, faint) {
  const { cols, rows, bytes } = strip;
  const key = `${cols}x${rows}:${bytes.length}:${strip.mdf ? strip.mdf.length : 0}:`
    + `${faint ? 'f' : 's'}:${palette().text}`;
  if (engraveCache.key === key && engraveCache.canvas) return engraveCache.canvas;

  const W = DIAL.size, c = document.createElement('canvas');
  c.width = W; c.height = W;
  const ctx = c.getContext('2d');
  const cx = W / 2, cy = W / 2;
  const colour = palette().text;
  const { f, power } = dominating(bytes, cols, rows);
  const dim = faint ? 0.78 : 1;

  // the recording: one hair per instant
  for (let n = 0; n < cols; n += 1) {
    const theta = (n / cols) * Math.PI * 2 - Math.PI / 2;
    const r = ART_STEP(f[n], ART_IN, ART_IN + ART_SPAN);
    const len = 66 + power[n] * 120;
    const inner = r - len * 0.45, outer = r + len * 0.55;
    const cos = Math.cos(theta), sin = Math.sin(theta);
    const g = ctx.createLinearGradient(cx + cos * inner, cy + sin * inner,
                                       cx + cos * outer, cy + sin * outer);
    g.addColorStop(0, 'transparent');
    g.addColorStop(0.5, colour);
    g.addColorStop(1, 'transparent');
    ctx.globalAlpha = (0.26 + Math.sqrt(power[n]) * 0.5) * dim;
    ctx.strokeStyle = g;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(cx + cos * inner, cy + sin * inner);
    ctx.lineTo(cx + cos * outer, cy + sin * outer);
    ctx.stroke();
  }

  // the harmonic pass: twice as many, half as far out, a whisper
  for (let n = 0; n < cols * 2; n += 1) {
    const theta = (n / (cols * 2)) * Math.PI * 2 - Math.PI / 2;
    const r = ART_STEP(f[Math.floor(n / 2)] * 0.55, ART_IN, ART_IN + ART_SPAN);
    const cos = Math.cos(theta), sin = Math.sin(theta);
    ctx.globalAlpha = 0.09 * dim;
    ctx.strokeStyle = colour;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx + cos * (r - 22), cy + sin * (r - 22));
    ctx.lineTo(cx + cos * (r + 22), cy + sin * (r + 22));
    ctx.stroke();
  }
  ctx.globalAlpha = 1;

  engraveCache = { key, canvas: c };
  return c;
}

// ── the dial ──────────────────────────────────────────────────────────────
let dialLoop = 0;
let dialPlayed = '';
let dialStart = 0;
let dialState = null;
let dialHover = -1;
let dialLastPlay = 0;
let dialNonce = 0;

function animateDial(canvas, strip, opts, key) {
  if (!canvas) return;
  dialState = { canvas, strip, opts, key };
  const playing = dialLoop !== 0;

  if (reduced || !key || key === dialPlayed) {
    if (!playing) drawDial(canvas, strip, Object.assign({}, opts, { progress: 1 }));
    if (key) dialPlayed = key;
    return;
  }
  if (playing) return;

  dialStart = performance.now();
  const frame = (now) => {
    const state = dialState;
    const t = Math.min(1, (now - dialStart) / 900);
    drawDial(state.canvas, state.strip, Object.assign({}, state.opts, { progress: t }));
    if (t < 1) {
      dialLoop = requestAnimationFrame(frame);
    } else {
      dialLoop = 0;
      dialPlayed = state.key;
      dialLastPlay = Date.now();
    }
  };
  dialLoop = requestAnimationFrame(frame);
}

function drawDial(canvas, strip, opts) {
  const options = opts || {};
  const status = options.status || 'grey';
  const W = DIAL.size, cx = W / 2, cy = W / 2;
  if (canvas.width !== W) { canvas.width = W; canvas.height = W; }
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, W);

  const shown = canvas.clientWidth || 560;
  const typeK = shown < 420 ? 1.7 : 1;       // the reference sizes, phone-adjusted
  const pal = palette();
  const p = options.progress === undefined ? 1 : options.progress;
  const clamp = (v) => Math.max(0, Math.min(1, v));
  const sweep = clamp(p / 0.62);
  const ringIn = clamp((p - 0.25) / 0.35);
  const lineIn = clamp((p - 0.45) / 0.35);
  const wordIn = clamp((p - 0.75) / 0.25);

  // the recording, swept in like ink around the dial
  if (strip && strip.bytes) {
    const art = engravingFor(strip, !!options.sample);
    if (sweep >= 1) {
      ctx.drawImage(art, 0, 0);
    } else {
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.arc(cx, cy, 600, -Math.PI / 2, -Math.PI / 2 + sweep * Math.PI * 2);
      ctx.closePath();
      ctx.clip();
      ctx.drawImage(art, 0, 0);
      ctx.restore();
    }
  }

  // faint concentric rules
  ctx.strokeStyle = pal.line;
  for (let r = 140; r <= 520; r += 40) {
    ctx.globalAlpha = ringIn * 0.5;
    ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.stroke();
  }

  // three zones: the traffic light bent into a circle
  for (let z = 0; z < 3; z += 1) {
    const a0 = -Math.PI / 2 + ZONE_EDGES[z] * Math.PI * 2;
    const a1 = -Math.PI / 2 + ZONE_EDGES[z + 1] * Math.PI * 2;
    const active = ZONE_WORDS[z] === DIAL_WORD[status];
    ctx.beginPath(); ctx.arc(cx, cy, RING, a0, a1);
    ctx.strokeStyle = pal[ZONE_TOKENS[z].slice(2)];
    ctx.globalAlpha = ringIn * (options.sample ? 0.10 : (active ? 0.30 : 0.14));
    ctx.lineWidth = 14; ctx.stroke();

    const mid = (a0 + a1) / 2;
    ctx.globalAlpha = ringIn * 0.42;
    ctx.fillStyle = pal.text;
    ctx.font = `500 ${Math.round(15 * typeK)}px ui-monospace, SFMono-Regular, Menlo, monospace`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(ZONE_WORDS[z], cx + Math.cos(mid) * (RING + 32),
      cy + Math.sin(mid) * (RING + 32));
  }

  // the line that carries the verdict, drawn on as the sweep passes it
  if (strip && strip.mdf && strip.mdf.length) {
    const upto = Math.round((strip.mdf.length - 1) * sweep);
    ctx.beginPath();
    let started = false;
    for (let n = 0; n <= upto; n += 1) {
      const f = strip.mdf[n];
      if (!f) continue;
      const theta = (n / strip.cols) * Math.PI * 2 - Math.PI / 2;
      const r = ART_STEP((f - 1) / 34, ART_IN, ART_IN + ART_SPAN);
      const x = cx + Math.cos(theta) * r, y = cy + Math.sin(theta) * r;
      if (!started) { ctx.moveTo(x, y); started = true; } else { ctx.lineTo(x, y); }
    }
    ctx.strokeStyle = pal[zoneToken(status).slice(2)] || pal.grey;
    ctx.globalAlpha = lineIn * (options.sample ? 0.4 : 1);
    ctx.lineWidth = 3.2;
    ctx.lineJoin = 'round';
    ctx.stroke();
  }

  // the reading: a needle at how spent the muscle is, not a progress bar
  if (!options.sample && status !== 'grey' && wordIn > 0) {
    const score = Math.max(0, Math.min(100, options.score || 0));
    let lo = 0.02, hi = 0.28, frac = (100 - score) / 30;
    if (status === 'red') { lo = 0.60; hi = 0.94; frac = (39 - Math.min(score, 39)) / 39; }
    else if (status === 'amber') { lo = 0.32; hi = 0.58; frac = (69 - score) / 29; }
    const spent = lo + (hi - lo) * clamp(frac);
    const a = -Math.PI / 2 + spent * Math.PI * 2;
    const x = cx + Math.cos(a) * RING, y = cy + Math.sin(a) * RING;
    ctx.globalAlpha = wordIn;
    ctx.beginPath(); ctx.arc(x, y, 11, 0, Math.PI * 2);
    ctx.fillStyle = pal.text; ctx.fill();
    ctx.beginPath(); ctx.arc(x, y, 5.5, 0, Math.PI * 2);
    ctx.fillStyle = pal[zoneToken(status).slice(2)] || pal.grey; ctx.fill();
  }

  if (wordIn <= 0) { ctx.globalAlpha = 1; return; }

  // centre: subject, verdict, age
  ctx.globalAlpha = wordIn;
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillStyle = pal.text3;
  ctx.font = `500 ${Math.round(24 * typeK)}px ui-monospace, SFMono-Regular, Menlo, monospace`;
  ctx.fillText((options.subject || '').toUpperCase(), cx, cy - 78 * typeK);

  const label = options.word || '';
  ctx.fillStyle = pal.text;
  let size = 96;
  const face = '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
  ctx.font = `300 ${size}px ${face}`;
  while (size > 22 && ctx.measureText(label).width > 420) {
    size *= 0.92;
    ctx.font = `300 ${size}px ${face}`;
  }
  ctx.fillText(label, cx, cy + 8);

  ctx.fillStyle = pal.text3;
  ctx.font = `400 ${Math.round(21 * typeK)}px ui-monospace, SFMono-Regular, Menlo, monospace`;
  ctx.fillText(options.foot || '', cx, cy + 78 * typeK);
  ctx.globalAlpha = 1;

  // the reference's one interaction: the hair under the pointer ignites
  if (dialHover >= 0 && strip && strip.bytes && canvas === $('#home-disc')) {
    const { cols, rows, bytes } = strip;
    const { f, power } = dominating(bytes, cols, rows);
    const n = Math.min(cols - 1, dialHover);
    const theta = (n / cols) * Math.PI * 2 - Math.PI / 2;
    const r = ART_STEP(f[n], ART_IN, ART_IN + ART_SPAN);
    const len = 66 + power[n] * 120;
    const inner = r - len * 0.45, outer = r + len * 0.55;
    const cos = Math.cos(theta), sin = Math.sin(theta);
    ctx.globalAlpha = 1;
    ctx.strokeStyle = pal.text;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(cx + cos * inner, cy + sin * inner);
    ctx.lineTo(cx + cos * outer, cy + sin * outer);
    ctx.stroke();
  }
}

// ── naming ────────────────────────────────────────────────────────────────
const PLAIN_NAME = { VL: 'quad', VM: 'quad', RF: 'quad', BF: 'hamstring',
                     ST: 'hamstring', TA: 'shin', GM: 'calf', GL: 'calf' };

function plainName(result) {
  return PLAIN_NAME[result.id] || 'muscle';
}

function muscleName(result) {
  if (result.name) return result.name;
  const up = state.upload;
  if (up && up.muscles) {
    const match = up.muscles.find((m) => m.column === result.column && m.recognised);
    if (match && match.name) return match.name;
  }
  return result.column || result.id || 'muscle';
}

// ── copy ──────────────────────────────────────────────────────────────────
const HOME_COPY = {
  red: (n) => `Your ${n} is showing fatigue. Rest it today, or keep it very light, `
    + 'and check again tomorrow.',
  amber: (n) => `Your ${n} is carrying some fatigue. Keep the volume down today `
    + 'and check again tomorrow.',
  green: (n) => `No fatigue detected in this recording of your ${n}. Train as `
    + 'planned, and stop if anything hurts.',
};

const LAST_COPY = {
  red: (n, when) => `On ${when} your ${n} was showing fatigue.`,
  amber: (n, when) => `On ${when} your ${n} was carrying some fatigue.`,
  green: (n, when) => `On ${when} your ${n} showed no fatigue.`,
};

function whyPlain(result) {
  const mdf = Number(result.delta_mdf);
  const rms = Number(result.delta_rms);
  if (!Number.isFinite(mdf) || !Number.isFinite(rms)) {
    return 'This recording did not have enough clean signal at both ends, so '
      + 'only the overall picture is reported.';
  }
  const freq = mdf < 0 ? `fell ${Math.abs(Math.round(mdf))}%` : `rose ${Math.round(mdf)}%`;
  const amp = rms < 0 ? `fell ${Math.abs(Math.round(rms))}%` : `rose ${Math.round(rms)}%`;
  const moved = `Over the hold the signal's median frequency ${freq} and its `
    + `amplitude ${amp}.`;

  if (result.level === 'fatigued' || result.status === 'red') {
    return `${moved} Both moves point the same way, which is the pattern of a `
      + 'tiring muscle.';
  }
  if (result.level === 'mixed' || result.status === 'amber') {
    return `${moved} One of the two moved without much change in the other, which `
      + 'is the pattern of early fatigue.';
  }
  return `${moved} Neither moved far, which is what a fresh muscle looks like.`;
}

// ── dates ─────────────────────────────────────────────────────────────────
const TIME_FMT = new Intl.DateTimeFormat(undefined, { hour: 'numeric', minute: '2-digit' });
const DATE_FMT = new Intl.DateTimeFormat(undefined,
  { weekday: 'short', day: 'numeric', month: 'short' });

function whenLabel(iso) {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '';
  const now = new Date();
  const midnight = (x) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
  const days = Math.round((midnight(now) - midnight(d)) / 864e5);
  const time = TIME_FMT.format(d);
  if (days <= 0) return `today ${time}`;
  if (days === 1) return `yesterday ${time}`;
  const year = d.getFullYear() === now.getFullYear() ? '' : ` ${d.getFullYear()}`;
  return `${DATE_FMT.format(d)}${year} ${time}`;
}

// ── the model ─────────────────────────────────────────────────────────────
// One pure function decides what the screen may claim, so the answer cannot
// disagree with the evidence: no verdict from a sample, from a recording that
// could not be read, or from a check older than a day and a half.
function worstOf(results) {
  const scored = (results || [])
    .filter((r) => r && r.status in ZONE_INDEX)
    .sort((a, b) => ((RANK[b.status] || 0) - (RANK[a.status] || 0))
      || ((a.score || 0) - (b.score || 0)));
  return scored[0] || null;
}

function firstSaved() {
  for (const row of state.history || []) {
    if (!Array.isArray(row.results) || !row.results.length) continue;
    const worst = worstOf(row.results);
    if (worst) return { row, worst };
  }
  return null;
}

function homeModel() {
  const analysis = state.analysis;
  const results = analysis ? Object.values(analysis.results || {}) : [];
  const worst = worstOf(results);
  const now = Date.now();

  if (worst && analysis && analysis.sample) {
    return { mode: 'sample', worst, results, canClaim: false, multi: results.length > 1 };
  }
  if (worst) {
    const ageH = analysis.at ? (now - analysis.at) / 36e5 : 0;
    return { mode: 'measured', worst, results, at: analysis.at, ageH,
             stale: ageH > FRESH_H, canClaim: ageH <= FRESH_H,
             multi: results.length > 1, strip: worst.strip || null };
  }
  if (analysis && results.length) {
    return { mode: 'unreadable', worst: null, results, canClaim: false };
  }

  const saved = firstSaved();
  if (saved) {
    const ageH = (now - new Date(saved.row.ts).getTime()) / 36e5;
    return { mode: 'last', worst: saved.worst, results: saved.row.results,
             at: saved.row.ts, ageH, stale: ageH > FRESH_H,
             canClaim: ageH <= FRESH_H, multi: saved.row.results.length > 1 };
  }
  if ((state.history || []).length) {
    return { mode: 'unreadable', worst: null, results: [], canClaim: false };
  }
  return { mode: 'none', worst: null, results: [], canClaim: false };
}

// ── painting ──────────────────────────────────────────────────────────────
let homeSeq = 0;

function paintHome(model) {
  const { mode, worst, results } = model;
  const status = worst ? worst.status : 'grey';
  const name = worst ? muscleName(worst) : 'muscle';
  const plain = worst ? plainName(worst) : 'muscle';
  const seconds = worst && worst.analysed_seconds
    ? Math.round(worst.analysed_seconds) : null;
  const claim = model.canClaim;
  const stale = !!model.stale;

  const ask = $('#home-ask'), title = $('#home-title'), lead = $('#home-lead');
  const verdict = $('#home-verdict'), more = $('#home-more');
  const savedWhen = model.at ? whenLabel(model.at) : '';

  if (mode === 'measured' && claim) {
    ask.textContent = `Checked ${(state.checkedAt || 'just now').toLowerCase()}`;
    title.textContent = `Is your ${plain} ready?`;
    lead.textContent = HOME_COPY[status](plain);
  } else if (mode === 'measured' || mode === 'last') {
    ask.textContent = savedWhen ? `Last check · ${savedWhen}` : 'Last check';
    title.textContent = `Check your ${plain} again`;
    lead.textContent = mode === 'measured'
      ? `This reading is from ${savedWhen}. Check again for today's answer.`
      : `${LAST_COPY[status](plain, savedWhen)} Check again before you train.`;
  } else if (mode === 'sample') {
    ask.textContent = 'Sample check';
    title.textContent = 'This is the sample';
    lead.textContent = 'A real recording, so you can see how a check reads. Run '
      + 'your own to replace it.';
  } else if (mode === 'unreadable') {
    ask.textContent = 'Check attempted';
    title.textContent = "Couldn't read this recording";
    lead.textContent = (results[0] && results[0].tip)
      || 'The signal was too small or too flat to measure. Check the sensors and try again.';
  } else {
    ask.textContent = 'No check yet';
    title.textContent = 'Check a muscle';
    lead.textContent = 'Hold a steady contraction for 30 seconds. A 30-second hold '
      + 'shows whether this muscle tires early.';
  }

  const showVerdict = claim && !stale && (mode === 'measured' || mode === 'last');
  verdict.hidden = !showVerdict;
  if (showVerdict) {
    verdict.querySelector('i').style.background = cssVar(VERDICT_CLASS[status]);
    verdict.querySelector('i').style.boxShadow = '';
    $('#home-verdict-text').textContent = `${VERDICT_WORD[status]}`
      + (model.multi ? ` · worst of ${results.length}` : '');
    $('#home-verdict-text').style.color = cssVar(VERDICT_CLASS[status]);
  }

  more.hidden = !showVerdict;
  if (showVerdict && worst) {
    $('#why-word').textContent = VERDICT_WORD[status];
    $('#why-plain').textContent = whyPlain(worst);
    const list = $('#det-body');
    list.replaceChildren();
    const rows = [['Muscle', name]];
    if (Number.isFinite(Number(worst.delta_mdf))) {
      rows.push(['Median frequency', worst.mdf_start !== undefined
        ? `${worst.mdf_start} → ${worst.mdf_end} Hz, ${Math.round(worst.delta_mdf)}%`
        : `${Math.round(worst.delta_mdf)}%`]);
      rows.push(['Amplitude (RMS)', `${worst.delta_rms > 0 ? '+' : ''}`
        + `${Math.round(worst.delta_rms)}%`]);
    }
    if (seconds) rows.push(['Hold', `${seconds} s`]);
    rows.push(['Method', 'Zhao et al. 2025']);
    for (const [label, value] of rows) {
      list.append(el('dt', { text: label }), el('dd', { text: value }));
    }
  }

  const dialWord = mode === 'none' ? 'NO CHECK'
    : mode === 'sample' ? 'SAMPLE'
      : mode === 'unreadable' ? 'NO READ'
        : showVerdict ? DIAL_WORD[status] : 'STALE';

  const cta = (mode === 'measured' && claim) || mode === 'last'
    ? 'Check again · 30 s' : 'Start a check · 30 s';
  $('#home-start').textContent = cta;
  $('#home-start-m').textContent = cta;
  const hint = document.querySelector('.home-cta .hint');
  if (hint) hint.textContent = state.profile ? 'Saved to your history' : 'Sign in to save this';

  // The engraving is drawn from this athlete's own recording. The bundled
  // sample is used only where it is labelled as one: the empty screen and the
  // sample itself.
  const strip = model.strip
    || ((mode === 'none' || mode === 'sample') ? stripData : null);
  if (strip && strip.grid && !strip.bytes) {
    strip.bytes = Uint8Array.from(atob(strip.grid), (c) => c.charCodeAt(0));
  }
  const dialKey = `${mode}|${status}|${worst ? worst.score : ''}|${model.at || ''}|${dialNonce}`;
  animateDial($('#home-disc'), strip, {
    status: mode === 'unreadable' || !showVerdict ? 'grey' : status,
    sample: mode === 'sample' || mode === 'none',
    score: worst ? worst.score : 0,
    subject: mode === 'sample' || mode === 'none' ? '' : plain.toUpperCase(),
    word: dialWord,
    foot: mode === 'sample' ? 'NOT YOUR DATA'
      : seconds && showVerdict ? `${seconds} S HOLD`
        : mode === 'last' ? (savedWhen || '').toUpperCase() : '',
  }, dialKey);

  const disc = $('#home-disc');
  disc.setAttribute('aria-label', showVerdict
    ? `${name}: ${VERDICT_WORD[status]}, score ${worst ? worst.score : 0} of 100, `
      + `in the ${zoneWord(status)} band.${stale ? ' Stale reading.' : ''}`
    : mode === 'sample' ? 'The sample recording, not your data.'
      : mode === 'unreadable' ? 'This recording could not be measured.'
        : 'No check yet. The dial fills with your first check.');
  $('#home-disc-label').textContent = showVerdict
    ? `your recording${seconds ? ` · ${seconds} s` : ''}`
    : mode === 'last' || mode === 'measured' ? 'from your last saved check'
      : mode === 'sample' ? 'sample recording, not yours' : '';

  // the foot: one mark per check, newest on the right
  const foot = $('#home-foot');
  foot.replaceChildren();
  const history = (state.history || []).slice(0, 11).reverse();
  const marks = history.length
    ? history.map((row) => row.status || 'grey')
    : (showVerdict ? [status] : []);
  if (!marks.length) {
    foot.append(el('span', { class: 'foot-txt', text: 'No checks yet' }));
    return;
  }
  const counts = marks.reduce((acc, s) => {
    acc[s] = (acc[s] || 0) + 1;
    return acc;
  }, {});
  const summary = Object.entries(counts).map(([s, n]) => `${n} ${zoneWord(s)}`).join(', ');
  foot.append(el('span', {
    class: 'foot-txt',
    text: marks.length === 1 ? '1 check' : `last ${marks.length} checks`,
  }));
  const dots = el('span', { class: 'foot-dots', role: 'img', 'aria-label': summary });
  for (const s of marks) {
    dots.append(el('i', { class: `d ${s}`, title: zoneWord(s) }));
  }
  foot.append(dots);
  if (marks.length > 1) {
    foot.append(el('span', { class: 'foot-txt', text: 'newest on the right' }));
  }
}

async function renderHome() {
  const seq = ++homeSeq;
  const model = homeModel();
  paintHome(model);

  // The empty screen shows the sample, so it has to be fetched once.
  if (!stripData && (model.mode === 'none' || model.mode === 'sample')) {
    try {
      await loadStrip();
      if (seq === homeSeq) paintHome(homeModel());
    } catch (e) { /* no strip: the dial is still correct, just empty */ }
  }

  if (!state.profile) return;
  const staleFetch = !state.historyAt || Date.now() - state.historyAt > 60_000;
  if (!staleFetch && (state.history || []).length) return;
  try {
    const data = await api('/history');
    if (seq !== homeSeq) return;
    const before = JSON.stringify(state.history || []);
    state.history = (data && data.history) || [];
    state.historyAt = Date.now();
    if (JSON.stringify(state.history) !== before) paintHome(homeModel());
  } catch (e) { /* offline, or signed out mid-flight */ }
}

// ── train ───────────────────────────────────────────────────────────────────
const CALL = {
  green: { title: 'Train as planned', body: 'Your muscles are recovered. A normal session is fine today.' },
  amber: { title: 'Keep it light', body: 'Some muscles still show fatigue. Cut the volume and keep today easy.' },
  red: { title: 'Rest today', body: 'At least one muscle needs more time. Heavy work can wait a day.' },
  grey: { title: 'Nothing to go on', body: 'The last check could not measure anything. Check the recording and the sampling rate.' },
};

const CALL_FRESH_H = 36;     // older than this and today's call is not today's

async function renderTrain() {
  const seq = (renderTrain.seq = (renderTrain.seq || 0) + 1);
  const box = clear($('#train-body'));
  let history = [];
  let saved = null;

  if (state.profile) {
    try {
      const data = await api('/history');
      history = data.history || [];
      saved = history[0] || null;
    } catch (error) { /* not signed in, or the server is down: fall through */ }
  }

  let latest = saved;
  if (state.analysis && !state.analysis.sample) {
    const run = {
      status: state.analysis.overall.status,
      avg_score: state.analysis.overall.mean_score,
      ts: new Date(state.analysis.at || Date.now()).toISOString(),
      results: Object.values(state.analysis.results).map((r) => ({
        id: r.id, column: r.column, status: r.status, score: r.score, tip: r.tip,
      })),
      unsaved: true,
    };
    // Whichever is newer wins: a check just done outranks one saved days ago.
    if (!saved || new Date(run.ts) > new Date(saved.ts)) latest = run;
  }

  if (seq !== renderTrain.seq) return;   // a newer render is in flight

  const age = latest ? (Date.now() - new Date(latest.ts).getTime()) / 36e5 : null;
  if (latest && age > CALL_FRESH_H) {
    latest = null;
    history = history.slice(0, 7);
  }

  if (!latest) {
    const stale = history.length
      ? `The last check was ${formatDate(history[0].ts)}. Check again and the call for today appears here.`
      : 'Nothing to go on yet. Check a muscle and the call for today appears here.';
    box.append(el('div', { class: 'card empty' }, ICON.empty(),
      el('p', { text: stale }),
      el('button', { class: 'btn btn--quiet mt-16', type: 'button', onclick: () => nav('check') }, 'Start a check')));
    return;
  }

  const status = latest.status in CALL ? latest.status : 'grey';
  const call = CALL[status];
  const when = latest.unsaved ? 'From the check you just did' : 'From your check on ' + formatDate(latest.ts);

  const byColumn = new Map((state.upload ? state.upload.muscles : []).map((m) => [m.column, m]));
  const nameOf = (r) => {
    const meta = byColumn.get(r.column);
    return meta && meta.recognised ? `${meta.id} · ${meta.name}` : r.id;
  };

  box.append(el('div', { class: 'card' },
    el('div', { class: 'call' },
      el('h2', { class: `status--${status}`, text: call.title }),
      el('p', { text: call.body }),
      el('p', { class: 'meta mt-8', text: when })),
    watchList(latest.results || [], nameOf),
    latest.avg_score === null || latest.avg_score === undefined ? null
      : el('p', { class: 'meta mt-12', text: `Mean score ${Math.round(latest.avg_score)} of 100 across the muscles measured.` })));

  if (history.length) box.append(weekStrip(history));

  const actions = el('div', { class: 'btn-row mt-16' });
  actions.append(el('button', { class: 'btn btn--primary grow', type: 'button', onclick: () => nav('check') }, 'New check'));
  if (state.analysis) {
    actions.append(el('button', { class: 'btn btn--quiet', type: 'button', onclick: () => { nav('check'); setStage('results'); renderResults(); } }, 'Full results'));
  }
  box.append(actions);

  if (state.profile) {
    box.append(el('button', { class: 'btn--text mt-12', type: 'button', onclick: () => nav('history') }, 'See every check'));
  } else {
    box.append(el('div', { class: 'notice mt-16' }, ICON.info(),
      el('span', null, 'This call comes from the check you just did and is not saved. ',
        el('button', { class: 'btn--text', type: 'button', onclick: () => nav('profile') }, 'Add a profile'),
        ' to keep a history.')));
  }
}

function watchList(results, nameOf) {
  const watch = results.filter((r) => r.status === 'red' || r.status === 'amber')
    .sort((a, b) => statusRank(a.status) - statusRank(b.status));
  const rows = el('div', { class: 'rows' });
  if (!watch.length) {
    rows.append(el('div', { class: 'row' },
      el('span', { class: 'dot dot--green' }),
      el('div', { class: 'name' }, el('strong', { text: 'All clear' }),
        el('span', { text: 'No muscle needs special attention today.' })),
      el('span', { class: 'score score--green', text: '—' }),
      el('span', { class: 'word', text: '' })));
    return rows;
  }
  for (const r of watch) {
    rows.append(el('div', { class: 'row' },
      el('span', { class: `dot dot--${r.status}` }),
      el('div', { class: 'name' },
        el('strong', { text: nameOf(r) }),
        el('span', { text: r.status === 'red' ? 'Give this one a day off.' : 'Go easy on this one today.' })),
      el('span', { class: `score score--${r.status}`, text: r.score === undefined ? '—' : String(r.score) }),
      el('span', { class: 'word', text: WORDS[r.status] || '' })));
  }
  return rows;
}

function weekStrip(history) {
  const strip = el('div', { class: 'week' });
  for (let i = 6; i >= 0; i -= 1) {
    const day = new Date(Date.now() - i * 864e5);
    const hit = history.find((h) => new Date(h.ts).toDateString() === day.toDateString());
    const label = day.toLocaleDateString(undefined, { weekday: 'short' }).slice(0, 2);
    const status = hit ? hit.status : null;
    strip.append(el('div', null,
      el('span', { class: `dot ${status ? 'dot--' + status : 'none'}` }),
      el('span', { text: label }),
      el('span', { text: hit && hit.avg_score !== null && hit.avg_score !== undefined ? String(Math.round(hit.avg_score)) : '' })));
  }
  return el('div', { class: 'card' }, el('h3', { text: 'Last seven days' }), strip);
}

// ── history ─────────────────────────────────────────────────────────────────
async function loadHistory() {
  const box = clear($('#history-body'));
  if (!state.profile) {
    box.append(el('div', { class: 'card empty' }, ICON.empty(),
      el('div', { text: 'Choose a profile to see saved checks.' }),
      el('button', { class: 'btn btn--quiet mt-12', type: 'button', onclick: () => nav('profile') }, 'Choose profile')));
    return;
  }
  box.append(el('div', { class: 'card' }, el('p', { class: 'meta', text: 'Loading…' })));
  try {
    const data = await api('/history');
    state.history = data.history || [];
    renderHistory();
  } catch (error) {
    clear(box).append(el('div', { class: 'card empty', text: error.message }));
  }
}

function renderHistory() {
  const box = clear($('#history-body'));
  const rows = state.history;
  if (!rows.length) {
    box.append(el('div', { class: 'card empty' }, ICON.empty(),
      el('p', { text: 'No checks saved yet for ' + state.profile.name + '.' })));
    return;
  }
  const scored = rows.filter((r) => r.avg_score !== null && r.avg_score !== undefined);
  if (scored.length >= 2) {
    box.append(el('div', { class: 'card' }, el('h2', { text: 'Trend' }), trendSvg(scored.slice(0, 12).reverse())));
  }
  if (scored.length) {
    const best = Math.round(Math.max(...scored.map((r) => r.avg_score)));
    const mean = Math.round(scored.reduce((a, b) => a + b.avg_score, 0) / scored.length);
    box.append(el('div', { class: 'card summary3' },
      el('div', null, el('strong', { text: String(best) }), el('span', { text: 'Best' })),
      el('div', null, el('strong', { text: String(mean) }), el('span', { text: 'Average' })),
      el('div', null, el('strong', { text: String(rows.length) }), el('span', { text: 'Checks' }))));
  }
  for (const item of rows) {
    const status = item.status || 'grey';
    box.append(el('button', { class: 'hist', type: 'button', onclick: () => openHistoryItem(item) },
      el('span', { class: `score score--${status}`, text: item.avg_score === null ? '—' : String(Math.round(item.avg_score)) }),
      el('span', { class: 'when' },
        el('strong', { text: formatDate(item.ts) }),
        el('span', { text: `${WORDS[status] || status}${item.filename ? ' · ' + item.filename : ''}` }))));
  }
}

function trendSvg(items) {
  const W = 320, H = 96, PAD = 8;
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('class', 'trend');
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', 'Score trend, oldest to newest');
  const y = (score) => H - PAD - (score / 100) * (H - 2 * PAD);
  const x = (i) => PAD + i * (W - 2 * PAD) / Math.max(1, items.length - 1);
  for (const level of [70, 40]) {
    const line = document.createElementNS(ns, 'line');
    line.setAttribute('x1', PAD); line.setAttribute('x2', W - PAD);
    line.setAttribute('y1', y(level)); line.setAttribute('y2', y(level));
    line.setAttribute('stroke', 'currentColor');
    line.setAttribute('stroke-opacity', '0.15');
    line.setAttribute('stroke-dasharray', '3 3');
    svg.append(line);
  }
  const points = items.map((item, i) => `${x(i)},${y(item.avg_score)}`).join(' ');
  const path = document.createElementNS(ns, 'polyline');
  path.setAttribute('points', points);
  path.setAttribute('fill', 'none');
  path.setAttribute('stroke', cssVar('--accent'));
  path.setAttribute('stroke-width', '2');
  path.setAttribute('stroke-linejoin', 'round');
  svg.append(path);
  items.forEach((item, i) => {
    const dot = document.createElementNS(ns, 'circle');
    dot.setAttribute('cx', x(i)); dot.setAttribute('cy', y(item.avg_score));
    dot.setAttribute('r', '4');
    dot.setAttribute('fill', cssVar('--surface'));
    dot.setAttribute('stroke', tokenRgb(STATUS_TOKEN[item.status] || '--grey'));
    dot.setAttribute('stroke-width', '2.5');
    svg.append(dot);
  });
  return svg;
}

function openHistoryItem(item) {
  const body = clear($('#sheet-body'));
  $('#sheet-title').textContent = formatDate(item.ts);
  const rows = el('div', { class: 'rows' });
  for (const r of item.results || []) {
    rows.append(el('div', { class: 'row' },
      el('span', { class: `dot dot--${r.status || 'grey'}` }),
      el('div', { class: 'name' }, el('strong', { text: r.id }), el('span', { text: r.tip || '' })),
      el('span', { class: `score score--${r.status || 'grey'}`, text: r.score === null ? '—' : String(r.score) }),
      el('span', { class: 'word', text: WORDS[r.status] || '' })));
  }
  body.append(rows);
  if (item.filename) body.append(el('p', { class: 'meta mt-12', text: item.filename }));
  $('#sheet').hidden = false;
}

// ── profile ─────────────────────────────────────────────────────────────────
async function renderProfile() {
  const seq = (renderProfile.seq = (renderProfile.seq || 0) + 1);
  const box = clear($('#profile-body'));
  if (state.profile) {
    box.append(el('div', { class: 'card' },
      el('h2', { text: state.profile.name }),
      el('p', { class: 'meta', text: 'Results are saved under this name.' }),
      el('div', { class: 'btn-row mt-12' },
        el('button', { class: 'btn btn--quiet', type: 'button', onclick: signOut }, 'Sign out'),
        el('button', { class: 'btn btn--quiet', type: 'button', onclick: deleteMine }, 'Delete profile'))));
  } else {
    box.append(el('div', { class: 'card' },
      el('h2', { text: 'Not signed in' }),
      el('p', { class: 'meta', text: 'Choose your name below, or add a new profile.' })));
  }

  let profiles = [];
  try {
    const data = await api('/profiles');
    if (seq !== renderProfile.seq) return;   // a newer render is in flight
    profiles = data.profiles || [];
  } catch (error) {
    if (seq !== renderProfile.seq) return;
    box.append(el('div', { class: 'card empty', text: error.message }));
  }
  if (profiles.length) {
    const list = el('div', { class: 'card' }, el('h3', { text: 'Profiles on this server' }));
    const choices = el('div', { class: 'choices' });
    for (const profile of profiles) {
      const active = state.profile && state.profile.id === profile.id;
      choices.append(el('button', {
        class: 'hist mb-0', type: 'button',
        onclick: () => chooseProfile(profile),
      }, el('span', { class: 'when' },
        el('strong', { text: profile.name }),
        el('span', { text: profile.has_pin ? 'PIN protected' : 'No PIN' })),
        active ? el('span', { class: 'meta', text: 'Signed in' }) : null));
    }
    list.append(choices);
    box.append(list);
  }

  if (!state.profile) {
    box.append(el('div', { class: 'card' },
      el('h3', { text: 'New profile' }),
      el('label', { class: 'field mt-12' },
        el('span', { text: 'Name' }),
        el('input', { type: 'text', id: 'new-name', maxlength: '30', autocomplete: 'off' })),
      el('label', { class: 'field' },
        el('span', { text: 'PIN (optional, four digits)' }),
        el('input', { type: 'password', id: 'new-pin', inputmode: 'numeric', maxlength: '4', autocomplete: 'off' })),
      el('button', { class: 'btn btn--primary', type: 'button', onclick: createProfile }, 'Add profile')));
  }

  box.append(el('div', { class: 'card' },
    el('h3', { text: 'About' }),
    el('p', { class: 'meta mt-8', text: 'FAST Athlete reads surface EMG from one muscle at a time and says whether it is ready to train. Scores run from 0 to 100: 70 and above is ready, 40 to 69 is go easy, below 40 is rest. Checks you save are stored on this server.' }),
    el('p', { class: 'meta mt-8', text: 'The method is a research prototype. It is not a diagnosis.' }),
    el('p', { class: 'meta mt-12', id: 'version-line', text: '' })));

  loadVersion();
}

async function chooseProfile(profile) {
  if (!profile.has_pin) {
    return signIn(profile.id, '');
  }
  const input = el('input', {
    type: 'password', id: 'pin-entry', inputmode: 'numeric',
    maxlength: '4', autocomplete: 'off',
  });
  const submit = () => {
    const pin = input.value.trim();
    if (!/^[0-9]{4}$/.test(pin)) {
      input.focus();
      return toast('A PIN is four digits');
    }
    $('#sheet').hidden = true;
    signIn(profile.id, pin);
  };
  input.addEventListener('keydown', (event) => { if (event.key === 'Enter') submit(); });

  const body = clear($('#sheet-body'));
  $('#sheet-title').textContent = profile.name;
  body.append(
    el('p', { class: 'meta', text: 'This profile is protected by a four-digit PIN.' }),
    el('label', { class: 'field mt-12' }, el('span', { text: 'PIN' }), input),
    el('div', { class: 'btn-row' },
      el('button', { class: 'btn btn--quiet', type: 'button', onclick: () => { $('#sheet').hidden = true; } }, 'Cancel'),
      el('button', { class: 'btn btn--primary', type: 'button', onclick: submit }, 'Sign in')));
  $('#sheet').hidden = false;
  input.focus();
}

async function signIn(id, pin) {
  try {
    const data = await api('/profiles/verify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id, pin }),
    });
    state.profile = { id: data.id, name: data.name };
    state.history = []; state.historyAt = 0;
    state.analysis = null; state.checkedAt = null;
    toast(`Signed in as ${data.name}`);
    renderProfile();
    if (state.screen === 'home') renderHome();
  } catch (error) {
    toast(error.message);
  }
}

async function createProfile() {
  const name = $('#new-name').value.trim();
  const pin = $('#new-pin').value.trim();
  if (!name) return toast('Enter a name first');
  const button = document.querySelector('#profile-body .btn--primary');
  if (button && button.disabled) return;
  if (button) button.disabled = true;
  try {
    const created = await api('/profiles', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, pin: pin || null }),
    });
    await signIn(created.id, pin);
  } catch (error) {
    toast(error.message);
  } finally {
    if (button) button.disabled = false;
  }
}

function resetUserState() {
  state.profile = null;
  state.history = []; state.historyAt = 0; state.checkedAt = null;
  state.analysis = null;
  state.upload = null;
  state.file = null;
  state.selected = new Set();
  state.saved = false;
  renderTrain.seq = (renderTrain.seq || 0) + 1;   // cancel a render in flight
  homeSeq += 1;
  if (state.screen === 'home') renderHome();
  clear($('#home-last'));
  clear($('#train-body'));
  setStage('idle');
}

async function signOut() {
  try { await api('/profiles/logout', { method: 'POST' }); } catch (e) { /* ignore */ }
  resetUserState();
  toast('Signed out');
  renderProfile();
}

async function deleteMine() {
  if (!state.profile) return;
  if (!window.confirm(`Delete ${state.profile.name} and every saved result?`)) return;
  try {
    await api(`/profiles/${state.profile.id}`, { method: 'DELETE' });
    resetUserState();
    toast('Profile deleted');
    renderProfile();
  } catch (error) {
    toast(error.message);
  }
}

// ── home ────────────────────────────────────────────────────────────────────
async function renderHomeLast() {
  const seq = (renderHomeLast.seq = (renderHomeLast.seq || 0) + 1);
  const box = clear($('#home-last'));
  if (!state.profile) return;
  try {
    const data = await api('/history');
    if (seq !== renderHomeLast.seq) return;   // a newer render is in flight
    const last = (data.history || [])[0];
    if (!last) return;
    box.append(el('div', { class: 'card mt-24' },
      el('h3', { text: 'Last check' }),
      el('p', { class: 'meta mt-4' },
        `${formatDate(last.ts)} · `,
        el('span', { class: `status--${last.status}`, text: WORDS[last.status] || 'No reading' }),
        last.avg_score !== null ? ` · ${Math.round(last.avg_score)}` : ''),
      el('button', { class: 'btn--text mt-8', type: 'button', onclick: () => nav('history') }, 'See history')));
  } catch (e) { /* not signed in */ }
}

// ── formatting ──────────────────────────────────────────────────────────────
function signed(value) {
  const rounded = Math.round(value);
  return rounded > 0 ? `+${rounded}` : String(rounded);
}

function formatDuration(seconds) {
  if (seconds === null || seconds === undefined) return '—';
  const total = Math.round(seconds);
  if (total < 60) return `${total} s`;
  const m = Math.floor(total / 60);
  return `${m} min ${total - m * 60} s`;
}

async function loadVersion() {
  const line = document.getElementById('version-line');
  if (!line) return;
  try {
    const health = await api('/healthz');
    line.textContent = `Version ${health.version}` + (health.engine ? ` · ${health.engine}` : '')
      + (health.build && health.build !== 'local' ? ` · build ${health.build}` : '');
  } catch (error) {
    line.textContent = 'Version unavailable';
  }
}

function formatDate(iso) {
  try {
    return new Date(iso).toLocaleString(undefined, {
      day: 'numeric', month: 'short', hour: 'numeric', minute: '2-digit',
    });
  } catch (e) { return iso || ''; }
}

// ── wiring ──────────────────────────────────────────────────────────────────
function wire() {
  document.querySelectorAll('[data-nav]').forEach((node) => {
    node.addEventListener('click', (event) => {
      event.preventDefault();
      nav(node.dataset.nav);
    });
  });
  $('#theme-toggle').addEventListener('click', toggleTheme);
  $('#profile-btn').addEventListener('click', () => nav('profile'));

  const input = $('#file-input');
  const zone = $('#dropzone');
  input.addEventListener('change', () => {
    const file = input.files && input.files[0];
    if (file) uploadFile(file);
    input.value = '';
  });
  ['dragenter', 'dragover'].forEach((name) => zone.addEventListener(name, (event) => {
    event.preventDefault(); zone.classList.add('dragover');
  }));
  ['dragleave', 'drop'].forEach((name) => zone.addEventListener(name, (event) => {
    event.preventDefault(); zone.classList.remove('dragover');
  }));
  zone.addEventListener('drop', (event) => {
    const file = event.dataTransfer && event.dataTransfer.files[0];
    if (file) uploadFile(file);
  });

  $('#demo-btn').addEventListener('click', async () => {
    try {
      const response = await fetch('/demo_recording.csv');
      if (!response.ok) throw new Error('The sample recording is unavailable');
      const blob = await response.blob();
      state.samplePending = true;
      uploadFile(new File([blob], 'sample-recording.csv', { type: 'text/csv' }), 1000);
    } catch (error) {
      toast(error.message);
    }
  });

  $('#analyse-btn').addEventListener('click', analyse);

  $('#rate-input').addEventListener('change', () => {
    const rate = Number($('#rate-input').value);
    if (!state.file) return;
    if (!rate || rate < 100 || rate > 20000) {
      toast('Enter a rate between 100 and 20000 Hz');
      return;
    }
    uploadFile(state.file, rate);
  });

  $('#home-start').addEventListener('click', () => nav('check'));
  $('#home-start-m').addEventListener('click', () => nav('check'));

  // The reference's one interaction: the hair under the pointer ignites.
  const disc = $('#home-disc');
  const redrawStatic = () => {
    if (!dialState || dialLoop) return;
    drawDial(dialState.canvas, dialState.strip,
      Object.assign({}, dialState.opts, { progress: 1 }));
  };
  disc.addEventListener('pointermove', (event) => {
    const box = disc.getBoundingClientRect();
    if (!box.width) return;
    const W = DIAL.size;
    const x = ((event.clientX - box.left) / box.width) * W - W / 2;
    const y = ((event.clientY - box.top) / box.height) * W - W / 2;
    if (Math.hypot(x, y) < 150) {
      if (dialHover !== -1) { dialHover = -1; redrawStatic(); }
      return;
    }
    let angle = Math.atan2(y, x) + Math.PI / 2;
    if (angle < 0) angle += Math.PI * 2;
    const cols = (dialState && dialState.strip && dialState.strip.cols) || 120;
    const index = Math.round((angle / (Math.PI * 2)) * cols) % cols;
    if (index !== dialHover) { dialHover = index; redrawStatic(); }
  });
  disc.addEventListener('pointerleave', () => {
    if (dialHover !== -1) { dialHover = -1; redrawStatic(); }
  });


  $('#sheet-close').addEventListener('click', () => { $('#sheet').hidden = true; });
  $('#sheet').addEventListener('click', (event) => {
    if (event.target === $('#sheet')) $('#sheet').hidden = true;
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') $('#sheet').hidden = true;
  });

  window.addEventListener('scroll', () => {
    $('#topbar').classList.toggle('scrolled', window.scrollY > 4);
  }, { passive: true });
}

wire();
nav('home');
// The cookie outlives a reload, so the server is asked who it belongs to;
// otherwise a shared device looks signed out while the session is still live.
api('/profiles/me')
  .then((me) => { state.profile = { id: me.id, name: me.name }; })
  .catch(() => { /* not signed in */ })
  .finally(() => {
    renderHomeLast();
    if (state.screen === 'profile') renderProfile();
    if (state.screen === 'home') renderHome();
  });
