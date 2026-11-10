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
  history: [],
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
      state.history = [];
      clear($('#home-last'));
      if (state.screen === 'profile') renderProfile();
      if (state.screen === 'train') renderTrain();
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
}

// ── navigation ──────────────────────────────────────────────────────────────
function nav(screen) {
  state.screen = screen;
  for (const name of ['check', 'train', 'history', 'profile']) {
    $('#screen-' + name).classList.toggle('hidden', name !== screen);
  }
  document.querySelectorAll('.tab').forEach((tab) => {
    tab.setAttribute('aria-selected', String(tab.dataset.nav === screen));
  });
  window.scrollTo({ top: 0 });
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
      el('span', { text: `This recording is ${formatDuration(up.duration)} long. FAST analyses the first 30 seconds.` })));
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
    : 'This file does not say. FAST cannot measure frequencies without it.';
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
      el('p', { class: 'meta', text: 'Channels FAST did not recognise. Check them only if you know they are EMG.' }));
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
    state.analysis = { ...data, at: Date.now() };
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
const WORDS = { green: 'Recovered', amber: 'Some fatigue', red: 'Needs rest', grey: 'No reading' };

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
    title.append(el('span', { class: 'status--red', text: 'Needs rest' }));
    title.append(': ' + sentence(listNames(results.filter((r) => r.status === 'red').map(nameOf))));
  } else if (counts.amber) {
    title.append(el('span', { class: 'status--amber', text: 'Some fatigue' }));
    title.append(' in ' + sentence(listNames(results.filter((r) => r.status === 'amber').map(nameOf))));
  } else if (counts.green) {
    title.append(el('span', { class: 'status--green', text: 'Recovered' }));
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
      { green: 'recovered', amber: 'some fatigue', red: 'needs rest' }[key]));
  }
  box.append(el('div', { class: 'verdict' }, title, lede, countsBox));

  // rows
  const rows = el('div', { class: 'rows' });
  for (const r of results) {
    const meta = byColumn.get(r.column) || {};
    rows.append(el('div', { class: 'row' },
      el('span', { class: `dot dot--${r.status}` }),
      el('div', { class: 'name' },
        el('strong', { text: meta.id && meta.recognised ? `${meta.id} · ${meta.name}` : r.id }),
        el('span', { text: r.tip || '' })),
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
      el('dt', { text: 'Band' }), el('dd', { text: '1–35 Hz, power-weighted mean frequency' }),
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
  const results = Object.values(state.analysis.results).map((r) => ({
    id: r.id, column: r.column, status: r.status, score: r.score,
    nfi: r.nfi, tip: r.tip, centroid: r.centroid,
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
  if (state.analysis) {
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
      : 'Nothing to go on yet. Check your muscles first and the call for today appears here.';
    box.append(el('div', { class: 'card empty' }, ICON.empty(),
      el('div', { text: stale }),
      el('button', { class: 'btn btn--primary mt-16', type: 'button', onclick: () => nav('check') }, 'Check your muscles')));
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
      el('div', { text: 'No checks saved yet for ' + state.profile.name + '.' })));
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
    el('p', { class: 'meta mt-8', text: 'FAST estimates muscle fatigue from surface EMG. Scores run from 0 to 100: 70 and above reads recovered, 40 to 69 some fatigue, below 40 needs rest. Results are stored on this server.' }),
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
    toast(`Signed in as ${data.name}`);
    renderProfile();
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
  state.history = [];
  state.analysis = null;
  state.upload = null;
  state.file = null;
  state.selected = new Set();
  state.saved = false;
  renderTrain.seq = (renderTrain.seq || 0) + 1;   // cancel a render in flight
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
// The cookie outlives a reload, so the server is asked who it belongs to;
// otherwise a shared device looks signed out while the session is still live.
api('/profiles/me')
  .then((me) => { state.profile = { id: me.id, name: me.name }; })
  .catch(() => { /* not signed in */ })
  .finally(() => { renderHomeLast(); if (state.screen === 'profile') renderProfile(); });
