"""Build three concept screens for FAST Athlete, in the mould of the
nousresearch.com/ site: engraved artwork, monumental condensed type, a single
electric accent, hairline rules.

Writes ~/review-artifacts/concepts/NNN-*/index.html with the demo recording's
real data inlined, so each file opens on its own.
"""
import base64
import json
import os

ROOT = os.path.expanduser('~/review-artifacts/concepts')
APP = os.path.expanduser('~/dev/fast-athlete-review/current build')

strip = json.load(open(os.path.join(APP, 'static', 'demo-strip.json')))
GRID = strip['grid']

# A short waveform for the signal concept, from the recording itself.
import numpy as np  # noqa: E402
raw = np.loadtxt(os.path.join(APP, 'demo_recording.csv'), delimiter=',',
                 skiprows=1, usecols=[5])[:6000]          # GM column
wave = raw.reshape(-1, 5).mean(axis=1)
wave = (wave / np.abs(wave).max() * 100).round(1).tolist()
WAVE = json.dumps(wave, separators=(',', ':'))

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
         '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link href="https://fonts.googleapis.com/css2?'
         'family=Barlow+Condensed:wght@200;300;400;500&'
         'family=Archivo:wght@400;500&family=JetBrains+Mono:wght@400;500&display=swap"'
         ' rel="stylesheet">')

BASE = """
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#191930; --deep:#0D0D24; --ink:#FAFAFF; --blue:#0000F2;
  --dim:rgba(250,250,255,.52); --faint:rgba(250,250,255,.28);
  --line:rgba(250,250,255,.14);
  --display:"Barlow Condensed",Impact,sans-serif;
  --body:"Archivo",-apple-system,sans-serif;
  --mono:"JetBrains Mono",ui-monospace,monospace;
}
html,body{height:100%}
body{background:var(--bg);color:var(--ink);font-family:var(--body);
  -webkit-font-smoothing:antialiased;overflow-x:hidden}
a{color:inherit;text-decoration:none}
.frame{position:fixed;inset:14px;border:1px solid rgba(0,0,242,.9);
  pointer-events:none;z-index:50}
.mono{font-family:var(--mono);font-size:11px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--dim)}
.nav{display:flex;align-items:center;gap:38px;padding:30px 46px 0}
.nav .brand{font-family:var(--display);font-weight:400;font-size:17px;
  letter-spacing:.22em;text-transform:uppercase}
.nav .links{display:flex;gap:30px;margin-left:auto}
.nav .links a{font-family:var(--display);font-size:13px;letter-spacing:.18em;
  text-transform:uppercase;color:var(--dim);transition:color .2s}
.nav .links a:hover{color:var(--ink)}
.pill{display:inline-flex;align-items:center;gap:9px;background:var(--ink);
  color:#0B0B1A;font-family:var(--display);font-size:13px;letter-spacing:.14em;
  text-transform:uppercase;padding:11px 20px;border:0;cursor:pointer;
  transition:transform .2s,background .2s}
.pill:hover{transform:translateY(-1px);background:#fff}
"""

# ── concept A: the etched disc ─────────────────────────────────────────────
A = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FAST Athlete — is your left calf ready?</title>
""" + FONTS + "<style>" + BASE + """
:root{--ok:#3CE08C;--warn:#FFB020;--bad:#FF5468}
.hero{display:grid;grid-template-columns:1.02fr .98fr;gap:16px;
  padding:4.5vh 46px 0;align-items:center;min-height:74vh}
.ask{font-family:var(--mono);font-size:12px;letter-spacing:.15em;color:#9AA6FF;
  text-transform:uppercase;margin-bottom:18px;display:flex;align-items:center;gap:12px}
.ask i{display:inline-block;width:26px;height:1px;background:currentColor}
h1{font-family:var(--display);font-weight:200;font-size:clamp(44px,5.6vw,84px);
  line-height:.94;letter-spacing:-.02em}
h1 b{font-weight:400}
.verdict{display:flex;align-items:center;gap:12px;margin-top:26px;
  font-family:var(--display);font-size:26px;font-weight:300;letter-spacing:.01em}
.verdict i{width:12px;height:12px;border-radius:50%;background:var(--bad);
  box-shadow:0 0 0 4px rgba(255,84,104,.16),0 0 20px rgba(255,84,104,.55)}
.verdict b{color:var(--bad);font-weight:400}
.lead{margin-top:14px;max-width:38ch;font-size:15.5px;line-height:1.62;color:var(--dim)}
.cta{margin-top:28px;display:flex;align-items:center;gap:18px}
.cta .hint{font-family:var(--mono);font-size:11px;letter-spacing:.1em;color:var(--faint)}
.more{margin-top:24px}
.more > button{background:none;border:0;padding:12px 0;color:var(--ink);
  font-family:var(--display);font-size:16px;font-weight:400;letter-spacing:.02em;
  cursor:pointer;display:flex;align-items:center;gap:9px}
.more > button::after{content:"+";color:var(--faint);font-size:15px;transition:transform .2s}
.more > button.open::after{content:"–"}
.answer{display:none;margin-top:2px;max-width:40ch;font-size:14.5px;line-height:1.62;color:var(--dim)}
.answer.open{display:block}
.answer .safety{margin-top:12px;color:rgba(250,250,255,.72);font-size:13.5px;
  border-left:2px solid rgba(255,176,32,.6);padding-left:12px}
.answer dl{display:none;grid-template-columns:auto 1fr;gap:5px 20px;margin-top:14px;
  font-family:var(--mono);font-size:11px;color:var(--dim)}
.answer dl.open{display:grid}
.answer dt{color:var(--faint);font-size:9.5px;letter-spacing:.12em;text-transform:uppercase;padding-top:3px}
.answer dd{color:var(--ink)}
.answer .det{background:none;border:0;color:var(--faint);font-family:var(--mono);font-size:10px;
  letter-spacing:.12em;text-transform:uppercase;cursor:pointer;margin-top:12px;
  border-bottom:1px solid rgba(250,250,255,.2);padding:0 0 2px}
.disc{display:grid;justify-items:center}
.disc canvas{width:100%;max-width:600px;height:auto;display:block}
.disc .label{margin-top:18px;font-family:var(--mono);font-size:10px;letter-spacing:.15em;
  color:var(--faint);text-transform:uppercase;text-align:center}
.foot{display:flex;align-items:center;gap:22px;padding:16px 46px 26px;
  border-top:1px solid var(--line);margin:0 46px}
.foot .dots{display:flex;gap:7px}
.foot .dots i{width:9px;height:9px;border-radius:50%;display:block}
.foot .txt{font-family:var(--mono);font-size:10px;letter-spacing:.12em;color:var(--faint);
  text-transform:uppercase}
.mobile-cta{display:none}
@media (max-width:940px){
  .hero{grid-template-columns:1fr;padding:18px 20px 0;gap:8px}
  .disc{order:-1}
  .disc canvas{max-width:300px}
  h1{font-size:38px}
  .nav{padding:18px 20px 0;gap:18px}
  .nav .links{display:none}
  .foot{margin:0 20px;padding:14px 0 18px;flex-wrap:wrap}
  .cta{display:none}
  .more{margin-bottom:96px}
  .mobile-cta{display:block;position:fixed;left:0;right:0;bottom:0;padding:12px 20px 22px;
    background:linear-gradient(transparent,rgba(13,13,36,.96) 40%);
    z-index:60;pointer-events:none}
  .mobile-cta button{pointer-events:auto;width:100%;height:58px;border:0;
    background:var(--ink);color:#0B0B1A;font-family:var(--display);font-size:15px;
    letter-spacing:.16em;text-transform:uppercase}
  .frame{inset:8px}
}
</style></head><body>
<div class="frame"></div>
<nav class="nav"><span class="brand">Fast Athlete</span>
  <span class="links"><a href="#">Check</a><a href="#">History</a></span>
  <button class="pill" id="go">Sign in</button></nav>

<section class="hero">
  <div>
    <p class="ask"><i></i>Checked this morning, 08:12</p>
    <h1>Is your<br><b>left calf</b> ready?</h1>

    <div class="verdict"><i></i>This morning: <b>rest it</b></div>

    <p class="lead">Your left calf is still showing fatigue. Skip calf work today.
      Check again tomorrow before you load it.</p>

    <div class="cta"><button class="pill" id="go2">Check my calf · 30 sec</button>
      <span class="hint">Saved to your history</span></div>

    <div class="more">
      <button id="why">Why red?</button>
      <div class="answer" id="why-body">
        <p>During your 30-second hold the calf's signal got <b>41 % slower</b> and
          <b>45 % stronger</b>. A tired muscle does both. Green means neither moved
          much, amber means they started to, red means the calf has not recovered.</p>
        <p class="safety">Sharp pain is not fatigue. If it hurts, stop and get it
          looked at, whatever colour this shows.</p>
        <button class="det" id="det">The numbers behind it</button>
        <dl id="det-body">
          <dt>Muscle</dt><dd>Gastrocnemius medialis, left</dd>
          <dt>Frequency</dt><dd>19.5 → 11.5 Hz, −41 %</dd>
          <dt>Amplitude</dt><dd>+45 % RMS</dd>
          <dt>Method</dt><dd>Zhao et al. 2025</dd>
        </dl>
      </div>
    </div>
  </div>

  <div class="disc"><canvas id="disc" width="1200" height="1200"></canvas>
    <div class="label">your own recording · 30 s hold</div></div>
</section>

<div class="foot">
  <span class="txt">Last 11</span>
  <span class="dots" id="dots"></span>
  <span class="txt" id="foot-note">last today 08:12</span>
</div>

<div class="mobile-cta"><button id="go3">Check my calf · 30 sec</button></div>

<script>
const GRID = "__GRID__", COLS = 120, ROWS = 68;
const bytes = Uint8Array.from(atob(GRID), c => c.charCodeAt(0));
const mdf = __MDF__;
const VERDICT = '#FF5468', SCORE = 0.13;

const cv = document.getElementById('disc'), ctx = cv.getContext('2d');
const W = cv.width, cx = W / 2, cy = W / 2;
let hover = -1;
const FEW_HAIRS = window.innerWidth < 700;

function dominant(j){
  let best = 0, bi = 0;
  for (let i = 0; i < ROWS; i++) { const v = bytes[i * COLS + j]; if (v > best) { best = v; bi = i; } }
  return bi / (ROWS - 1);
}

function draw(){
  ctx.clearRect(0, 0, W, W);
  const R = 552;                      // the dial
  // three zones, so the ring is the traffic light bent into a circle
  const zones = [[0.00, 0.34, 'rgba(60,224,140,', 'READY'],
                 [0.34, 0.67, 'rgba(255,176,32,', 'EASY'],
                 [0.67, 1.00, 'rgba(255,84,104,', 'REST']];
  zones.forEach(([from, to, rgb, word], k) => {
    const a0 = -Math.PI / 2 + from * Math.PI * 2;
    const a1 = -Math.PI / 2 + to * Math.PI * 2;
    ctx.beginPath(); ctx.arc(cx, cy, R, a0, a1);
    ctx.strokeStyle = rgb + (k === 2 ? '.30)' : '.14)'); ctx.lineWidth = 14; ctx.stroke();
    const mid = (a0 + a1) / 2;
    ctx.save();
    ctx.translate(cx + Math.cos(mid) * (R + 40), cy + Math.sin(mid) * (R + 40));
    ctx.fillStyle = 'rgba(250,250,255,.42)';
    ctx.font = '500 15px "JetBrains Mono", monospace';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(word, 0, 0);
    ctx.restore();
  });
  // the reading: a needle at how spent the muscle is, not a progress bar
  const na = -Math.PI / 2 + (1 - SCORE) * Math.PI * 2;
  ctx.fillStyle = '#FAFAFF';
  ctx.beginPath();
  ctx.arc(cx + Math.cos(na) * R, cy + Math.sin(na) * R, 11, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = VERDICT;
  ctx.beginPath();
  ctx.arc(cx + Math.cos(na) * R, cy + Math.sin(na) * R, 5.5, 0, Math.PI * 2);
  ctx.fill();

  // faint concentric rules
  for (let r = 140; r <= 520; r += 40){
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(250,250,255,.045)'; ctx.lineWidth = 1.2; ctx.stroke();
  }
  // the recording, engraved: one hair per instant, radius = its frequency
  const step = FEW_HAIRS ? 2 : 1;
  for (let j = 0; j < COLS; j += step){
    const theta = (j / COLS) * Math.PI * 2 - Math.PI / 2;
    const f = dominant(j);
    const power = bytes[Math.round(f * (ROWS - 1)) * COLS + j] / 255;
    const r = 140 + f * 380;
    const len = 66 + power * 120;
    const inner = r - len * .45, outer = r + len * .55;
    const hot = hover === j;
    const g = ctx.createLinearGradient(cx + Math.cos(theta) * inner, cy + Math.sin(theta) * inner,
                                       cx + Math.cos(theta) * outer, cy + Math.sin(theta) * outer);
    g.addColorStop(0, 'rgba(250,250,255,.04)');
    g.addColorStop(.5, hot ? 'rgba(255,255,255,.98)' : 'rgba(250,250,255,' + (0.20 + power * 0.62).toFixed(3) + ')');
    g.addColorStop(1, 'rgba(250,250,255,.03)');
    ctx.beginPath();
    ctx.moveTo(cx + Math.cos(theta) * inner, cy + Math.sin(theta) * inner);
    ctx.lineTo(cx + Math.cos(theta) * outer, cy + Math.sin(theta) * outer);
    ctx.strokeStyle = g; ctx.lineWidth = hot ? 3 : 1.5; ctx.stroke();
  }
  // harmonic texture
  for (let j = 0; j < COLS * 2; j += step){
    const theta = (j / (COLS * 2)) * Math.PI * 2 - Math.PI / 2;
    const r = 140 + dominant(Math.floor(j / 2)) * 0.55 * 380;
    ctx.beginPath();
    ctx.moveTo(cx + Math.cos(theta) * (r - 22), cy + Math.sin(theta) * (r - 22));
    ctx.lineTo(cx + Math.cos(theta) * (r + 22), cy + Math.sin(theta) * (r + 22));
    ctx.strokeStyle = 'rgba(250,250,255,.09)'; ctx.lineWidth = 1; ctx.stroke();
  }
  // the line that carries the verdict
  ctx.beginPath();
  mdf.forEach((f, j) => {
    if (!f) return;
    const theta = (j / COLS) * Math.PI * 2 - Math.PI / 2;
    const r = 140 + ((f - 1) / 34) * 380;
    const x = cx + Math.cos(theta) * r, y = cy + Math.sin(theta) * r;
    j ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.strokeStyle = VERDICT; ctx.lineWidth = 3.2; ctx.stroke();

  // centre: subject, verdict, age -- in that order
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillStyle = 'rgba(250,250,255,.55)';
  ctx.font = '500 24px "JetBrains Mono", monospace';
  ctx.fillText('LEFT CALF', cx, cy - 78);
  ctx.fillStyle = '#FAFAFF';
  ctx.font = '300 96px "Barlow Condensed", sans-serif';
  ctx.fillText('REST', cx, cy + 8);
  ctx.fillStyle = 'rgba(250,250,255,.45)';
  ctx.font = '400 21px "JetBrains Mono", monospace';
  ctx.fillText('CHECKED 08:12', cx, cy + 78);
}
draw();

// eleven checks, newest on the right
const history = ['ok','ok','warn','ok','bad','warn','ok','ok','warn','bad','bad'];
const colours = {ok:'#3CE08C', warn:'#FFB020', bad:'#FF5468'};
const dots = document.getElementById('dots');
history.forEach(h => {
  const i = document.createElement('i');
  i.style.background = colours[h];
  i.style.opacity = h === 'bad' ? '1' : '.65';
  dots.appendChild(i);
});

cv.addEventListener('pointermove', (e) => {
  const b = cv.getBoundingClientRect();
  const dx = (e.clientX - b.left) / b.width * W - cx;
  const dy = (e.clientY - b.top) / b.height * W - cy;
  let a = Math.atan2(dy, dx) + Math.PI / 2;
  if (a < 0) a += Math.PI * 2;
  hover = Math.round((a / (Math.PI * 2)) * COLS) % COLS;
  draw();
});
cv.addEventListener('pointerleave', () => { hover = -1; draw(); });
const why = document.getElementById('why');
why.onclick = () => {
  document.getElementById('why-body').classList.toggle('open');
  why.classList.toggle('open');
};
document.getElementById('det').onclick = () => {
  document.getElementById('det-body').classList.toggle('open');
};
['go','go2','go3'].forEach(id => {
  const el = document.getElementById(id);
  if (el) el.onclick = () => alert('Sketch: this opens the 30-second check.');
});
</script></body></html>"""

A = A.replace('__GRID__', GRID).replace('__MDF__', json.dumps(strip['mdf']))

# ── concept B: the engraved plate ──────────────────────────────────────────
B = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FAST Athlete — plate</title>
""" + FONTS + "<style>" + BASE + """
body{background:var(--deep)}
.sheet{position:fixed;inset:14px;border:1px solid var(--line);overflow:hidden;
  background:var(--bg);display:flex;flex-direction:column}
.frame{display:none}
.nav{padding:26px 40px 0;gap:34px}
.plate{flex:1;display:grid;grid-template-columns:1fr 236px;gap:0;min-height:0}
.field{position:relative;padding:26px 0 0 40px;display:flex;align-items:center}
.field canvas{width:100%;height:100%;max-height:none;object-fit:contain;display:block}
.margin{border-left:1px solid var(--line);padding:26px 26px 0 22px;
  display:flex;flex-direction:column;gap:18px}
.margin h2{font-family:var(--display);font-weight:300;font-size:26px;line-height:1.05;
  letter-spacing:-.01em}
.margin dl{display:grid;gap:11px;margin-top:2px}
.margin dt{font-family:var(--mono);font-size:9.5px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--faint)}
.margin dd{font-family:var(--display);font-size:17px;font-weight:300;margin-top:2px}
.num{display:flex;align-items:flex-end;gap:26px;padding:2vh 40px 22px}
.num b{font-family:var(--display);font-weight:200;font-size:clamp(72px,9.4vw,140px);
  line-height:.78;letter-spacing:-.03em}
.num .cap{font-family:var(--mono);font-size:10.5px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--dim);padding-bottom:10px;width:190px;line-height:1.8}
.num .spacer{flex:1}
.rule{position:absolute;inset:0;pointer-events:none;
  background:repeating-linear-gradient(90deg,transparent 0 4.99%,rgba(250,250,255,.05) 5% 5.03%)}
.num .pill{align-self:center;margin-bottom:10px}
.plateNo{position:absolute;top:22px;right:22px;font-family:var(--mono);font-size:10px;
  letter-spacing:.2em;color:var(--faint)}
</style></head><body>
<div class="sheet">
  <nav class="nav"><span class="brand">Fast Athlete</span>
    <span class="links"><a href="#">Check</a><a href="#">Train</a><a href="#">History</a></span>
    <button class="pill" id="go">Start a check</button></nav>

  <div class="plate">
    <div class="field"><canvas id="plate" width="2400" height="1360"></canvas>
      <div class="plateNo">Plate I · Example recording</div></div>
    <aside class="margin">
      <h2>Gastrocnemius medialis</h2>
      <dl>
        <div><dt>Hold</dt><dd>30 s isometric</dd></div>
        <div><dt>Band</dt><dd>1–35 Hz, 34 bands</dd></div>
        <div><dt>Median fr.</dt><dd>19.5 → 11.5 Hz</dd></div>
        <div><dt>Amplitude</dt><dd>+45 % rms</dd></div>
        <div><dt>Threshold</dt><dd>−5 % crossed at 6 s</dd></div>
        <div><dt>Reading</dt><dd style="color:#FF5468">High fatigue</dd></div>
      </dl>
      <p class="mono" style="margin-top:auto;padding-bottom:20px;line-height:1.8">
        Engraved from the<br>recording's own<br>time-frequency plane.</p>
    </aside>
  </div>

  <div class="num">
    <b><span style="font-size:.5em;vertical-align:.28em">&minus;&thinsp;</span>41<span style="font-size:.42em;vertical-align:.45em">%</span></b>
    <span class="cap">Median frequency lost<br>across the hold</span>
    <span class="spacer"></span>
    <button class="pill" id="go2">Check your fatigue</button>
  </div>
</div>
<script>
const GRID = "__GRID__", COLS = 120, ROWS = 68;
const bytes = Uint8Array.from(atob(GRID), c => c.charCodeAt(0));
const cv = document.getElementById('plate'), ctx = cv.getContext('2d');
const W = cv.width, H = cv.height, pad = {l: 96, r: 40, t: 40, b: 64};
const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;

// engraved ground
ctx.fillStyle = '#14142B'; ctx.fillRect(0, 0, W, H);
for (let i = 0; i < ROWS; i++){
  const y = pad.t + (1 - i / (ROWS - 1)) * ph;
  for (let j = 0; j < COLS; j++){
    const v = bytes[i * COLS + j] / 255;
    if (v < 0.06) continue;
    const x = pad.l + (j / COLS) * pw;
    const w = (pw / COLS) * (0.5 + 0.5);
    ctx.fillStyle = 'rgba(250,250,255,' + (0.05 + v * 0.92).toFixed(3) + ')';
    ctx.fillRect(x, y - 1.2, w, 1.6 + v * 4.2);
  }
}
// rules
ctx.strokeStyle = 'rgba(250,250,255,.13)'; ctx.lineWidth = 1.5;
[10, 20, 30].forEach(hz => {
  const y = pad.t + (1 - (hz - 1) / 34) * ph;
  ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
  ctx.fillStyle = 'rgba(250,250,255,.4)';
  ctx.font = '500 22px "JetBrains Mono", monospace';
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  ctx.fillText(hz + ' Hz', pad.l - 18, y);
});
[0, 10, 20, 30].forEach(s => {
  const x = pad.l + (s / 30) * pw;
  ctx.fillStyle = 'rgba(250,250,255,.4)';
  ctx.font = '500 22px "JetBrains Mono", monospace';
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  ctx.fillText(s + ' s', x, pad.t + ph + 18);
});
// baseline window
ctx.fillStyle = 'rgba(0,0,242,.10)';
ctx.fillRect(pad.l, pad.t, (5 / 30) * pw, ph);
ctx.strokeStyle = 'rgba(0,0,242,.9)'; ctx.lineWidth = 3;
ctx.beginPath(); ctx.moveTo(pad.l + (5 / 30) * pw, pad.t); ctx.lineTo(pad.l + (5 / 30) * pw, pad.t + ph); ctx.stroke();
document.getElementById('go').onclick = () => alert('This is a sketch.');
document.getElementById('go2').onclick = () => alert('This is a sketch.');
</script></body></html>"""

B = B.replace('__GRID__', GRID)

# ── concept C: the contour plate ───────────────────────────────────────────
C = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FAST Athlete — contour</title>
""" + FONTS + "<style>" + BASE + """
body{background:var(--deep);overflow:hidden}
.frame{display:none}
.top{display:flex;align-items:flex-start;gap:30px;padding:26px 44px 0;position:relative;z-index:2}
.top .brand{font-family:var(--display);font-size:17px;letter-spacing:.22em;text-transform:uppercase}
.top .meta{margin-left:auto;display:flex;gap:44px}
.top .meta div{font-family:var(--mono);font-size:9.5px;letter-spacing:.16em;
  text-transform:uppercase;color:var(--faint)}
.top .meta div b{display:block;color:var(--ink);font-weight:400;margin-top:5px;letter-spacing:.08em}
.map{position:fixed;inset:0;display:grid;place-items:center}
.map canvas{width:100vw;max-height:78vh;height:auto;opacity:.95}
.hero{position:absolute;left:44px;bottom:44px;right:44px;display:flex;
  align-items:flex-end;gap:40px;pointer-events:none}
h1{font-family:var(--display);font-weight:200;font-size:clamp(46px,6.2vw,96px);
  line-height:.9;letter-spacing:-.02em}
h1 em{font-style:normal;color:var(--blue)}
.legend{margin-left:auto;text-align:right;pointer-events:auto}
.legend .row{display:flex;align-items:baseline;gap:14px;justify-content:flex-end}
.legend .row b{font-family:var(--display);font-weight:300;font-size:30px}
.legend .row span{font-family:var(--mono);font-size:9.5px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--faint);width:130px;text-align:left}
.legend button{margin-top:18px;background:transparent;color:var(--ink);
  border:1px solid rgba(250,250,255,.35);padding:12px 22px;cursor:pointer;
  font-family:var(--display);font-size:12.5px;letter-spacing:.16em;text-transform:uppercase}
.legend button:hover{background:var(--ink);color:#0B0B1A}
</style></head><body>
<header class="top"><span class="brand">Fast Athlete</span>
  <div class="meta">
    <div>Muscle<b>Gastrocnemius med.</b></div>
    <div>Hold<b>30 s · 1–35 Hz</b></div>
    <div>Reading<b style="color:#FF5468">High fatigue</b></div>
  </div></header>

<div class="map"><canvas id="map" width="2400" height="1300"></canvas></div>

<div class="hero">
  <h1>The shape<br>of <em>fatigue</em>.</h1>
  <div class="legend">
    <div class="row"><span>Median fr.</span><b>19.5 → 11.5 Hz</b></div>
    <div class="row"><span>Amplitude</span><b>+45 %</b></div>
    <div class="row"><span>Threshold</span><b>−5 % at 6 s</b></div>
    <button id="go">Check your fatigue</button>
  </div>
</div>

<script>
const GRID = "__GRID__", COLS = 120, ROWS = 68;
const bytes = Uint8Array.from(atob(GRID), c => c.charCodeAt(0));
const mdf = __MDF__;
const cv = document.getElementById('map'), ctx = cv.getContext('2d');
const W = cv.width, H = cv.height;
const pad = {l: 120, r: 120, t: 90, b: 150};
const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
// Two box passes, so the contours come out as lines rather than polygons.
let field = new Float32Array(COLS * ROWS);
for (let i = 0; i < ROWS; i++) for (let j = 0; j < COLS; j++) field[i * COLS + j] = bytes[i * COLS + j] / 255;
for (let pass = 0; pass < 3; pass++){
  const out = new Float32Array(field.length);
  for (let i = 0; i < ROWS; i++){
    for (let j = 0; j < COLS; j++){
      let sum = 0, n = 0;
      for (let di = -1; di <= 1; di++) for (let dj = -1; dj <= 1; dj++){
        const ii = i + di, jj = j + dj;
        if (ii < 0 || jj < 0 || ii >= ROWS || jj >= COLS) continue;
        sum += field[ii * COLS + jj]; n++;
      }
      out[i * COLS + j] = sum / n;
    }
  }
  field = out;
}
const at = (i, j) => field[i * COLS + j];

function contour(level, colour, width){
  ctx.beginPath();
  for (let i = 0; i < ROWS - 1; i++){
    for (let j = 0; j < COLS - 1; j++){
      const a = at(i, j), b = at(i + 1, j), c = at(i + 1, j + 1), d = at(i, j + 1);
      const pts = [];
      const cut = (v1, v2, x1, y1, x2, y2) => {
        const t = (level - v1) / (v2 - v1 || 1e-6);
        return [x1 + (x2 - x1) * t, y1 + (y2 - y1) * t];
      };
      const x = (jj) => pad.l + (jj / (COLS - 1)) * pw;
      const y = (ii) => pad.t + (1 - ii / (ROWS - 1)) * ph;
      if ((a < level) !== (b < level)) pts.push(cut(a, b, x(j), y(i), x(j), y(i + 1)));
      if ((b < level) !== (c < level)) pts.push(cut(b, c, x(j), y(i + 1), x(j + 1), y(i + 1)));
      if ((c < level) !== (d < level)) pts.push(cut(c, d, x(j + 1), y(i + 1), x(j + 1), y(i)));
      if ((d < level) !== (a < level)) pts.push(cut(d, a, x(j + 1), y(i), x(j), y(i)));
      if (pts.length >= 2){
        ctx.moveTo(pts[0][0], pts[0][1]);
        ctx.lineTo(pts[1][0], pts[1][1]);
      }
    }
  }
  ctx.strokeStyle = colour; ctx.lineWidth = width; ctx.stroke();
}

ctx.fillStyle = '#0D0D24'; ctx.fillRect(0, 0, W, H);
// ground rules
ctx.strokeStyle = 'rgba(250,250,255,.07)'; ctx.lineWidth = 1.5;
[10, 20, 30].forEach(hz => {
  const y = pad.t + (1 - (hz - 1) / 34) * ph;
  ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
});
// contours, every fourth one heavier
for (let k = 1; k <= 22; k++){
  const level = (k / 23) * 0.95;
  contour(level, k % 4 === 0 ? 'rgba(250,250,255,.42)' : 'rgba(250,250,255,.13)',
          k % 4 === 0 ? 2.2 : 1.4);
}
// the ridge that matters
ctx.beginPath();
mdf.forEach((f, j) => {
  if (!f) return;
  const x = pad.l + ((j + 0.5) / COLS) * pw;
  const y = pad.t + (1 - (f - 1) / 34) * ph;
  j ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
});
ctx.strokeStyle = 'rgba(0,0,242,.95)'; ctx.lineWidth = 4; ctx.stroke();
ctx.strokeStyle = 'rgba(140,170,255,.55)'; ctx.lineWidth = 1.4; ctx.stroke();
document.getElementById('go').onclick = () => alert('This is a sketch.');
</script></body></html>"""

C = C.replace('__GRID__', GRID).replace('__MDF__', json.dumps(strip['mdf']))

files = {'001-etched-disc': A, '002-engraved-plate': B, '003-contour-plate': C}
for name, html in files.items():
    d = os.path.join(ROOT, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'index.html'), 'w') as fh:
        fh.write(html)
    print(name, len(html) // 1024, 'KB')
