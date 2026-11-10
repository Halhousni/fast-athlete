# FAST — design decisions

Filter and Aggregate Synchrosqueezed Transform. This file records how the app
works now and why the parts were chosen. It is not a user guide; that is
README.md.

## The pipeline

`engine.py`, per channel.

1. `resample_to_200` decimates to 200 Hz with `scipy.signal.resample_poly`, using
   a rational approximation of `200 / fs`. It returns the true rate of the
   returned samples, so every frequency downstream is in real hertz. A recording
   at or below 200 Hz is left alone and keeps its own rate. Nothing is relabelled:
   the previous build set `fs = 200` whether or not the decimation had run, which
   put every frequency off by the ratio of the real rate to 200.
2. `transform` takes one wavelet synchrosqueezed transform of the whole channel:
   `wsstpy.wsst`, analytic Morlet, 32 voices per octave, `extend_signal=True`
   (MATLAB's default, which reduces edge effects in the first seconds where the
   baseline is taken).
3. The transform is masked: points whose power is below 0.9 times the mean power
   of the plane are set to zero. This suppresses the noise floor before any
   frequency statistic is computed.
4. `mean_instantaneous_frequency` computes, per time frame, the power-weighted
   mean frequency over 1–35 Hz.

The previous build built its matrix from 35 band-filtered transforms (1 Hz to
35 Hz in 1 Hz steps, each with its own SSQ-CWT) and summed them. The bands shaped
the spectrum; the transform already resolves frequency, so the band can be
selected on the frequency axis. On the bundled demo recording every channel gives
the same verdict from one transform, with scores within one point, at 1/6 of the
cost (33.1 s to 5.7 s for eight channels). The measurement is in ENGINE-NOTES.md
and `scripts/compare_transform.py` reproduces it.

## Fatigue classification

`classify` compares the median of the recording with the median of the first five
seconds of the same contraction:

```
NFI = (MIF_baseline - MIF_median) / MIF_baseline * 100
```

A positive NFI means the spectrum moved down during the recording, which is the
usual signature of fatigue. The baseline is the subject's own opening seconds, so
the method is within-subject and does not compare an athlete to a population.

`score_from_nfi` maps NFI to 0–100 and `status_from_score` maps the score to a
traffic light, in that order, so the number and the colour cannot disagree. The
overall verdict is the worst muscle's status (`overall`), with the mean score
reported separately: seven recovered muscles and one that needs rest is not a
recovered athlete.

Guards, all added because a missing guard produced a confident green:

- a recording under 15 s is grey, not scored;
- if fewer than half of the baseline frames survive the power mask, grey;
- if no frame survives, grey;
- one NaN sample no longer discards a channel: `ingest` interpolates gaps up to
  1 % of samples and 50 ms, and drops a channel only beyond that, with the reason.

**The thresholds are provisional.** 15 % and 35 % await clinical validation from
Prof Samit Chakrabarty at the University of Leeds. The results screen says so in
plain words rather than only here.

## Bowen LOW centroid

`bowen_centroid.py` implements the LOW 8–23 Hz centroid described in "FAST LOW
8–23 Hz Centroid Calculation" (Bowen, 22 July 2026). It runs on the same masked
transform through `run_bowen_pipeline` and returns a `BowenResult`: per time point
the LOW centroid, LOW mass, LOW fraction and LOW MDF with a validity mask, then
phase means split 60/40 into CON and ECC, early, middle and late centroids.

It is reported to the athlete but does not drive the traffic light. The file's own
notes say why: the spec assigns no universal fatigue direction to the centroid,
and any threshold on it would be this project's addition.

## Reading a recording

`ingest.load` handles what device exports actually contain.

- Sampling rate, in order of authority: a value the athlete supplies, a key-value
  line in the first 20 lines (`Sampling rate: 2000`, `fs=2048`), a time column
  (uniform steps, recognised by name or by being monotonic with sub-second steps).
  If none exists, `fs` is None and the page asks for it.
- Separators and decimals: every candidate separator is scored by how many numeric
  columns it yields, because a semicolon file with comma decimals also parses as
  one bad comma-separated column. `decimal=','` is used where the separator is
  not a comma.
- A units row under the header (`mV,mV,mV`) is detected by its column names and
  skipped.
- Channels that are not EMG are dropped: time, marker, trigger, sync, reference,
  event, frame, sample, accelerometer, gyroscope, IMU, force, torque, angle,
  magnetometer, ECG, respiration, and unit-only names.
- `.mat`: scalars are skipped (the previous build counted `fs` as a channel and
  truncated every channel to one sample), 2-D arrays are split along the time axis
  into channels, and `fs` is matched by exact key first.
- Channels are matched to muscles by full name first, then by abbreviation as a
  whole token or digit-anchored, so `VASTUS LATERALIS` cannot match `ST`.

## Storage and access

`store.py`.

- Profiles and results in SQLite, WAL mode, a connection per request, foreign keys
  on. The database is at `$FAST_DATA_DIR/fast.db`, a volume in the container.
- A PIN is stored as `scrypt(pin, per-profile salt)`. Verifying issues a random
  token stored as sha256, returned as an HttpOnly cookie, and required by
  `/history`, `/save` and `DELETE /profiles/{id}`. The previous build checked the
  PIN in the browser and left the routes open.
- Five wrong PINs per profile are throttled for fifteen minutes.
- Profiles from the previous build carry an unsalted `sha256('fast|' + pin)`. That
  hash is accepted once, then replaced with the salted one, so nobody is locked
  out by the change.
- `/save` recomputes the verdict from the stored result rather than trusting the
  client's score.
- Uploads are held in memory as float32, at most 20 sessions, expiring after 30
  minutes. They used to be pickled into a temporary directory that survived
  nothing and filled the container layer.

## Interface

`templates/index.html`, `static/app.css`, `static/app.js`.

The page was 1 630 lines inside a Python string. It is now three files, which
allows a strict Content-Security-Policy (`script-src 'self'`, no inline styles),
caching, linting and reviewable diffs. Nodes are built with `textContent`; no
user-supplied string reaches `innerHTML`, which closes the stored XSS that a
profile name or file name could trigger.

- Four tabs: Check, Train, History, Profile.
- Check: recording, then the rate if it is unknown, then muscles, then the result.
- Train: the call for today, capped by the worst muscle, with the muscles to watch,
  a seven-day strip and no invented data. It reads the saved check, or the one just
  run for someone who has not signed in.
- History: a trend only from real points, from two onwards.
- Profile: sign in, add, delete, and the version.

Visual system: three status colours carry the meaning and one ember accent is
reserved for actions, at measured contrast in both themes (the previous palette
put `#00E676` on white at 1.67:1). Six type sizes in two weights, sentence case,
a 4 px spacing grid. Removed: the gauge duplicated on two screens, the notification
bell with a fake badge, the example-result card, the video of the app inside the
app, and a hardcoded chart of M T W T F S at invented heights that was shown as
the athlete's trend.

The body map uses MuscleMapJS (MIT) unchanged from the vendored bundle, with the
local `highlightSide` patch. It is tinted with the status colours; unmeasured
regions use the surface line colour so the silhouette reads as one figure.

## Deployment

`docker-compose.yml` builds `signal-analyzer` from `current build`, publishes it on
`$FAST_BIND_IP:8501` (loopback by default, because published ports bypass UFW),
keeps the database on the `fast_data` volume, and allows 1 GB: one channel peaks
near 60 MB and eight near 450 MB, measured.

The image runs as a non-root user, has a healthcheck against `/healthz`, and takes
`FAST_BUILD` as a build argument so the running container names its commit.
