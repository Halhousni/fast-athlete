# FAST

Muscle fatigue from surface EMG. Upload a recording, pick the muscles, get a
traffic light for each one and a call for today.

FAST is a research prototype. It is not a diagnostic device, and the thresholds
it uses are still being validated.

## How it measures

1. Each channel is resampled to 200 Hz with a polyphase filter. The rate comes
   from the athlete, from a header line in the file, or from a time column. FAST
   refuses to guess.
2. One wavelet synchrosqueezed transform of the whole channel (`wsstpy`, analytic
   Morlet, 32 voices per octave, signal extension on).
3. Points whose power is below 0.9 times the mean power of the plane are zeroed,
   which suppresses the noise floor.
4. Per time frame, the power-weighted mean instantaneous frequency over 1–35 Hz.
5. The median of the whole recording is compared with the median of the first
   five seconds of the same contraction:

   ```
   NFI = (MIF_baseline - MIF_median) / MIF_baseline × 100
   ```

6. NFI becomes a score, and the score becomes the traffic light:

   | NFI | status | score |
   |---|---|---|
   | under 15 % | recovered | 70–100 |
   | 15–35 % | some fatigue | 40–69 |
   | over 35 % | needs rest | 0–39 |

   The overall verdict is the worst muscle, not the mean. A 30-second window is
   analysed, and the screen says so when the recording is longer. A recording
   under 15 seconds is not scored.

`bowen_centroid.py` also reports the LOW 8–23 Hz centroid for each muscle. It is
shown, but it does not drive the traffic light.

## Layout

```
app.py            HTTP routes, page, security headers
engine.py         the method: resample, transform, classify, verdict
ingest.py         file parsing: CSV, TXT, MAT, sampling rate, channel matching
store.py          SQLite profiles and results, PIN hashing, session tokens
version.py        what is running
bowen_centroid.py LOW 8–23 Hz centroid
templates/        index.html
static/           app.css, app.js, theme.js, icon.svg, vendor/musclemap.min.js
tests/            pytest: engine and ingest
scripts/          compare_transform.py, the aggregate-vs-single evidence
ENGINE-NOTES.md   why the method runs one transform, with the measurements
```

## Run locally

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r "current build/requirements.txt"
pip install pytest

FAST_DATA_DIR=/tmp/fast-data python3 -m uvicorn app:app --port 8501
```

with `cd "current build"` first, or set `PYTHONPATH`. Then open
http://127.0.0.1:8501.

Tests:

```bash
python3 -m pytest tests -q
```

## Deploy

```bash
cd "current build"
FAST_BIND_IP=127.0.0.1 docker compose up -d --build
```

`FAST_BIND_IP` is the address the container publishes on. The default is
loopback: Docker's published ports bypass UFW, so binding every interface puts
the app on the internet directly, around whatever proxy terminates TLS. Set it to
the address your proxy reaches.

The database lives on the `fast_data` volume at `/data`. Uploads are held in
memory only and expire after 30 minutes.

The running version is at `/healthz` and on the Profile screen. Stamp the commit
into the image with `FAST_BUILD=$(git rev-parse --short HEAD)`.

## Profiles and privacy

A profile PIN gates a session token, and the token is required by `/history`,
`/save` and deleting a profile. PINs are stored with `scrypt` and a per-profile
salt, five wrong attempts per profile are throttled for fifteen minutes, and
profiles created by earlier builds (unsalted hash) are accepted once and
upgraded.

The profile list is visible to anyone who can reach the page, because choosing a
profile is how you sign in. Names are the only thing it exposes.

## Limits

- One analysis at a time is allowed two in parallel; a channel costs about
  0.7 s, and eight about 6 s.
- Uploads are capped at 25 MB.
- The interface has no accounts, no email, and no recovery: a forgotten PIN means
  a new profile. Say so if that matters for your use.
