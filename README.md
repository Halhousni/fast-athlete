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

The running version is at `/healthz` and at the foot of the You screen. Stamp the commit
into the image with `FAST_BUILD=$(git rev-parse --short HEAD)`.

## Accounts and privacy

An account is a username and a password. Passwords are hashed with `scrypt` at
n=2^15, and the cost and the salt travel inside the stored string so the cost can
be raised later without locking anyone out. Signing in mints a random session
token whose sha256 the database holds, so a copy of `fast.db` cannot be turned
into a set of working cookies. The cookie is HttpOnly, SameSite=Lax, Secure
whenever `FAST_COOKIE_SECURE=1`, and lasts thirty days; its expiry is enforced
server-side, not only in the browser.

Five wrong attempts per account and address are throttled for fifteen minutes,
and twelve per account name, because behind the proxy every request can look like
it came from the proxy: the pair alone would let one person lock an athlete out
for everyone. Sign-in answers "Wrong username or password" for both a missing
account and a wrong password, and runs a dummy hash in the first case so the two
take the same time.

Nobody can list the accounts on this server. Deleting an account asks for the
password again, then removes the account, its results and every session.

Accounts created before this build have no password. The operator sets one from
the container, prompted on the terminal, so no credential passes through a shell
argument, a file or a chat message:

    docker compose exec signal-analyzer python3 store.py passwd hilal

There is no email on this server, so that command is also the reset path. It
signs out every existing session for the account.

- Say so plainly if it matters: the app is a research prototype, its thresholds
  are provisional, and an account here guards saved readings, not medical data.
