# FAST Athlete

Muscle fatigue from surface EMG. One recording, one muscle, one answer: ready,
go easy, or rest, with the two numbers that produced it.

![The home screen after a check](docs/screenshot.png)

FAST Athlete is a research prototype. The thresholds are provisional and it is
not a diagnostic device.

A running copy is at <https://fast.ptrehab.icu>.

## What it does

You hold a contraction for thirty seconds while an EMG amplifier records. You
upload the file and say which muscle each channel is. The app measures how the
signal changed between the start of the hold and the end, and scores it.

- **Ready** means neither measure moved far enough to call fatigue. Train as
  planned.
- **Go easy** means both moved a little. Fewer sets today.
- **Rest** means both moved a lot. Leave the muscle unloaded and check again
  tomorrow.

The recommendation reads your saved checks as well as the recording, so a muscle
that has read the same or worse for a few days gets told to rest longer. The
wording never claims a trend it cannot see: two checks are two checks.

## Quick start

```bash
git clone https://github.com/Halhousni/fast-athlete
cd fast-athlete
docker compose -f app/docker-compose.yml up --build
```

Open <http://127.0.0.1:8501>. The bundled `app/demo_recording.csv` is a synthetic
eight-channel recording with a known answer, so you can try the whole flow
without an amplifier. Its sampling rate is 1000 Hz.

Without Docker:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r app/requirements.txt pytest

cd app
FAST_DATA_DIR=/tmp/fast-data python3 -m uvicorn app:app --port 8501
```

## How it measures

Each channel is resampled to 200 Hz, split into 34 overlapping bands, and each
band gets its own synchrosqueezed transform. The band powers are summed onto one
grid. Median frequency and RMS are then tracked through the recording, and the
last five seconds are compared with the first five.

Fatigue shows up as a fall in median frequency and a rise in RMS. Both count, so
a frequency drop with no amplitude change is half the evidence and scores
accordingly. The full method, including the thresholds and what they mean, is in
[docs/method.md](docs/method.md).

The method follows Zhao, Charoonsrisawad, Ichiyama and Chakrabarty (2025).

## Accounts

An account is a username and a password. Passwords are hashed with scrypt at
n=2^15, with the cost and salt inside the stored string so it can be raised
later. A session is a random token whose sha256 the database holds, so a copy of
the database cannot be turned into working cookies. The cookie is HttpOnly,
SameSite=Lax, thirty days, and Secure when `FAST_COOKIE_SECURE=1`.

Five wrong attempts per account and address are throttled for fifteen minutes,
and twelve per account name, because behind a proxy every request looks like the
proxy and a single attacker could otherwise lock an athlete out. Sign-in gives
the same answer and the same timing for a missing account and a wrong password.
Nobody can list the accounts on a server. Deleting an account asks for the
password again.

There is no email on the server, so the reset path is the operator command. It
prompts on the terminal, which keeps the password out of shell history:

```bash
docker compose exec signal-analyzer python3 store.py passwd <username>
```

## Running it

`app/docker-compose.yml` builds one container, publishes it on loopback by
default, and keeps the database on a volume.

| variable | default | what it does |
|---|---|---|
| `FAST_BIND_IP` | `127.0.0.1` | address the container publishes on |
| `FAST_COOKIE_SECURE` | `0` | set to 1 behind TLS |
| `FAST_DATA_DIR` | `/data` | where `fast.db` lives |
| `FAST_BUILD` | `local` | commit stamped into the image |
| `FAST_FA_WORKERS` | `6` | threads for the band transforms |

The port is bound to loopback on purpose. Docker's published ports bypass UFW,
so binding every interface puts the app on the internet around whatever proxy
terminates TLS.

Uploads are held in memory and expire after thirty minutes; only a saved check
is written to the database. One analysis takes about half a minute for eight
channels, and the app runs two at a time.

## Development

```
app/app.py            HTTP routes, page, security headers
app/engine.py         the measurement
app/ingest.py         CSV, TXT and MAT parsing, sampling rate, channel matching
app/store.py          SQLite accounts, results, sessions
app/version.py        what is running
app/static/           app.css, app.js, theme.js, icon.svg, vendor/musclemap
app/templates/        index.html
docs/method.md        the measurement in full
scripts/              regenerating the demo data and the spectrogram strip
tests/                pytest
```

```bash
python3 -m pytest tests -q
```

The running version is at `/healthz` and at the foot of the You screen.

## Limits

- The thresholds are provisional and are being validated.
- A result compares one muscle with itself, within one recording. Electrode
  placement matters more than anything else you do.
- Sharp pain is not fatigue. If it hurts, stop.
- Nothing here replaces a clinician.

## Licence

MIT. See [LICENSE](LICENSE). Third-party notices are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
