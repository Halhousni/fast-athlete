# FAST Athlete

FAST Athlete reads a surface EMG recording of one muscle and turns it into a
recommendation for the day: train as planned, go easy, or rest.

![The demo recording, analysed: the calf comes out at rest](docs/screenshot.png)

This is a research prototype. The thresholds are provisional and it is not a
diagnostic device. A running copy is at <https://fast.ptrehab.icu>.

## What it does

You hold a contraction for thirty seconds while an EMG amplifier records. You
upload the file and confirm which muscle each channel is. FAST compares the end
of the hold with its start, using two measures: the median frequency of the
signal and its amplitude (RMS).

- **Ready.** Neither measure moved past its limit. Train as planned.
- **Go easy.** Both moved past their limits. Fewer sets today.
- **Rest.** Both moved well past them. Leave the muscle unloaded and check again
  tomorrow.

Both measures count. A fall in median frequency on its own is half the evidence,
so it has to be twice as large to reach the same verdict.

The advice also takes your saved checks into account. When the previous checks of
that muscle read the same or worse, the recommendation is to rest for longer.

## Limits

- The thresholds are provisional and are being validated.
- A result compares one muscle with itself inside one recording. Comparing today
  with last week assumes the electrodes went on the same way.
- The measures were characterised on sustained holds. Use it for those.
- Sharp pain is not fatigue. If it hurts, stop.

## Quick start

```bash
git clone https://github.com/Halhousni/fast-athlete
cd fast-athlete
docker compose -f app/docker-compose.yml up --build
```

Open <http://127.0.0.1:8501> and upload `app/demo_recording.csv`. It is a
synthetic eight-channel recording with a known answer, so you can try the whole
flow without an amplifier: **VL, VM, RF and TA should read ready, BF and ST go
easy, GM and GL rest.** The app asks for the sampling rate because the file has
no header; it is 1000 Hz.

Without Docker, with Python 3.12 and `pytest`:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r app/requirements.txt pytest

cd app
FAST_DATA_DIR=/tmp/fast-data python3 -m uvicorn app:app --port 8501
```

### The file

CSV, TXT or MAT. One column per channel, one row per sample, an optional header
row and an optional time column. Column names are matched against the muscles
the app knows, and you confirm the mapping before the analysis.

The sampling rate comes from what you type, from a `fs=`, `rate=` or `fs:` line
in the file, or from a time column. If none of those exist, the app asks. It
never guesses.

Short gaps are interpolated, up to 1% of the samples and never longer than 50 ms.
Accelerometer, gyroscope, force, angle, marker and trigger channels are dropped
rather than offered for analysis.

## How it measures

Each channel is resampled to 200 Hz, split into 34 overlapping bands, and each
band gets its own synchrosqueezed transform. The band powers are summed onto one
grid. Median frequency and RMS are then tracked through the recording, and the
last five seconds are compared with the first five.

The full method, with the formula and the thresholds, is in
[docs/method.md](docs/method.md).

Zhao, Charoonsrisawad, Ichiyama and Chakrabarty (2025), *A non-invasive,
quantitative method for detection of muscle fatigue and neural mechanism of
compensation during muscle fatigue*, Cambridge Open Engage.
[doi:10.33774/coe-2025-j3v1z](https://doi.org/10.33774/coe-2025-j3v1z)

## Accounts

You can use the app without one. An account keeps your checks so the advice can
compare this week with the last.

An account is a username and a password, held on the server that runs the app.
There is no account directory to browse and nothing is sent to another service.
To leave, delete the account from the You screen: it asks for your password and
removes your saved checks with it.

There is no email on the server, so a forgotten password is reset by whoever runs
it:

```bash
docker compose -f app/docker-compose.yml exec signal-analyzer \
  python3 store.py passwd <username>
```

## Running it

`app/docker-compose.yml` builds one container and keeps the database on a volume.

| variable | default | what it does |
|---|---|---|
| `FAST_BIND_IP` | `127.0.0.1` | address the container publishes on |
| `FAST_COOKIE_SECURE` | `0` | set to 1 behind TLS |
| `FAST_DATA_DIR` | `/data` | where `fast.db` lives |
| `FAST_BUILD` | `local` | commit stamped into the image |
| `FAST_FA_WORKERS` | `6` | threads for the band transforms |

It publishes on loopback by default. Published Docker ports bypass UFW, so put a
proxy that terminates TLS in front of it rather than exposing the container.

Eight channels take about half a minute and the app runs two analyses at a time.
Uploads are held in memory and expire after thirty minutes; only a saved check is
written to the database.

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

Run the tests from the repository root:

```bash
python3 -m pytest tests -q
```

The running version is at `/healthz` and at the foot of the You screen.

## Licence

MIT. See [LICENSE](LICENSE). Third-party notices are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
