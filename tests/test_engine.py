"""Engine tests: the rate is real hertz, and the colour always matches the score."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'current build'))

import engine  # noqa: E402


def tone(freq_hz: float, fs: float, seconds: float = 30.0) -> np.ndarray:
    t = np.arange(0, seconds, 1.0 / fs)
    return np.sin(2 * np.pi * freq_hz * t)


@pytest.mark.parametrize('fs_true', [200.0, 250.0, 300.0, 1000.0, 2000.0])
def test_reported_frequency_is_the_true_frequency(fs_true):
    """A 20 Hz tone must read 20 Hz whatever the device rate was.

    This is the regression test for the old `_run_fast`, which labelled any
    recording between 200 and 400 Hz as 200 Hz without resampling it: a 20 Hz
    tone read 15.9 Hz at 250 Hz and 13.4 Hz at 300 Hz.
    """
    sig = tone(20.0, fs_true)
    resampled, fs = engine.resample_to_200(sig, fs_true)
    Tx, freqs = engine.transform(resampled, fs)
    mif = engine.mean_instantaneous_frequency(Tx, freqs)
    measured = float(np.nanmedian(mif))
    assert measured == pytest.approx(20.0, rel=0.03), f'at {fs_true} Hz got {measured:.2f}'


@pytest.mark.parametrize('fs_true', [250.0, 300.0, 1000.0, 2000.0])
def test_resample_rate_is_truthful(fs_true):
    _, fs = engine.resample_to_200(tone(20.0, fs_true), fs_true)
    assert fs == pytest.approx(200.0, rel=0.001)


def test_low_rate_recording_keeps_its_rate():
    sig = tone(20.0, 150.0)
    resampled, fs = engine.resample_to_200(sig, 150.0)
    assert fs == 150.0
    assert len(resampled) == len(sig)


@pytest.mark.parametrize('nfi', [-10, 0, 7, 14.9, 15, 20, 34.9, 35, 40, 55, 80])
def test_score_and_status_cannot_disagree(nfi):
    score = engine.score_from_nfi(nfi)
    status = engine.status_from_score(score)
    assert 0 <= score <= 100
    assert (status == 'green') == (score >= 70)
    assert (status == 'amber') == (40 <= score < 70)
    assert (status == 'red') == (score < 40)


def test_threshold_edges():
    assert engine.status_from_score(engine.score_from_nfi(14.9)) == 'green'
    assert engine.status_from_score(engine.score_from_nfi(15.1)) == 'amber'
    assert engine.status_from_score(engine.score_from_nfi(34.9)) == 'amber'
    assert engine.status_from_score(engine.score_from_nfi(35.1)) == 'red'


def test_short_recording_is_grey_not_green():
    sig = tone(20.0, 200.0, seconds=3.0)
    Tx, freqs = engine.transform(sig, 200.0)
    result = engine.classify(Tx, freqs, 200.0)
    assert result['status'] == 'grey'
    assert 'at least' in result['tip']


def test_flat_signal_is_not_scored():
    sig = np.zeros(200 * 30)
    with pytest.raises(engine.EngineError):
        engine.analyse_channel(sig, 200.0, 'VL')


def test_overall_is_capped_by_the_worst_muscle():
    results = [dict(id='VL', status='green', score=100),
               dict(id='VM', status='green', score=98),
               dict(id='BF', status='red', score=12)]
    verdict = engine.overall(results)
    assert verdict['status'] == 'red'
    assert verdict['worst'] == 'BF'
    assert verdict['mean_score'] == 70


def test_overall_ignores_grey():
    results = [dict(id='VL', status='grey', score=0),
               dict(id='VM', status='amber', score=55)]
    verdict = engine.overall(results)
    assert verdict['status'] == 'amber'
    assert verdict['mean_score'] == 55


def test_overall_all_grey_is_grey():
    verdict = engine.overall([dict(id='VL', status='grey', score=0)])
    assert verdict['status'] == 'grey'
    assert verdict['mean_score'] is None


def test_missing_samples_are_rejected_before_the_transform():
    sig = tone(20.0, 200.0)
    sig[10] = np.nan
    with pytest.raises(engine.EngineError):
        engine.analyse_channel(sig, 200.0, 'VL')


def test_analyse_channel_reports_mdf_and_rms():
    """The indices the method reads: median frequency and amplitude per epoch."""
    sig = tone(20.0, 1000.0, seconds=45.0)
    result = engine.analyse_channel(sig, 1000.0, 'VL')
    assert result['analysed_seconds'] == pytest.approx(30.0, abs=0.1)
    # A 20 Hz tone has its median frequency at 20 Hz.
    assert result['mdf_start'] == pytest.approx(20.0, abs=2.5)
    assert result['delta_mdf'] == pytest.approx(0.0, abs=5.0)
    assert result['delta_rms'] == pytest.approx(0.0, abs=10.0)
    assert result['level'] == 'low'
    # The LOW centroid is not computed any more: it carries no fatigue direction.
    assert 'centroid' not in result


# ── the verdict itself ───────────────────────────────────────────────────────
# Without these, swapping `mif_baseline - mif_median` for the reverse would turn
# every fatigued muscle green and the suite would still pass.

def stationary(freq_hz: float, fs: float, seconds: float = 30.0) -> np.ndarray:
    return tone(freq_hz, fs, seconds)


def slowing(fs: float = 200.0, seconds: float = 30.0,
            start_hz: float = 25.0, end_hz: float = 15.0) -> np.ndarray:
    """A contraction whose spectral content slides down: what fatigue looks like."""
    n = int(fs * seconds)
    t = np.arange(n) / fs
    freq = np.linspace(start_hz, end_hz, n)
    phase = 2 * np.pi * np.cumsum(freq) / fs
    return np.sin(phase)


def test_a_steady_contraction_is_recovered():
    result = engine.analyse_channel(stationary(20.0, 200.0), 200.0, 'VL')
    assert result['nfi'] == pytest.approx(0.0, abs=5.0)
    assert result['status'] == 'green'
    assert result['score'] >= 70


def test_a_slowing_contraction_is_not_green():
    result = engine.analyse_channel(slowing(), 200.0, 'VL')
    assert result['nfi'] > 10, f"NFI came out {result['nfi']}"
    assert result['status'] in ('amber', 'red')


def test_a_steep_slowdown_is_red():
    """A contraction that drops quickly and stays low.

    The statistic compares the median of the recording with the median of the
    first five seconds, so a slow linear sweep only reaches about a third of its
    drop (the median of a ramp is its midpoint). A step-like loss of frequency
    is what reads as severe fatigue.
    """
    fs, seconds = 200.0, 30.0
    n = int(fs * seconds)
    t = np.arange(n) / fs
    freq = 12.0 + 23.0 * np.exp(-t / 2.0)
    sig = np.sin(2 * np.pi * np.cumsum(freq) / fs)
    result = engine.analyse_channel(sig, fs, 'VL')
    assert result['status'] == 'red', f"NFI came out {result['nfi']}"
    assert result['score'] < 40


def test_the_bundled_demo_gives_the_designed_verdict():
    """The demo was built with a known amount of fatigue in four muscles."""
    import ingest
    path = os.path.join(os.path.dirname(__file__), '..', 'current build',
                        'demo_recording.csv')
    # The demo has no time column, so the rate is supplied, as the page does.
    parsed = ingest.load(open(path, 'rb').read(), 'demo_recording.csv', fs=1000.0)
    got = {}
    for channel in parsed.channels:
        if channel['drop_reason']:
            continue
        got[channel['name']] = engine.analyse_channel(channel['data'], parsed.fs,
                                                      channel['name'])
    expected = {'VL': 'green', 'VM': 'green', 'RF': 'green', 'TA': 'green',
                'BF': 'amber', 'ST': 'amber', 'GM': 'red', 'GL': 'red'}
    assert {k: v['status'] for k, v in got.items()} == expected
    # ... and the classes behind the colours.
    levels = {k: v['level'] for k, v in got.items()}
    assert levels == {'VL': 'low', 'VM': 'low', 'RF': 'low', 'TA': 'low',
                      'BF': 'medium', 'ST': 'medium', 'GM': 'high', 'GL': 'high'}
    # Fatigued channels lost frequency and gained amplitude, as the method expects.
    for muscle in ('BF', 'GM'):
        assert got[muscle]['delta_mdf'] < -5
        assert got[muscle]['delta_rms'] > 10
