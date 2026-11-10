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


def test_analyse_channel_returns_centroid_and_window():
    sig = tone(20.0, 1000.0, seconds=45.0)
    result = engine.analyse_channel(sig, 1000.0, 'VL')
    assert result['analysed_seconds'] == pytest.approx(30.0, abs=0.1)
    assert result['centroid'] is not None
    assert 8 <= result['centroid'] <= 23
