# How FAST measures fatigue

One recording of one muscle, from the start of a held contraction to its end. The
question is whether that muscle tires early. FAST answers with two changes and
turns them into a score.

The method follows Zhao, Charoonsrisawad, Ichiyama and Chakrabarty (2025), *A
non-invasive, quantitative method for detection of muscle fatigue and neural
mechanism of compensation during muscle fatigue* (Cambridge Open Engage,
doi:10.33774/coe-2025-j3v1z): as a muscle fatigues, the median frequency of its
surface EMG falls and its amplitude rises. Neither alone is enough. A frequency drop with no amplitude change can be
a change in electrode contact; amplitude rising without a frequency drop can be
more motor units firing.

## The pipeline

`app/engine.py`, per channel.

1. **Resample.** The channel is decimated to 200 Hz with a rational polyphase
   filter. The rate comes from the athlete, a header line, or a time column in
   the file. FAST refuses to guess, because every frequency below is reported in
   real hertz.
2. **Filter and aggregate.** The channel is split into 34 overlapping bands
   between 1 and 35 Hz. Each band gets its own synchrosqueezed transform, at
   eight times its own top edge, so a 1 to 3 Hz band is transformed at 24 Hz and
   a 30 Hz band at 240 Hz. The band powers are summed back onto one 0.5 Hz grid.
   Bands run in parallel.
3. **Mask.** Time and frequency points below 0.9 times the mean power of the
   plane are zeroed, which removes the noise floor before any frequency statistic
   is computed.
4. **Two series.** Median frequency and RMS are both computed in 0.5 s epochs
   over the whole recording.
5. **Two changes.** The median of the first 5 s is compared with the median of
   the last 5 s:

   ```
   change in MDF = (MDF_end - MDF_start) / MDF_start × 100
   change in RMS = (RMS_end - RMS_start) / RMS_start × 100
   ```

6. **One number.** `r` is how far the pair has moved past its two limits, where
   the limits are a 10 % rise in RMS and a 5 % fall in MDF:

   ```
   r = max(0, (ΔRMS / 10 − ΔMDF / 5) / 2)
   ```

   `r` becomes a score from 0 to 100, and the score becomes the traffic light.
   The status is derived from the score in one place, so the two can never
   disagree.

| r | score | status | what it means |
|---|---|---|---|
| 0 to 1 | 100 to 70 | ready | one or both measures are at their limit, not past it |
| 1 to 2 | 70 to 40 | go easy | both are past their limit |
| 2 to 4 | 40 to 0 | rest | both are well past it, and the score floors at zero |

The verdict for a file is its worst muscle. A 30 s window is analysed and the
screen says so when the recording is longer. A recording under 15 s is not
scored.

## What is also reported

Mean instantaneous frequency and the NFI it gives are still computed and stored.
They are shown in the details, and they do not decide the verdict: the NFI
compares a whole recording with its own start, so a slowly tiring muscle can
score the same as a noisy one.

The trend slopes of both series are reported in percent per second. They describe
the shape of the recording and are not used as pass or fail.

## What FAST does not claim

- The thresholds are provisional and are being validated. A result here is not a
  diagnosis.
- The comparison is within one recording. Two recordings taken with different
  electrode placement are not directly comparable, and FAST does not pretend
  otherwise.
- Sharp pain is not fatigue. The app says so on screen.

The engine version that produced a result is stored with it, and `/healthz`
reports the version that is running.
