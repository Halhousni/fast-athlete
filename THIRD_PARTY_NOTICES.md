# Third-party notices

FAST Athlete is MIT licensed. It uses the following.

## MuscleMapJS

`app/static/vendor/musclemap.min.js`

MIT. <https://github.com/abdofallah/MuscleMapJS>, a port of the MuscleMap
SwiftUI SDK by Melih Colpan.

Bundled unmodified except for one added method, `highlightSide(muscle, side,
colour, opacity)`, which tints one side of the body so a single leg can be
highlighted. The bundle is served as a static file and used as-is.

## wsstpy

A pip dependency, pinned in `app/requirements.txt`.

MIT. <https://github.com/Halhousni/wsstpy>. Wavelet synchrosqueezing in NumPy and
SciPy that reproduces MATLAB's `wsst`. The version in use is reported by
`/healthz` and stored with every result.

## The method

Daubechies, Lu and Wu, *Applied and Computational Harmonic Analysis* 30(2), 2011.
Thakur, Brevdo, Fuckar and Wu, *Signal Processing* 93(5), 2013.

Zhao, Charoonsrisawad, Ichiyama and Chakrabarty, 2025, for the fatigue measure
itself: the change in median frequency and in RMS across a hold.
