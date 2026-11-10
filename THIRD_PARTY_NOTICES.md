# Third-party notices

## MuscleMapJS

`current build/static/vendor/musclemap.min.js`

MIT. https://github.com/abdofallah/MuscleMapJS — a port of the MuscleMap SwiftUI
SDK by Melih Colpan.

Bundled unmodified except for one added method, `highlightSide(muscle, side,
colour, opacity)`, which tints one side of the body so a single leg can be shown.
The bundle is served as a static file and used as-is.

## wsstpy

`pip` dependency, pinned in `requirements.txt`.

MIT. https://github.com/Halhousni/wsstpy — wavelet synchrosqueezing in NumPy with
the same numbers as MATLAB's Wavelet Toolbox. The transform FAST computes is
`wsstpy.wsst`; the version in use is reported by `/healthz`.

## Algorithm

Daubechies, Lu and Wu, *Appl. Comput. Harmon. Anal.* 30(2), 2011. Thakur, Brevdo,
Fuckar and Wu, *Signal Processing* 93(5), 2013.

The LOW 8–23 Hz centroid follows "FAST LOW 8–23 Hz Centroid Calculation" (Bowen,
22 July 2026); `bowen_centroid.py` records where it may differ from the MATLAB
implementation.

## Removed in this build

Earlier builds loaded Inter from Google Fonts and drew the notification bell from
Font Awesome. Both are gone: the interface uses the system font stack and the
icons are drawn in the page, so there is no third-party request at load time and
no external IP is disclosed to a font or icon CDN.
