# Third-party notices

FAST bundles or depends on the following third-party work.

## MuscleMapJS

The interactive muscle map is [MuscleMapJS](https://github.com/abdofallah/MuscleMapJS)
by Abdofallah, a TypeScript port of the MuscleMap SwiftUI SDK by Melih Colpan.
It is licensed under the MIT License and is bundled into `current build/app.py`
as a minified IIFE build produced with esbuild. The local copy carries one patch,
a `highlightSide(muscle, side, color)` method used to colour a single leg.

```
MIT License

Copyright (c) 2026 Abdofallah

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Python dependencies

Declared in `current build/requirements.txt`. All are permissively licensed.

| Package | Licence |
|---------|---------|
| fastapi | MIT |
| uvicorn | BSD-3-Clause |
| python-multipart | Apache-2.0 |
| numpy | BSD-3-Clause |
| scipy | BSD-3-Clause |
| ssqueezepy | MIT |
| pandas | BSD-3-Clause |

