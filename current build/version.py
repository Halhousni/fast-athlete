"""What is running.

`VERSION` is bumped by hand when the deployed behaviour changes. `BUILD` is set
by the deploy step so a running container can be traced back to a commit:

    FAST_BUILD=$(git rev-parse --short HEAD) docker compose up -d --build

If `FAST_BUILD` is not set the value is 'local', which is what a development
copy shows.
"""
from __future__ import annotations

import os

VERSION = '0.3.0'

# 0.1.0  one HTML string, ssqueezepy, 35 band transforms per channel
# 0.2.0  split modules, wsstpy, one transform, tests, pinned sampling rate
# 0.3.0  rebuilt interface, four tabs, server-side PIN sessions (this one)


def build_id() -> str:
    return os.environ.get('FAST_BUILD', 'local')


def info(engine_version: str | None = None) -> dict:
    data = {'version': VERSION, 'build': build_id()}
    if engine_version:
        data['engine'] = engine_version
    return data


def label(engine_version: str | None = None) -> str:
    parts = [f'FAST {VERSION}']
    if engine_version:
        parts.append(engine_version)
    build = build_id()
    if build and build != 'local':
        parts.append(build)
    return ' · '.join(parts)
