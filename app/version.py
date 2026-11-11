"""What is running.

`VERSION` is bumped by hand when the deployed behaviour changes. `BUILD` is the
commit the image was built from, so a running container can be traced back to a
line of history:

    FAST_BUILD=$(git rev-parse --short HEAD) docker compose up -d --build

An unset `FAST_BUILD` reads as 'local', which is what a development copy
shows.
"""
from __future__ import annotations

import os

VERSION = '0.02'

# 0.01  one file, one transform per channel, a PIN, a traffic light
# 0.02  the paper's measure, accounts, four chapters, the etched dial


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
