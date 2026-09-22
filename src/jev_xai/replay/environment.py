"""Runtime fingerprint via importlib.metadata, not ``pip freeze``."""

from __future__ import annotations

import os
import platform
import sys
from importlib.metadata import distributions

from jev_xai.canonical import content_hash
from jev_xai.evidence.schema import RuntimeInfo


def runtime_fingerprint(seed: int) -> RuntimeInfo:
    packages: list[str] = []
    for dist in distributions():
        name = dist.metadata["Name"]
        if name:
            packages.append(f"{name}=={dist.version}")
    packages.sort()
    return RuntimeInfo(
        python_version=sys.version.split()[0],
        package_lock_hash=content_hash(packages),
        seed=seed,
        container_digest=os.environ.get("CONTAINER_DIGEST"),
        hardware=platform.machine() or None,
    )
