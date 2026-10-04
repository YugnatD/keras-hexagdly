from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from .layers import (
    AvgPool2d,
    Conv2d,
    Conv2d_CustomKernel,
    Conv3d,
    Conv3d_CustomKernel,
    MaxPool2d,
    MaxPool3d,
    ring_maps_2d,
)

__all__ = [
    "AvgPool2d",
    "Conv2d",
    "Conv2d_CustomKernel",
    "Conv3d",
    "Conv3d_CustomKernel",
    "MaxPool2d",
    "MaxPool3d",
    "ring_maps_2d",
]

# Read from the installed package's own metadata (set from pyproject.toml's
# `version` at build time) rather than a second hardcoded string here -- a
# manually-duplicated constant drifted stale across three releases in a row
# (stuck at "0.3.0" through 0.3.1, 0.4.0, and 0.4.1) before this fix.
try:
    __version__ = _version("keras-hexagdly")
except PackageNotFoundError:
    # Running from source without an installed/editable package (e.g. a
    # bare checkout on sys.path with no `pip install`).
    __version__ = "unknown"
