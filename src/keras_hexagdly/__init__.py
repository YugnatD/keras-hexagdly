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


_QUANTIZED_NAMES = (
    "QAvgPool2d",
    "QConv2d",
    "QConv2d_CustomKernel",
    "QConv3d",
    "QConv3d_CustomKernel",
    "QMaxPool2d",
    "QMaxPool3d",
    "freeze",
)


def __getattr__(name):
    # The quantized layers and freeze need the optional dependency hgq2 (pip install
    # keras-hexagdly[qat]), so they are loaded on first access instead of at import time. They are
    # not in __all__ for the same reason: `from keras_hexagdly import *` must not require hgq2.
    if name in _QUANTIZED_NAMES:
        from . import quantized

        return getattr(quantized, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_QUANTIZED_NAMES))


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
