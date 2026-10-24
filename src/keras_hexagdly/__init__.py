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

# The quantized layers need the optional dependency hgq2 (pip install keras-hexagdly[qat]). They
# are imported here when it is installed, so that their Keras registration exists as soon as the
# package is imported (load_model() of a saved QAT model needs it), and are absent otherwise. They
# are not in __all__ either: `from keras_hexagdly import *` must not depend on hgq2.
try:
    from .quantized import (  # noqa: F401
        QAvgPool2d,
        QConv2d,
        QConv2d_CustomKernel,
        QConv3d,
        QConv3d_CustomKernel,
        QMaxPool2d,
        QMaxPool3d,
        freeze,
    )
except Exception as _quantized_failure:  # nothing may break the plain layers
    _QUANTIZED_ERROR = f"{type(_quantized_failure).__name__}: {_quantized_failure}"
    import importlib.util

    if importlib.util.find_spec("hgq") is not None:
        # hgq2 is there but unusable (wrong version, broken install...): say so once
        import warnings

        warnings.warn(
            f"keras_hexagdly: the quantized layers are unavailable ({_QUANTIZED_ERROR})",
            stacklevel=2,
        )
else:
    _QUANTIZED_ERROR = None


def __getattr__(name):
    # only reached for a name that is not defined: hasattr / getattr(default) need an AttributeError
    if name in _QUANTIZED_NAMES and _QUANTIZED_ERROR is not None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}: {_QUANTIZED_ERROR}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
