"""keras-hexagdly: Keras 3 port of HexagDLy (https://github.com/ai4iacts/hexagdly).

HexagDLy is Copyright (c) 2018 ai4iacts (Tim Lukas Holch, Constantin Steppa),
MIT licensed; this file is a derivative work, see ../../LICENSE and
../../NOTICE.md at the package root.

A faithful re-implementation of the PyTorch hexagdly hexagonal convolution /
pooling, kept structurally identical to the original: the same ``HexBase``
sub-kernel decomposition, the same ``shape_for_*`` addressing arithmetic, and
the same ``operation_with_*`` orchestration. Only the *leaf* primitives and the
tensor layout are translated:

  * Tensor layout: PyTorch channels-first (NCHW / NCDHW) -> channels-last
    (NHWC / NDHWC). Every explicit axis index is shifted accordingly; in
    particular the W axis (PyTorch ``size(-1)``) becomes axis ``self.dimensions``.
  * Kernel layout: PyTorch ``(out, in, kH, kW)`` -> ``(kH, kW, in, out)`` (and the
    3D analogue), which is exactly the layout ``keras.ops.conv`` expects, so the
    stored weights feed it directly.
  * Padding: ``nn.ZeroPad2d`` / ``ConstantPad3d`` -> ``keras.ops.pad`` (with the
    pad 4-tuple reordered to the ``[[axis_before, axis_after], ...]`` form).
  * Dilation: hexagdly uses dilation *and* stride > 1 on the same axis, which the
    conv primitive forbids. We therefore pre-dilate the kernel by zero insertion
    (mathematically identical) and convolve with dilation = 1.

Backend-agnostic: built entirely on ``keras`` + ``keras.ops``, so the layers run
under any Keras 3 backend (tensorflow, jax, torch). The result is bit-for-bit
equivalent to the original PyTorch HexagDLy (validated in
tests/test_vs_pytorch_hexagdly.py against the upstream ``hexagdly`` PyPI
package, for random inputs/strides/kernel sizes).

Convolution layers (Conv2d, Conv3d and their *_CustomKernel variants) are the
primary target. MaxPool2d / MaxPool3d are also provided, implemented with a
dilation-capable windowed reduction (the pooling primitive can't combine
dilation+stride either, so pooling uses an explicit shifted-slice max -- see
_MaxPoolProcess).
"""

import operator

import keras
import numpy as np

# ----------------------------------------------------------------------------
# Leaf primitives (the per-framework "process" callables + kernel dilation)
# ----------------------------------------------------------------------------


def _dilate_kernel(kernel, dilation):
    """Zero-insert a conv kernel so a stride-only conv reproduces dilation.

    ``kernel`` is channels-last: ``(kH, kW, in, out)`` (2D) or
    ``(kD, kH, kW, in, out)`` (3D). ``dilation`` matches the spatial axes
    (``(dh, dw)`` or ``(dd, dh, dw)``). A dilated tap grid is built by inserting
    ``d-1`` zero rows between successive taps along each spatial axis -- so
    ``conv(stride=s, dilation=1)`` over it equals ``conv(stride=s, dilation=d)``
    over the original, sidestepping the "dilation and stride can't both be > 1"
    restriction.
    """
    for axis, d in enumerate(dilation):
        d = int(d)
        if d <= 1:
            continue
        shape = list(kernel.shape)
        k = shape[axis]
        # Insert a length-1 axis right after `axis`, pad it to `d`, then merge:
        # [k, ...] -> [k, 1, ...] -> [k, d, ...] -> [k*d, ...] -> trim trailing zeros.
        new_shape = shape[: axis + 1] + [1] + shape[axis + 1 :]
        kernel = keras.ops.reshape(kernel, new_shape)
        pad = [[0, 0]] * len(new_shape)
        pad[axis + 1] = [0, d - 1]
        kernel = keras.ops.pad(kernel, pad)
        merged = shape[:axis] + [k * d] + shape[axis + 1 :]
        kernel = keras.ops.reshape(kernel, merged)
        # Keep (k-1)*d + 1 rows; drop the d-1 trailing zeros after the last tap.
        kernel = kernel[(slice(None),) * axis + (slice(0, (k - 1) * d + 1),)]
    return kernel


# ----------------------------------------------------------------------------
# in_channels is always inferred in build() -- never a constructor argument.
# ----------------------------------------------------------------------------


def _require_positive_int(name, value):
    """Reject non-positive or non-integer kernel_size/stride up front.

    Without this, a bad value (0, negative, or a float) doesn't fail until
    deep inside the first call -- with an error that varies by backend and
    by *where* in the hex sub-kernel decomposition it happens to blow up
    (ZeroDivisionError, a backend-specific "stride must be > 0", a stray
    AttributeError...). Upstream PyTorch HexagDLy has the same gap; this is
    a deliberate improvement over just matching that, not a regression.
    """
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")


def _require_positive_int_or_pair(name, value):
    """Like _require_positive_int, but accepts a 2-tuple (depth, hex) too,
    matching the Conv3d/MaxPool3d kernel_size/stride call convention."""
    if isinstance(value, int):
        _require_positive_int(name, value)
        return
    try:
        a, b = value
    except (TypeError, ValueError):
        raise ValueError(
            f"{name} must be a positive integer or a 2-tuple of positive integers, got {value!r}."
        )
    _require_positive_int(f"{name}[0]", a)
    _require_positive_int(f"{name}[1]", b)


def _require_same_padding(padding):
    """Validate ``padding`` for the 2D hex layers.

    Hex layers always pad so the output size depends only on the stride, never
    on the kernel size -- the hexagonal analogue of Keras "same". The argument
    exists so they accept the same keywords as keras.layers.Conv2D /
    MaxPool2D; any other value is rejected rather than silently ignored.
    """
    if not isinstance(padding, str) or padding.lower() != "same":
        raise ValueError(
            f"padding={padding!r} is not supported: hexagonal layers always pad to "
            "keep the grid, the analogue of 'same'. Use padding='same'."
        )
    return "same"


def _require_bool(name, value):
    """Reject a non-boolean flag rather than reading it as truthy.

    Conv2d/Conv3d take ``use_bias`` in the fourth positional slot, where
    keras.layers.Conv2D takes ``padding``: without this check,
    ``Conv2d(8, 1, 1, "same")`` written the Keras way would silently mean
    ``use_bias="same"``, i.e. True. Returns a plain bool so the config stays
    serializable when a numpy bool is passed.
    """
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a bool, got {value!r}.")
    return bool(value)


def _resolve_pool_size(pool_size, kernel_size):
    """Accept the Keras name ``pool_size`` and the HexagDLy name ``kernel_size``."""
    if pool_size is not None and kernel_size is not None:
        raise ValueError("Pass either pool_size or kernel_size, not both.")
    if pool_size is not None:
        return pool_size
    return 1 if kernel_size is None else kernel_size


def _combined_fan_in_pool(total_fan_in, out_channels, debug):
    """Draw one correctly-scaled pool of weight values, to be sliced into a
    hex conv's several sub-kernel/ring-weight pieces.

    A hex convolution splits each output channel's receptive field across
    several separately-shaped tensors (kernel0, kernel1, ... -- or, with
    share_neighbors, one shared value per ring) whose contributions are
    SUMMED to produce the output. Calling a variance-scaling initializer
    (He-normal) on each piece independently -- as upstream hexagdly,
    pytorch_hexagdly, and this port originally did -- calibrates each
    piece's variance as if it alone were the entire receptive field, so the
    summed output ends up with variance inflated by roughly the number of
    pieces (measured: ~sqrt(N) too much stddev per layer, N = number of
    sub-kernels, compounding across depth).

    Since a variance-scaling initializer draws i.i.d. elements, a contiguous
    slice of a pool sized by the TRUE combined fan-in is itself correctly
    scaled -- so build() draws one (total_fan_in, out_channels) pool here and
    slices+reshapes pieces off it for each sub-kernel/ring tensor, instead of
    initializing each piece from its own (too-small) shape.

    debug=True keeps using all-ones (Ones() sliced anywhere is still all
    1s, so no special-casing needed there).
    """
    init = keras.initializers.Ones() if debug else keras.initializers.HeNormal()
    return keras.ops.convert_to_numpy(init((total_fan_in, out_channels)))


def _infer_in_channels(input_shape):
    """Infer the input channel count from a channels-last input_shape[-1].

    Raises a clear error if the channel axis is undefined, rather than
    failing obscurely deep inside the hex sub-kernel decomposition.
    """
    in_channels = input_shape[-1]
    if in_channels is None:
        raise ValueError(
            "The channel axis of the input is undefined; hexagdly layers need a "
            "static channel dimension. Got input_shape={}.".format(tuple(input_shape))
        )
    return int(in_channels)


def _require_static_spatial(input_shape, dimensions):
    """Enforce that the spatial dims (H, W for 2D; D, H, W for 3D) are known.

    The hex addressing arithmetic in ``shape_for_odd/even_columns`` reads the
    height and width as Python ints (pad/slice amounts depend on the parity of the
    column count), so a dynamic/None spatial dim cannot work. Fail here with a
    readable message rather than crashing obscurely deep inside the conv.
    """
    # channels-last: axis 0 = batch, last axis = channels; spatial axes in between.
    spatial = list(input_shape[1:-1])
    labels = ("H", "W") if dimensions == 2 else ("D", "H", "W")
    if any(s is None for s in spatial):
        named = ", ".join(f"{lbl}={s}" for lbl, s in zip(labels, spatial))
        raise ValueError(
            "hexagdly layers require statically-known spatial dimensions "
            "({}), but got input_shape={} ({}). The hex addressing arithmetic "
            "depends on the exact height/width (and on the parity of the column "
            "count), so they cannot be dynamic. Build the layer on a concrete "
            "shape (e.g. via keras.Input with fixed spatial sizes).".format(
                "/".join(labels), tuple(input_shape), named
            )
        )


# ----------------------------------------------------------------------------
# HexBase: structural twin of the PyTorch HexBase (channels-last)
# ----------------------------------------------------------------------------


class HexBase:
    def __init__(self):
        self.hexbase_size = None
        self.depth_size = None
        self.hexbase_stride = None
        self.depth_stride = None
        # Cache key (the [..., H, W] shape the slices/pads below were computed
        # for), not a one-shot "computed yet?" flag: operation_with_arbitrary_
        # stride is reachable more than once with genuinely different spatial
        # shapes on the SAME layer instance (e.g. a diagnostic helper probing
        # an already-built/already-called layer with a different-sized input;
        # see notebooks/hexplot.py). A plain "computed once" flag would keep
        # serving the first shape's slices/pads forever, silently misindexing
        # -- or, when the odd/even column counts happen to differ, crashing
        # with a shape-mismatch inside keras.ops.concatenate/add.
        self._arbitrary_stride_shape_cache = None
        self.odd_columns_slices = []
        self.odd_columns_pads = []
        self.even_columns_slices = []
        self.even_columns_pads = []
        self.dimensions = None
        self.combine = None
        self.process = None
        self.kwargs = dict()

    # --- input_size: a PyTorch-size-like accessor on channels-last tensors -----
    # The original reads input.size()[-2] (H) and [-1] (W) on an NCHW/NCDHW
    # tensor. Channels-last puts W at axis `dimensions` and H at axis
    # `dimensions - 1`, so we expose a helper returning a NCHW-ordered shape
    # (..., H, W) and keep the original [-2]/[-1] indexing verbatim below.
    def _hw_ordered_size(self, input):
        s = list(input.shape)
        # channels-last: [..., H, W, C]; return [..., H, W] so [-2]=H, [-1]=W.
        return s[:-1]

    def shape_for_odd_columns(self, input_size, kernel_number):
        slices = [None, None, None, None]
        pads = [0, 0, 0, 0]
        # left
        pads[0] = kernel_number
        # right
        pads[1] = max(0, kernel_number - ((input_size[-1] - 1) % (2 * self.hexbase_stride)))
        # top
        pads[2] = self.hexbase_size - int(kernel_number / 2)
        # bottom
        constraint = (
            input_size[-2]
            - 1
            - int((input_size[-2] - 1 - int(self.hexbase_stride / 2)) / self.hexbase_stride)
            * self.hexbase_stride
        )
        bottom = (self.hexbase_size - int((kernel_number + 1) / 2)) - constraint
        if bottom >= 0:
            pads[3] = bottom
        else:
            slices[1] = bottom

        return slices, pads

    def shape_for_even_columns(self, input_size, kernel_number):
        slices = [None, None, None, None]
        pads = [0, 0, 0, 0]
        # left
        left = kernel_number - self.hexbase_stride
        if left >= 0:
            pads[0] = left
        else:
            slices[2] = -left
        # right
        pads[1] = max(
            0,
            kernel_number
            - ((input_size[-1] - 1 - self.hexbase_stride) % (2 * self.hexbase_stride)),
        )
        # top
        top_shift = -(kernel_number % 2) if (self.hexbase_stride % 2) == 1 else 0
        top = (
            (self.hexbase_size - int(kernel_number / 2)) + top_shift - int(self.hexbase_stride / 2)
        )
        if top >= 0:
            pads[2] = top
        else:
            slices[0] = -top
        # bottom
        bottom_shift = 0 if (self.hexbase_stride % 2) == 1 else -(kernel_number % 2)
        pads[3] = max(
            0,
            self.hexbase_size
            - int(kernel_number / 2)
            + bottom_shift
            - ((input_size[-2] - int(self.hexbase_stride / 2) - 1) % self.hexbase_stride),
        )

        return slices, pads

    # --- framework leaves: padding / slicing / dilation / stride / reorder -----

    def get_padded_input(self, input, pads):
        # pads = [left, right, top, bottom] (W then H), as built above.
        left, right, top, bottom = pads
        if self.dimensions == 2:
            # NHWC: pad H (axis 1) and W (axis 2).
            paddings = [[0, 0], [top, bottom], [left, right], [0, 0]]
        else:
            # NDHWC: pad H (axis 2) and W (axis 3); D (axis 1) unpadded.
            paddings = [[0, 0], [0, 0], [top, bottom], [left, right], [0, 0]]
        return keras.ops.pad(input, paddings, mode="constant", constant_values=0)

    def get_sliced_input(self, input, slices):
        # slices index the H (slices[0:2]) and W (slices[2:4]) axes.
        if self.dimensions == 2:
            return input[:, slices[0] : slices[1], slices[2] : slices[3], :]
        else:
            return input[:, :, slices[0] : slices[1], slices[2] : slices[3], :]

    def get_dilation(self, dilation_2d):
        # dilation_2d = (dh, dw) over the hex base; depth never dilated.
        if self.dimensions == 2:
            return tuple(dilation_2d)
        else:
            return tuple([1] + list(dilation_2d))

    def get_stride(self):
        if self.dimensions == 2:
            return (self.hexbase_stride, 2 * self.hexbase_stride)
        else:
            return (self.depth_stride, self.hexbase_stride, 2 * self.hexbase_stride)

    def get_ordered_output(self, input, order):
        # Reorder along the W axis (= self.dimensions in channels-last).
        return keras.ops.take(input, order, axis=self.dimensions)

    def _stabilize_3d_shape(self, input):
        """Launder a 3D input through a slice+concat identity so graph-mode shape
        inference stays consistent through the column-split convolution.

        keras.ops.conv can *trace* an odd/even column width one larger than it
        actually produces at run time when its input tensor was created by a pad
        op -- either the internal depth_padding='same' pad, or an upstream
        ZeroPadding3D in user code.  The stale trace-time width then makes the
        column-reorder gather (get_ordered_output) index out of bounds under
        tf.function / model.predict, e.g. ``indices[9] = 10 is not in [0, 10)``.
        Eager execution is unaffected, which is why it only surfaces in graph
        mode.  Rebuilding the tensor with a genuine concatenate resets that
        inference so trace-time and run-time widths agree.

        Identity operation.  Only the 3D path needs it (2D is unaffected); the
        depth axis (axis 1) always has length >= 1 so the split is safe.
        """
        if self.dimensions == 3:
            input = keras.ops.concatenate([input[:, :1], input[:, 1:]], axis=1)
        return input

    # --- general operation with arbitrary stride (verbatim orchestration) ------
    def operation_with_arbitrary_stride(self, input):
        input = self._stabilize_3d_shape(input)
        assert self._hw_ordered_size(input)[-2] - (self.hexbase_stride // 2) >= 0, (
            "Too few rows to apply hex conv with the stide that is set"
        )
        odd_columns = None
        even_columns = None
        # With no more columns than the stride, the even-column group of the
        # output is empty and the odd group is the whole output. Computing the
        # empty group anyway runs an op on a zero-width input, which fails on
        # the TensorFlow and torch backends (and in upstream HexagDLy), so it
        # is skipped.
        has_even_columns = self._hw_ordered_size(input)[-1] > self.hexbase_stride

        # Recompute the per-sub-kernel slices/pads whenever the actual [..., H, W]
        # shape differs from the one they were last computed for -- not just
        # once ever. See the cache field's docstring in __init__ for why: this
        # method can be reached again with a genuinely different spatial shape
        # on the same layer instance, and reusing stale slices/pads silently
        # misindexes (or crashes on shape mismatch when the odd/even column
        # counts happen to differ between the two shapes).
        current_shape = tuple(self._hw_ordered_size(input))
        if current_shape != self._arbitrary_stride_shape_cache:
            self.odd_columns_slices = []
            self.odd_columns_pads = []
            self.even_columns_slices = []
            self.even_columns_pads = []
            for i in range(self.hexbase_size + 1):
                slices, pads = self.shape_for_odd_columns(self._hw_ordered_size(input), i)
                self.odd_columns_slices.append(slices)
                self.odd_columns_pads.append(pads)
                slices, pads = self.shape_for_even_columns(self._hw_ordered_size(input), i)
                self.even_columns_slices.append(slices)
                self.even_columns_pads.append(pads)
            self._arbitrary_stride_shape_cache = current_shape

        for i in range(self.hexbase_size + 1):
            dilation_base = (1, 1) if i == 0 else (1, 2 * i)

            if odd_columns is None:
                odd_columns = self.process(
                    self.get_padded_input(
                        self.get_sliced_input(input, self.odd_columns_slices[i]),
                        self.odd_columns_pads[i],
                    ),
                    getattr(self, "kernel" + str(i)),
                    dilation=self.get_dilation(dilation_base),
                    stride=self.get_stride(),
                    **self.kwargs,
                )
            else:
                odd_columns = self.combine(
                    odd_columns,
                    self.process(
                        self.get_padded_input(
                            self.get_sliced_input(input, self.odd_columns_slices[i]),
                            self.odd_columns_pads[i],
                        ),
                        getattr(self, "kernel" + str(i)),
                        dilation=self.get_dilation(dilation_base),
                        stride=self.get_stride(),
                    ),
                )

            if not has_even_columns:
                continue
            if even_columns is None:
                even_columns = self.process(
                    self.get_padded_input(
                        self.get_sliced_input(input, self.even_columns_slices[i]),
                        self.even_columns_pads[i],
                    ),
                    getattr(self, "kernel" + str(i)),
                    dilation=self.get_dilation(dilation_base),
                    stride=self.get_stride(),
                    **self.kwargs,
                )
            else:
                even_columns = self.combine(
                    even_columns,
                    self.process(
                        self.get_padded_input(
                            self.get_sliced_input(input, self.even_columns_slices[i]),
                            self.even_columns_pads[i],
                        ),
                        getattr(self, "kernel" + str(i)),
                        dilation=self.get_dilation(dilation_base),
                        stride=self.get_stride(),
                    ),
                )

        if not has_even_columns:
            return odd_columns

        concatenated_columns = keras.ops.concatenate((odd_columns, even_columns), self.dimensions)

        n_odd_columns = list(odd_columns.shape)[self.dimensions]
        n_even_columns = list(even_columns.shape)[self.dimensions]
        if n_odd_columns == n_even_columns:
            order = [int(i + x * n_even_columns) for i in range(n_even_columns) for x in range(2)]
        else:
            order = [int(i + x * n_odd_columns) for i in range(n_even_columns) for x in range(2)]
            order.append(n_even_columns)

        return self.get_ordered_output(concatenated_columns, order)

    # --- faster single-stride special case (verbatim orchestration) -----------
    def operation_with_single_hexbase_stride(self, input):
        input = self._stabilize_3d_shape(input)
        columns_mod2 = self._hw_ordered_size(input)[-1] % 2
        # A single-column grid has no even columns, so the odd-kernel
        # even-column group is empty -- skipped for the same reason as in
        # operation_with_arbitrary_stride.
        has_even_columns = self._hw_ordered_size(input)[-1] > 1
        odd_kernels_odd_columns = []
        odd_kernels_even_columns = []
        even_kernels_all_columns = []

        even_kernels_all_columns = self.process(
            self.get_padded_input(input, [0, 0, self.hexbase_size, self.hexbase_size]),
            self.kernel0,
            stride=(1, 1) if self.dimensions == 2 else (self.depth_stride, 1, 1),
            **self.kwargs,
        )
        if self.hexbase_size >= 1:
            odd_kernels_odd_columns = self.process(
                self.get_padded_input(
                    input, [1, columns_mod2, self.hexbase_size, self.hexbase_size - 1]
                ),
                self.kernel1,
                dilation=self.get_dilation((1, 2)),
                stride=self.get_stride(),
            )
            if has_even_columns:
                odd_kernels_even_columns = self.process(
                    self.get_padded_input(
                        input,
                        [0, 1 - columns_mod2, self.hexbase_size - 1, self.hexbase_size],
                    ),
                    self.kernel1,
                    dilation=self.get_dilation((1, 2)),
                    stride=self.get_stride(),
                )

        if self.hexbase_size > 1:
            for i in range(2, self.hexbase_size + 1):
                if i % 2 == 0:
                    even_kernels_all_columns = self.combine(
                        even_kernels_all_columns,
                        self.process(
                            self.get_padded_input(
                                input,
                                [
                                    i,
                                    i,
                                    self.hexbase_size - int(i / 2),
                                    self.hexbase_size - int(i / 2),
                                ],
                            ),
                            getattr(self, "kernel" + str(i)),
                            dilation=self.get_dilation((1, 2 * i)),
                            stride=(1, 1) if self.dimensions == 2 else (self.depth_stride, 1, 1),
                        ),
                    )
                else:
                    x = self.hexbase_size + int((1 - i) / 2)
                    odd_kernels_odd_columns = self.combine(
                        odd_kernels_odd_columns,
                        self.process(
                            self.get_padded_input(input, [i, i - 1 + columns_mod2, x, x - 1]),
                            getattr(self, "kernel" + str(i)),
                            dilation=self.get_dilation((1, 2 * i)),
                            stride=self.get_stride(),
                        ),
                    )
                    if has_even_columns:
                        odd_kernels_even_columns = self.combine(
                            odd_kernels_even_columns,
                            self.process(
                                self.get_padded_input(input, [i - 1, i - columns_mod2, x - 1, x]),
                                getattr(self, "kernel" + str(i)),
                                dilation=self.get_dilation((1, 2 * i)),
                                stride=self.get_stride(),
                            ),
                        )

        if not has_even_columns:
            return self.combine(even_kernels_all_columns, odd_kernels_odd_columns)

        odd_kernels_concatenated_columns = keras.ops.concatenate(
            (odd_kernels_odd_columns, odd_kernels_even_columns), self.dimensions
        )

        n_odd_columns = list(odd_kernels_odd_columns.shape)[self.dimensions]
        n_even_columns = list(odd_kernels_even_columns.shape)[self.dimensions]
        if n_odd_columns == n_even_columns:
            order = [int(i + x * n_even_columns) for i in range(n_even_columns) for x in range(2)]
        else:
            order = [int(i + x * n_odd_columns) for i in range(n_even_columns) for x in range(2)]
            order.append(n_even_columns)

        return self.combine(
            even_kernels_all_columns,
            self.get_ordered_output(odd_kernels_concatenated_columns, order),
        )


# ----------------------------------------------------------------------------
# Convolution process leaf
# ----------------------------------------------------------------------------


class _ConvProcess:
    """Callable mimicking F.conv2d/conv3d on channels-last tensors.

    Kernels are stored channels-last (``(kH, kW, in, out)`` / 3D analogue), the
    layout ``keras.ops.conv`` expects. Dilation is realised by pre-dilating the
    kernel (zero insertion), so the actual conv always runs with dilation 1 and
    'valid' padding (the explicit hexagdly padding is already applied upstream).
    """

    def __init__(self, dimensions):
        self.dimensions = dimensions

    def __call__(self, input, kernel, dilation=None, stride=None, bias=None):
        if self.dimensions == 2:
            if dilation is not None:
                kernel = _dilate_kernel(kernel, dilation)
            sh, sw = (1, 1) if stride is None else stride
            out = keras.ops.conv(
                input,
                kernel,
                strides=(sh, sw),
                padding="valid",
                data_format="channels_last",
            )
        else:
            if dilation is not None:
                kernel = _dilate_kernel(kernel, dilation)
            sd, sh, sw = (1, 1, 1) if stride is None else stride
            out = keras.ops.conv(
                input,
                kernel,
                strides=(sd, sh, sw),
                padding="valid",
                data_format="channels_last",
            )
        if bias is not None:
            # channels-last: bias shape (out_channels,) broadcasts over the last axis.
            out = keras.ops.add(out, bias)
        return out


# ----------------------------------------------------------------------------
# Max-pool process leaf (dilation + stride via shifted-slice reduction)
# ----------------------------------------------------------------------------


class _MaxPoolProcess:
    """Callable mimicking F.max_pool2d/3d on channels-last tensors.

    The pooling primitive can't combine dilation and stride, so we reduce over
    the window explicitly: for each tap offset in the (dilated) window we take a
    strided slice of the input and max them together. Exactly reproduces a hex
    max pool. The "kernel" here is a window *size* tuple, as in the original.
    """

    reduce = staticmethod(keras.ops.maximum)

    def __init__(self, dimensions):
        self.dimensions = dimensions

    def __call__(self, input, window, dilation=None, stride=None, bias=None):
        if self.dimensions == 2:
            kh, kw = window
            dh, dw = (1, 1) if dilation is None else dilation
            sh, sw = (1, 1) if stride is None else stride
            taps = [input[:, a * dh :: sh, b * dw :: sw, :] for a in range(kh) for b in range(kw)]
            return _reduce_aligned(taps, (1, 2), self.reduce)
        else:
            kd, kh, kw = window
            dd, dh, dw = (1, 1, 1) if dilation is None else dilation
            sd, sh, sw = (1, 1, 1) if stride is None else stride
            taps = [
                input[:, c * dd :: sd, a * dh :: sh, b * dw :: sw, :]
                for c in range(kd)
                for a in range(kh)
                for b in range(kw)
            ]
            return _reduce_aligned(taps, (1, 2, 3), self.reduce)


class _SumPoolProcess(_MaxPoolProcess):
    """Same window walk as _MaxPoolProcess, summing the taps instead.

    Sums with ``+`` rather than keras.ops.add: on the TensorFlow backend
    keras.ops.add rewrites some additions as bias_add, which fails on the
    zero-width column group a narrow grid with a large stride produces.
    keras.ops.maximum, used by MaxPool2d, has no such rewrite.
    """

    reduce = staticmethod(operator.add)


def _reduce_aligned(taps, axes, op):
    """Element-wise ``op`` (max or sum) of window taps, each cropped to the
    common minimum size.

    Strided shifted slices differ in length by up to one along each axis; the
    last (most-shifted) tap is the shortest and its size equals the VALID-conv
    output size, so cropping every tap to the per-axis minimum matches the
    PyTorch pooling output exactly.
    """
    min_sizes = {}
    for ax in axes:
        min_sizes[ax] = min(int(t.shape[ax]) for t in taps)
    acc = None
    for t in taps:
        sl = [slice(None)] * len(t.shape)
        for ax in axes:
            sl[ax] = slice(0, min_sizes[ax])
        t = t[tuple(sl)]
        acc = t if acc is None else op(acc, t)
    return acc


# ----------------------------------------------------------------------------
# Weight sharing (share_neighbors): tie kernel cells together by physical
# offset, three ways -- ported from pytorch-hexagdly, which is the reference
# implementation for this feature (this port previously had only "ring", and
# had a real bug in it; see the note on _PROBE_ANCHOR_PARITY below).
#
#   "ring" -- one weight per hex ring (n+1 weights total): TDSCAN-style, all
#             cells equidistant from the center share a weight.
#   "diag" -- antipodal pairs: cells at offset (dr,dc) and (-dr,-dc) share.
#             1 + 3*n*(n+1)/2 weights.
#   "sym"  -- adjacent pairs along the outer boundary of each ring, grouped by
#             60-degree rotational symmetry. Same weight count as "diag".
#
# The hexagdly offset layout does not map cleanly to a textbook hex-coordinate
# distance formula (the per-column vertical shift is asymmetric). Rather than
# risk a silently-wrong closed form, "ring" is derived EMPIRICALLY: a
# single-tap impulse through the (validated) custom-kernel conv reveals each
# cell's physical (row, col) offset from the center, and the ring is the
# smallest kernel size whose support contains that offset. "diag"/"sym" reuse
# pytorch-hexagdly's own hand-verified tables (n=1,2 only -- neither this port
# nor upstream defines them for n>2).
# ----------------------------------------------------------------------------

_WEIGHT_MAP_CACHE = {}

# _tap_offset probes kernel size nn using a synthetic grid of size g = 6*nn+11,
# centered at cen = g//2. Since g grows with nn, cen's COLUMN PARITY flips
# between probes of different nn (g=17 -> cen=8, even; g=23 -> cen=11, odd) --
# and hexagdly's addressing is parity-dependent (the zig-zag itself), so two
# probes run at different nn were being read against inconsistent parity
# references. Concretely: ring_maps_2d(2) classified 4 of 19 taps in the wrong
# ring (the (+-1,+-1) diagonal cells), because support[1] (probed at nn=1,
# even column) and support[2] (probed at nn=2, odd column) don't describe the
# same physical geometry. Anchoring every probe to the SAME column parity,
# verified to reproduce pytorch-hexagdly's hand-checked ring map for n=2
# exactly (element-for-element), fixes it.
_PROBE_ANCHOR_PARITY = 0  # even column; arbitrary but must be probe-invariant


def _tap_offset(n, i, r, c):
    """Physical (dr, dc) offset of sub-kernel cell (i, r, c) for kernel size n.

    Fires a single tap through Conv2d_CustomKernel on a centered impulse and
    reads back where the 1.0 lands. Pure measurement, no coordinate
    assumptions -- except the probe's center column parity is pinned to
    _PROBE_ANCHOR_PARITY (see that constant for why).
    """
    g = 6 * n + 11
    cen = g // 2
    if cen % 2 != _PROBE_ANCHOR_PARITY:
        cen += 1
    imp = np.zeros((1, g, g, 1), np.float32)
    imp[0, cen, cen, 0] = 1.0
    subk = []
    for k in range(n + 1):
        kh = 2 * n + 1 - k
        kw = 1 if k == 0 else 2
        a = np.zeros((1, 1, kh, kw), np.float32)
        if k == i:
            a[0, 0, r, c] = 1.0
        subk.append(a)
    out = keras.ops.convert_to_numpy(
        Conv2d_CustomKernel(sub_kernels=subk, strides=1)(keras.ops.convert_to_tensor(imp))
    )[0, :, :, 0]
    pos = np.argwhere(np.isclose(out, 1.0))
    return int(pos[0][0] - cen), int(pos[0][1] - cen)


def _ring_maps_2d_uncached(n):
    """Empirical ring derivation (see the module-level note above)."""
    support = {}
    for nn in range(1, n + 1):
        offs = set()
        for i in range(nn + 1):
            rows = 2 * nn + 1 - i
            cols = 1 if i == 0 else 2
            for r in range(rows):
                for c in range(cols):
                    offs.add(_tap_offset(nn, i, r, c))
        support[nn] = offs

    def ring_of(off):
        if off == (0, 0):
            return 0
        for nn in range(1, n + 1):
            if off in support[nn]:
                return nn
        raise ValueError(f"offset {off} not within kernel size {n}")

    ring_maps = []
    for i in range(n + 1):
        rows = 2 * n + 1 - i
        cols = 1 if i == 0 else 2
        m = np.zeros((rows, cols), dtype=np.int32)
        for r in range(rows):
            for c in range(cols):
                m[r, c] = ring_of(_tap_offset(n, i, r, c))
        ring_maps.append(m)
    return ring_maps, n + 1


# Hand-verified weight-group tables for "diag" and "sym", ported verbatim from
# pytorch-hexagdly's `_HARDCODED_MAPS` (the reference implementation for these
# two modes). Both modes are parity-independent at n=1,2 (upstream carries a
# per-parity variant purely for a "diag"/"sym" future beyond n=2 that neither
# port actually implements; the even- and odd-column tables it stores are
# identical, so only one copy is kept here). Sub-kernel shapes/ordering are
# identical between the two ports (both mirror hexagdly's own decomposition),
# so these arrays apply to keras-hexagdly's Conv2d_CustomKernel sub-kernels
# unchanged -- confirmed by cross-checking "ring" n=2 element-for-element
# against pytorch-hexagdly's own hardcoded ring table after the parity fix
# above.
_DIAG_SYM_MAPS = {
    ("diag", 1): ([np.array([[3], [0], [3]]), np.array([[1, 2], [2, 1]])], 4),
    ("diag", 2): (
        [
            np.array([[1], [5], [0], [5], [1]]),
            np.array([[3, 2], [8, 7], [7, 8], [2, 3]]),
            np.array([[6, 4], [9, 9], [4, 6]]),
        ],
        10,
    ),
    ("sym", 1): ([np.array([[2], [0], [3]]), np.array([[2, 1], [3, 1]])], 4),
    ("sym", 2): (
        [
            np.array([[4], [3], [0], [1], [7]]),
            np.array([[9, 4], [3, 2], [1, 2], [7, 6]]),
            np.array([[9, 5], [8, 5], [8, 6]]),
        ],
        10,
    ),
}

SHARE_NEIGHBORS_MODES = ("ring", "diag", "sym")


def weight_maps_2d(n, mode="ring"):
    """For kernel size ``n`` and sharing ``mode``, return ``(weight_maps, num_weights)``.

    ``weight_maps[i]`` is an int array of shape ``(rows_i, cols_i)`` giving the
    weight-group index of every cell of sub-kernel ``i``; ``num_weights`` is the
    number of distinct shared weights that mode produces for this kernel size.

    mode="ring": any kernel size (derived empirically, cached).
    mode="diag"/"sym": kernel_size 1 or 2 only (hand-verified tables, ported
    from pytorch-hexagdly, which does not define them beyond n=2 either).
    """
    if mode not in SHARE_NEIGHBORS_MODES:
        raise ValueError(
            f"share_neighbors mode must be one of {SHARE_NEIGHBORS_MODES}, got {mode!r}."
        )
    key = (mode, n)
    if key in _WEIGHT_MAP_CACHE:
        return _WEIGHT_MAP_CACHE[key]

    if mode == "ring":
        result = _ring_maps_2d_uncached(n)
    else:
        if (mode, n) not in _DIAG_SYM_MAPS:
            raise NotImplementedError(
                f"share_neighbors={mode!r} is only implemented for kernel_size 1 and 2, "
                f"got kernel_size={n}."
            )
        maps, num = _DIAG_SYM_MAPS[(mode, n)]
        result = ([m.copy() for m in maps], num)

    _WEIGHT_MAP_CACHE[key] = result
    return result


def ring_maps_2d(n):
    """Back-compat alias: ``weight_maps_2d(n, "ring")``."""
    return weight_maps_2d(n, "ring")


def _normalize_share_neighbors(value):
    """False/None -> False (off); True -> "ring" (back-compat default mode);
    a mode string -> itself, validated."""
    if value is False or value is None:
        return False
    if value is True:
        return "ring"
    if value in SHARE_NEIGHBORS_MODES:
        return value
    raise ValueError(
        f"share_neighbors must be False, True, or one of {SHARE_NEIGHBORS_MODES}, got {value!r}."
    )


# ----------------------------------------------------------------------------
# Public layers
# ----------------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="hexagdly_tf")
class Conv2d(HexBase, keras.layers.Layer):
    """2D hexagonal convolution (Keras port of hexagdly.Conv2d).

    Input/output are channels-last: ``(N, H, W, C)``.

    Precondition: the spatial dimensions ``H`` and ``W`` must be statically known
    at build time. The hex addressing arithmetic computes pad/slice amounts from
    the exact height/width (and the parity of the column count), so neither can be
    dynamic (``None``); ``build`` raises a clear ``ValueError`` if either is. The
    channel dimension is always inferred from ``input_shape[-1]`` -- there is no
    constructor argument for it, matching standard Keras layers.

    ``padding`` and ``activation`` are accepted as keyword arguments like in
    ``keras.layers.Conv2D``, so either layer can be built from one set of
    keyword arguments. ``activation`` takes the same values; ``padding`` only
    accepts ``"same"``, since the hex grid is always padded and the output
    size depends only on the stride.
    """

    def __init__(
        self,
        filters,
        kernel_size=1,
        strides=1,
        use_bias=True,
        debug=False,
        share_neighbors=False,
        name=None,
        padding="same",
        activation=None,
        **kwargs,
    ):
        keras.layers.Layer.__init__(self, name=name, **kwargs)
        HexBase.__init__(self)
        _require_positive_int("kernel_size", kernel_size)
        _require_positive_int("strides", strides)
        self.in_channels = None
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = _require_same_padding(padding)
        self.activation = keras.activations.get(activation)
        self.hexbase_size = kernel_size
        self.hexbase_stride = strides
        self.debug = _require_bool("debug", debug)
        self.use_bias = _require_bool("use_bias", use_bias)
        self.share_neighbors = _normalize_share_neighbors(share_neighbors)
        self.dimensions = 2
        self.process = _ConvProcess(2)
        self.combine = keras.ops.add
        # Weight-group geometry depends only on kernel size + mode, so it can be
        # derived now.
        if self.share_neighbors:
            self._ring_maps, self.num_rings = weight_maps_2d(
                self.hexbase_size, self.share_neighbors
            )
        self.kwargs = {"bias": None}

    def build(self, input_shape):
        _require_static_spatial(input_shape, self.dimensions)
        in_channels = _infer_in_channels(input_shape)
        self.in_channels = in_channels
        out_channels = self.filters

        # Total physical taps across kernel0..kernel{hexbase_size} -- the TRUE
        # combined receptive field whose summed output needs variance-preserving
        # init. See _combined_fan_in_pool for why this must be combined, not
        # computed per sub-kernel.
        total_taps = sum(
            (1 + 2 * self.hexbase_size - i) * (1 if i == 0 else 2)
            for i in range(self.hexbase_size + 1)
        )
        pool = _combined_fan_in_pool(total_taps * in_channels, out_channels, self.debug)
        offset = 0

        # NOTE: trainable weights are created with self.add_weight (NOT a raw
        # Variable) so Keras tracks them in trainable_variables -- essential for
        # use inside a functional Model. They are kept under dedicated attribute
        # names; the kernel{i} attributes the forward pass reads are *derived*
        # tensors set per call (materialized below), never the source of truth.
        if self.share_neighbors:
            # One weight per hex ring, broadcast to every cell of that ring at
            # forward time (see _materialize_kernels). num_rings = n + 1.
            count = self.num_rings * in_channels
            values = pool[offset : offset + count].reshape(
                self.num_rings, in_channels, out_channels
            )
            offset += count
            self.ring_weights = self.add_weight(
                name="ring_weights",
                shape=(self.num_rings, in_channels, out_channels),
                initializer=keras.initializers.Constant(values),
                trainable=True,
            )
            # Ring-index tensors only depend on kernel geometry, not on the
            # weight values, so convert them from numpy once here rather than
            # on every forward call (see _materialize_kernels).
            self._ring_idx = [
                keras.ops.convert_to_tensor(m, dtype="int32") for m in self._ring_maps
            ]
        else:
            self._base_kernels = []
            for i in range(self.hexbase_size + 1):
                kh = 1 + 2 * self.hexbase_size - i
                kw = 1 if i == 0 else 2
                count = kh * kw * in_channels
                values = pool[offset : offset + count].reshape(kh, kw, in_channels, out_channels)
                offset += count
                self._base_kernels.append(
                    self.add_weight(
                        name=f"base_kernel{i}",
                        shape=(kh, kw, in_channels, out_channels),
                        initializer=keras.initializers.Constant(values),
                        trainable=True,
                    )
                )

        if self.use_bias:
            bval = keras.initializers.Ones() if self.debug else keras.initializers.Constant(0.01)
            self.bias_tensor = self.add_weight(
                name="bias", shape=(out_channels,), initializer=bval, trainable=True
            )
            self.kwargs = {"bias": self.bias_tensor}
        else:
            self.kwargs = {"bias": None}
        super().build(input_shape)

    def _materialize_kernels(self):
        """Set the kernel{i} tensors the forward pass reads.

        Non-shared: kernel{i} is just the tracked base weight. Shared: kernel{i}
        is gathered from ring_weights via the ring map (gradient flows back into
        ring_weights, so every cell of a ring shares one weight, like TDSCAN).
        Either way kernel{i} is a derived tensor, recomputed each call, never a
        tracked Variable -- the tracked weights are ring_weights / _base_kernels.
        """
        if self.share_neighbors:
            for i in range(self.hexbase_size + 1):
                setattr(
                    self,
                    "kernel" + str(i),
                    keras.ops.take(self.ring_weights, self._ring_idx[i], axis=0),
                )
        else:
            for i in range(self.hexbase_size + 1):
                setattr(self, "kernel" + str(i), self._base_kernels[i])

    def call(self, input):
        self._materialize_kernels()
        if self.hexbase_stride == 1:
            out = self.operation_with_single_hexbase_stride(input)
        else:
            out = self.operation_with_arbitrary_stride(input)
        return self.activation(out)

    def get_config(self):
        # NOTE: no 'in_channels' key -- it's not a constructor argument (always
        # inferred in build()), so it must not appear here or from_config()
        # would pass it back in as an unrecognized kwarg.
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "activation": keras.activations.serialize(self.activation),
                "use_bias": self.use_bias,
                "debug": self.debug,
                "share_neighbors": self.share_neighbors,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="hexagdly_tf")
class Conv3d(HexBase, keras.layers.Layer):
    """3D hexagonal convolution (Keras port of hexagdly.Conv3d).

    Input/output are channels-last: ``(N, D, H, W, C)`` (D = depth/time).

    Precondition: the spatial dimensions ``D``, ``H`` and ``W`` must be statically
    known at build time. The hex addressing arithmetic computes pad/slice amounts
    from the exact height/width (and the parity of the column count), so they
    cannot be dynamic (``None``); ``build`` raises a clear ``ValueError`` if any
    is. The channel dimension is always inferred from ``input_shape[-1]`` -- there
    is no constructor argument for it, matching standard Keras layers.

    ``activation`` takes the same values as in ``keras.layers.Conv3D``. Border
    handling stays split as before: the hex plane is always padded, and the
    depth axis is controlled by ``depth_padding``.
    """

    def __init__(
        self,
        filters,
        kernel_size=1,
        strides=1,
        use_bias=True,
        debug=False,
        share_neighbors=False,
        depth_padding="valid",
        name=None,
        activation=None,
        **kwargs,
    ):
        keras.layers.Layer.__init__(self, name=name, **kwargs)
        HexBase.__init__(self)
        _require_positive_int_or_pair("kernel_size", kernel_size)
        _require_positive_int_or_pair("strides", strides)
        self.in_channels = None
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = keras.activations.get(activation)
        if isinstance(kernel_size, int):
            self.hexbase_size = kernel_size
            self.depth_size = kernel_size
        else:
            assert len(kernel_size) == 2, "Need a tuple of two ints to set kernel size"
            self.hexbase_size = kernel_size[1]
            self.depth_size = kernel_size[0]
        if isinstance(strides, int):
            self.hexbase_stride = strides
            self.depth_stride = strides
        else:
            assert len(strides) == 2, "Need a tuple of two ints to set strides"
            self.hexbase_stride = strides[1]
            self.depth_stride = strides[0]
        self.debug = _require_bool("debug", debug)
        self.use_bias = _require_bool("use_bias", use_bias)
        self.share_neighbors = _normalize_share_neighbors(share_neighbors)
        # Depth (time) border handling. "valid" (default) = no depth pad, output
        # depth = D - kernel + 1 (original hexagdly behaviour). "same" = zero-pad
        # (kernel-1)//2 on each side so output depth = input depth, kernel centred
        # on each time step -- this matches TDSCAN's temporal window (eps_t each
        # side, zero-padded borders). Only affects the depth axis; the hex base is
        # untouched.
        if depth_padding not in ("valid", "same"):
            raise ValueError("depth_padding must be 'valid' or 'same'.")
        self.depth_padding = depth_padding
        self.dimensions = 3
        self.process = _ConvProcess(3)
        self.combine = keras.ops.add
        if self.share_neighbors:
            self._ring_maps, self.num_rings = weight_maps_2d(
                self.hexbase_size, self.share_neighbors
            )
        self.kwargs = {"bias": None}

    def build(self, input_shape):
        _require_static_spatial(input_shape, self.dimensions)
        in_channels = _infer_in_channels(input_shape)
        self.in_channels = in_channels
        out_channels = self.filters

        # Total physical taps across kernel0..kernel{hexbase_size} (including
        # the depth axis) -- see Conv2d.build / _combined_fan_in_pool.
        total_taps = self.depth_size * sum(
            (1 + 2 * self.hexbase_size - i) * (1 if i == 0 else 2)
            for i in range(self.hexbase_size + 1)
        )
        pool = _combined_fan_in_pool(total_taps * in_channels, out_channels, self.debug)
        offset = 0

        # Trainable weights via self.add_weight (Keras tracks them); kernel{i} the
        # forward pass reads are derived tensors set per call (see _materialize_kernels).
        if self.share_neighbors:
            # Share over the HEX axes only; the depth (time) axis stays fully
            # independent -- exactly TDSCAN's (L, num_rings, ...) layout. So
            # ring_weights is (depth, num_rings, in, out).
            count = self.depth_size * self.num_rings * in_channels
            values = pool[offset : offset + count].reshape(
                self.depth_size, self.num_rings, in_channels, out_channels
            )
            offset += count
            self.ring_weights = self.add_weight(
                name="ring_weights",
                shape=(self.depth_size, self.num_rings, in_channels, out_channels),
                initializer=keras.initializers.Constant(values),
                trainable=True,
            )
            # See Conv2d.build: cache the ring-index tensors once rather than
            # reconverting them from numpy on every forward call.
            self._ring_idx = [
                keras.ops.convert_to_tensor(m, dtype="int32") for m in self._ring_maps
            ]
        else:
            self._base_kernels = []
            for i in range(self.hexbase_size + 1):
                kh = 1 + 2 * self.hexbase_size - i
                kw = 1 if i == 0 else 2
                count = self.depth_size * kh * kw * in_channels
                values = pool[offset : offset + count].reshape(
                    self.depth_size, kh, kw, in_channels, out_channels
                )
                offset += count
                self._base_kernels.append(
                    self.add_weight(
                        name=f"base_kernel{i}",
                        shape=(self.depth_size, kh, kw, in_channels, out_channels),
                        initializer=keras.initializers.Constant(values),
                        trainable=True,
                    )
                )

        if self.use_bias:
            bval = keras.initializers.Ones() if self.debug else keras.initializers.Constant(0.01)
            self.bias_tensor = self.add_weight(
                name="bias", shape=(out_channels,), initializer=bval, trainable=True
            )
            self.kwargs = {"bias": self.bias_tensor}
        else:
            self.kwargs = {"bias": None}
        super().build(input_shape)

    def _materialize_kernels(self):
        """Set kernel{i} for the forward pass: gathered from ring_weights along
        the ring axis (depth left independent) when shared, else the base weight."""
        if self.share_neighbors:
            for i in range(self.hexbase_size + 1):
                # ring_weights (depth, num_rings, in, out) gather axis=1 ->
                # (depth, rows, cols, in, out)
                setattr(
                    self,
                    "kernel" + str(i),
                    keras.ops.take(self.ring_weights, self._ring_idx[i], axis=1),
                )
        else:
            for i in range(self.hexbase_size + 1):
                setattr(self, "kernel" + str(i), self._base_kernels[i])

    def call(self, input):
        self._materialize_kernels()
        if self.depth_padding == "same":
            # Symmetric zero-pad the depth axis (NDHWC -> axis 1) so the temporal
            # kernel is centred and output depth == input depth, like TDSCAN.
            # The resulting pad-op tensor is laundered by _stabilize_3d_shape at
            # the entry of operation_with_*_hexbase_stride so it stays correct in
            # graph mode (see that method for why).
            pad = (self.depth_size - 1) // 2
            top = pad
            bot = self.depth_size - 1 - pad  # handles even kernels too
            input = keras.ops.pad(input, [[0, 0], [top, bot], [0, 0], [0, 0], [0, 0]])
        if self.hexbase_stride == 1:
            out = self.operation_with_single_hexbase_stride(input)
        else:
            out = self.operation_with_arbitrary_stride(input)
        return self.activation(out)

    def get_config(self):
        # NOTE: no 'in_channels' key -- see the matching note in Conv2d.get_config.
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "activation": keras.activations.serialize(self.activation),
                "use_bias": self.use_bias,
                "debug": self.debug,
                "share_neighbors": self.share_neighbors,
                "depth_padding": self.depth_padding,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="hexagdly_tf")
class Conv2d_CustomKernel(HexBase, keras.layers.Layer):
    """2D hexagonal convolution with caller-supplied sub-kernels.

    ``sub_kernels`` is a list of numpy arrays in the *PyTorch* hexagdly layout
    ``(out, in, rows, cols)`` (so existing hexagdly kernels drop in unchanged);
    they are transposed to channels-last layout internally.

    Precondition: the spatial dimensions (``H``, ``W``) must be statically known
    when the layer is called -- the hex addressing arithmetic reads them as ints.
    """

    def __init__(
        self,
        sub_kernels=None,
        strides=1,
        bias=None,
        trainable=False,
        debug=False,
        name=None,
        **kwargs,
    ):
        # Note: 'bias' here is the actual bias array (data), not a boolean
        # flag like Conv2d's -- it has no Keras-native rename, unlike stride.
        _require_positive_int("strides", strides)
        keras.layers.Layer.__init__(self, name=name, **kwargs)
        HexBase.__init__(self)
        self.dimensions = 2
        self.process = _ConvProcess(2)
        self.combine = keras.ops.add
        self.strides = strides
        self.hexbase_stride = strides
        self._trainable_kernels = trainable
        self._debug = debug

        # np.asarray here (not just list(...)): get_config()/from_config() round-trips
        # sub_kernels through plain Python lists (via .tolist()), and _check_sub_kernels
        # below needs real arrays (.ndim, .shape).
        sub_kernels = (
            [np.asarray(sk, dtype=np.float32) for sk in sub_kernels] if sub_kernels else []
        )
        if debug or len(sub_kernels) == 0:
            sub_kernels = [
                np.array([[[[1], [1], [1]]]]),
                np.array([[[[1, 1], [1, 1]]]]),
            ]
        self.hexbase_size = len(sub_kernels) - 1
        self._check_sub_kernels(sub_kernels)
        # Keep the construction sub-kernels (PyTorch layout) + bias for get_config;
        # actual weight *values* are restored separately by Keras on load.
        self._sub_kernels_init = [np.asarray(sk, dtype=np.float32) for sk in sub_kernels]
        self._bias_init = None if bias is None else np.asarray(bias, dtype=np.float32)
        for i, sk in enumerate(sub_kernels):
            # (out, in, rows, cols) -> (rows, cols, in, out)
            tf_k = np.transpose(np.asarray(sk, dtype=np.float32), (2, 3, 1, 0))
            setattr(
                self,
                "kernel" + str(i),
                keras.Variable(tf_k, trainable=trainable, name=f"kernel{i}"),
            )

        if not debug and bias is not None:
            self.bias_tensor = keras.Variable(
                np.asarray(bias, dtype=np.float32), trainable=trainable, name="bias"
            )
            self.kwargs = {"bias": self.bias_tensor}
            self.use_bias = True
        else:
            self.use_bias = False
            self.kwargs = {"bias": None}

    def _check_sub_kernels(self, sub_kernels):
        for i, sk in enumerate(sub_kernels):
            assert sk.ndim == 4, "sub-kernels must be rank 4 for a 2d convolution"
            if i == 0:
                assert sk.shape[3] == 1, "first sub-kernel must have only 1 column"
                assert sk.shape[2] == 2 * self.hexbase_size + 1, (
                    "first sub-kernel must have 2*(kernel size)+1 rows"
                )
                self.out_channels = sk.shape[0]
                self.in_channels = sk.shape[1]
            else:
                assert sk.shape[3] == 2, f"sub-kernel {i}: must have 2 columns"
                assert sk.shape[2] == 2 * self.hexbase_size + 1 - i, (
                    f"sub-kernel {i}: must have 2*(kernel size)+1-{i} rows"
                )

    def call(self, input):
        if self.hexbase_stride == 1:
            return self.operation_with_single_hexbase_stride(input)
        return self.operation_with_arbitrary_stride(input)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sub_kernels": [sk.tolist() for sk in self._sub_kernels_init],
                "strides": self.hexbase_stride,
                "bias": None if self._bias_init is None else self._bias_init.tolist(),
                "trainable": self._trainable_kernels,
                "debug": self._debug,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="hexagdly_tf")
class Conv3d_CustomKernel(HexBase, keras.layers.Layer):
    """3D hexagonal convolution with caller-supplied sub-kernels.

    ``sub_kernels`` use the PyTorch hexagdly layout ``(out, in, depth, rows,
    cols)``; transposed to ``(depth, rows, cols, in, out)`` internally.

    Precondition: the spatial dimensions (``D``, ``H``, ``W``) must be statically
    known when the layer is called -- the hex addressing arithmetic reads them as
    ints.
    """

    def __init__(
        self,
        sub_kernels=None,
        strides=1,
        bias=None,
        trainable=False,
        debug=False,
        name=None,
        **kwargs,
    ):
        # Note: 'bias' here is the actual bias array (data), not a boolean
        # flag like Conv3d's -- it has no Keras-native rename, unlike stride.
        _require_positive_int_or_pair("strides", strides)
        keras.layers.Layer.__init__(self, name=name, **kwargs)
        HexBase.__init__(self)
        self.dimensions = 3
        self.process = _ConvProcess(3)
        self.combine = keras.ops.add
        self.strides = strides
        self._stride_init = strides
        self._trainable_kernels = trainable
        self._debug = debug
        if isinstance(strides, int):
            self.hexbase_stride = strides
            self.depth_stride = strides
        else:
            assert len(strides) == 2, "Need a tuple of two ints to set strides"
            self.hexbase_stride = strides[1]
            self.depth_stride = strides[0]

        # np.asarray here (not just list(...)): see the matching comment in
        # Conv2d_CustomKernel.__init__ -- config round-trips supply plain lists.
        sub_kernels = (
            [np.asarray(sk, dtype=np.float32) for sk in sub_kernels] if sub_kernels else []
        )
        if debug or len(sub_kernels) == 0:
            sub_kernels = [
                np.array([[[[[1], [1], [1]]]]]),
                np.array([[[[[1, 1], [1, 1]]]]]),
            ]
        self.hexbase_size = len(sub_kernels) - 1
        self._check_sub_kernels(sub_kernels)
        # Keep the construction sub-kernels (PyTorch layout) + bias for get_config;
        # actual weight *values* are restored separately by Keras on load.
        self._sub_kernels_init = [np.asarray(sk, dtype=np.float32) for sk in sub_kernels]
        self._bias_init = None if bias is None else np.asarray(bias, dtype=np.float32)
        for i, sk in enumerate(sub_kernels):
            # (out, in, depth, rows, cols) -> (depth, rows, cols, in, out)
            tf_k = np.transpose(np.asarray(sk, dtype=np.float32), (2, 3, 4, 1, 0))
            setattr(
                self,
                "kernel" + str(i),
                keras.Variable(tf_k, trainable=trainable, name=f"kernel{i}"),
            )

        if not debug and bias is not None:
            self.bias_tensor = keras.Variable(
                np.asarray(bias, dtype=np.float32), trainable=trainable, name="bias"
            )
            self.kwargs = {"bias": self.bias_tensor}
            self.use_bias = True
        else:
            self.use_bias = False
            self.kwargs = {"bias": None}

    def _check_sub_kernels(self, sub_kernels):
        for i, sk in enumerate(sub_kernels):
            assert sk.ndim == 5, "sub-kernels must be rank 5 for a 3d convolution"
            if i == 0:
                assert sk.shape[4] == 1, "first sub-kernel must have only 1 column"
                assert sk.shape[3] == 2 * self.hexbase_size + 1, (
                    "first sub-kernel must have 2*(kernel size)+1 rows"
                )
                self.out_channels = sk.shape[0]
                self.in_channels = sk.shape[1]
                self.depth_size = sk.shape[2]
            else:
                assert sk.shape[4] == 2, f"sub-kernel {i}: must have 2 columns"
                assert sk.shape[3] == 2 * self.hexbase_size + 1 - i, (
                    f"sub-kernel {i}: must have 2*(kernel size)+1-{i} rows"
                )
                assert sk.shape[2] == self.depth_size, f"sub-kernel {i}: depths are not consistent"

    def call(self, input):
        if self.hexbase_stride == 1:
            return self.operation_with_single_hexbase_stride(input)
        return self.operation_with_arbitrary_stride(input)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sub_kernels": [sk.tolist() for sk in self._sub_kernels_init],
                "strides": self._stride_init,
                "bias": None if self._bias_init is None else self._bias_init.tolist(),
                "trainable": self._trainable_kernels,
                "debug": self._debug,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="hexagdly_tf")
class MaxPool2d(HexBase, keras.layers.Layer):
    """2D hexagonal max pooling (Keras port of hexagdly.MaxPool2d). Channels-last.

    Precondition: the spatial dimensions (``H``, ``W``) must be statically known
    when the layer is called -- the hex addressing arithmetic reads them as ints.

    The window size can be passed as ``pool_size`` (the keras.layers.MaxPool2D
    name) or ``kernel_size`` (the HexagDLy name). ``padding`` only accepts
    ``"same"``, as in :class:`Conv2d`. The arguments come in the
    keras.layers.MaxPool2D order -- ``pool_size, strides, padding, name`` --
    so Keras-style positional calls mean the same thing here.
    """

    def __init__(
        self, pool_size=None, strides=1, padding="same", name=None, kernel_size=None, **kwargs
    ):
        pool_size = _resolve_pool_size(pool_size, kernel_size)
        _require_positive_int("pool_size", pool_size)
        _require_positive_int("strides", strides)
        keras.layers.Layer.__init__(self, name=name, **kwargs)
        HexBase.__init__(self)
        self.pool_size = pool_size
        self.kernel_size = pool_size
        self.strides = strides
        self.padding = _require_same_padding(padding)
        self.hexbase_size = pool_size
        self.hexbase_stride = strides
        self.dimensions = 2
        self.process = _MaxPoolProcess(2)
        self.combine = keras.ops.maximum
        for i in range(self.hexbase_size + 1):
            setattr(self, "kernel" + str(i), (1 + 2 * self.hexbase_size - i, 1 if i == 0 else 2))

    def call(self, input):
        if self.hexbase_stride == 1:
            return self.operation_with_single_hexbase_stride(input)
        return self.operation_with_arbitrary_stride(input)

    def get_config(self):
        config = super().get_config()
        config.update(
            {"pool_size": self.pool_size, "strides": self.strides, "padding": self.padding}
        )
        return config


@keras.saving.register_keras_serializable(package="hexagdly_tf")
class AvgPool2d(HexBase, keras.layers.Layer):
    """2D hexagonal average pooling. Channels-last.

    Each output cell is the mean over the cells of its hexagonal window (the
    same window as :class:`MaxPool2d`) that lie inside the grid. Positions past
    the grid border count in neither the sum nor the divisor, like
    ``keras.layers.AveragePooling2D(padding="same")``. Cells inside the grid
    always count, including empty (zero) ones such as the cells around a
    mapped camera image -- the same convention :class:`Conv2d` and
    :class:`MaxPool2d` apply to them.

    Arguments match :class:`MaxPool2d`. Not exportable to hls4ml yet:
    ``patch_model_for_hls`` raises on it.
    """

    def __init__(
        self, pool_size=None, strides=1, padding="same", name=None, kernel_size=None, **kwargs
    ):
        pool_size = _resolve_pool_size(pool_size, kernel_size)
        _require_positive_int("pool_size", pool_size)
        _require_positive_int("strides", strides)
        keras.layers.Layer.__init__(self, name=name, **kwargs)
        HexBase.__init__(self)
        self.pool_size = pool_size
        self.kernel_size = pool_size
        self.strides = strides
        self.padding = _require_same_padding(padding)
        self.hexbase_size = pool_size
        self.hexbase_stride = strides
        self.dimensions = 2
        self.process = _SumPoolProcess(2)
        self.combine = operator.add  # see _SumPoolProcess for why not keras.ops.add
        for i in range(self.hexbase_size + 1):
            setattr(self, "kernel" + str(i), (1 + 2 * self.hexbase_size - i, 1 if i == 0 else 2))

    def _window_sum(self, input):
        if self.hexbase_stride == 1:
            return self.operation_with_single_hexbase_stride(input)
        return self.operation_with_arbitrary_stride(input)

    def call(self, input):
        # Summing a tensor of ones over the same windows counts the in-grid
        # cells of each window, since the hex padding is zeros. The window
        # centre is always in the grid, so the count is never zero.
        total = self._window_sum(input)
        count = self._window_sum(keras.ops.ones_like(input[..., :1]))
        return total / count

    def get_config(self):
        config = super().get_config()
        config.update(
            {"pool_size": self.pool_size, "strides": self.strides, "padding": self.padding}
        )
        return config


@keras.saving.register_keras_serializable(package="hexagdly_tf")
class MaxPool3d(HexBase, keras.layers.Layer):
    """3D hexagonal max pooling (Keras port of hexagdly.MaxPool3d). Channels-last.

    Precondition: the spatial dimensions (``D``, ``H``, ``W``) must be statically
    known when the layer is called -- the hex addressing arithmetic reads them as
    ints.

    The window size can be passed as ``pool_size`` (the Keras name) or
    ``kernel_size`` (the HexagDLy name).
    """

    def __init__(self, pool_size=None, strides=1, name=None, kernel_size=None, **kwargs):
        kernel_size = _resolve_pool_size(pool_size, kernel_size)
        _require_positive_int_or_pair("pool_size", kernel_size)
        _require_positive_int_or_pair("strides", strides)
        keras.layers.Layer.__init__(self, name=name, **kwargs)
        HexBase.__init__(self)
        self.pool_size = kernel_size
        self.kernel_size = kernel_size
        self.strides = strides
        if isinstance(kernel_size, int):
            self.hexbase_size = kernel_size
            self.depth_size = kernel_size
        else:
            self.hexbase_size = kernel_size[1]
            self.depth_size = kernel_size[0]
        if isinstance(strides, int):
            self.hexbase_stride = strides
            self.depth_stride = strides
        else:
            self.hexbase_stride = strides[1]
            self.depth_stride = strides[0]
        self.dimensions = 3
        self.process = _MaxPoolProcess(3)
        self.combine = keras.ops.maximum
        for i in range(self.hexbase_size + 1):
            setattr(
                self,
                "kernel" + str(i),
                (self.depth_size, 1 + 2 * self.hexbase_size - i, 1 if i == 0 else 2),
            )

    def call(self, input):
        if self.hexbase_stride == 1:
            return self.operation_with_single_hexbase_stride(input)
        return self.operation_with_arbitrary_stride(input)

    def get_config(self):
        config = super().get_config()
        config.update({"pool_size": self.pool_size, "strides": self.strides})
        return config
