"""Cross-checks every keras_hexagdly layer against the upstream PyTorch
HexagDLy PyPI package (`pip install hexagdly`), bit-for-bit: same random
input, weights copied across frameworks (accounting for NCHW<->NHWC and
kernel-order), outputs compared. This is the oracle that proves the Keras
port faithful to the original.

Requires both `torch` and `hexagdly` (the upstream package, NOT this
repository) to be installed -- see the `dev` extra. Tests are skipped if
either is missing.

`share_neighbors` and `depth_padding` are NEW functionalities added in this
port and have no equivalent in upstream HexagDLy, so they are not covered
here -- see test_share_neighbors.py and test_depth_padding.py instead.
"""

import keras
import numpy as np
import pytest

import keras_hexagdly as hgly

torch = pytest.importorskip("torch")
hexagdly = pytest.importorskip("hexagdly")

RTOL, ATOL = 1e-4, 1e-4


@pytest.fixture(autouse=True)
def _force_cpu():
    """The upstream PyTorch oracle (`tl`) always runs on CPU (we never call
    .cuda() on it). On a machine with a GPU and the torch Keras backend, our
    own layers would otherwise run on GPU by default, and GPU vs. CPU float32
    reduction order alone produces ~1e-3-level differences that look like a
    correctness bug but aren't one. Force CPU placement so this file tests
    algorithm correctness, not GPU-vs-CPU numerics."""
    with keras.device("cpu"):
        yield


def _to_nhwc(x_nchw):
    return np.transpose(x_nchw, (0, 2, 3, 1))


def _to_ndhwc(x_ncdhw):
    return np.transpose(x_ncdhw, (0, 2, 3, 4, 1))


def _copy_conv_weights_2d(torch_layer, keras_layer):
    for i in range(torch_layer.hexbase_size + 1):
        w = getattr(torch_layer, f"kernel{i}").detach().numpy()  # (out,in,kh,kw)
        keras_layer._base_kernels[i].assign(np.transpose(w, (2, 3, 1, 0)))
    if getattr(torch_layer, "bias", False):
        keras_layer.bias_tensor.assign(torch_layer.kwargs["bias"].detach().numpy())


def _copy_conv_weights_3d(torch_layer, keras_layer):
    for i in range(torch_layer.hexbase_size + 1):
        w = getattr(torch_layer, f"kernel{i}").detach().numpy()  # (out,in,d,kh,kw)
        keras_layer._base_kernels[i].assign(np.transpose(w, (2, 3, 4, 1, 0)))
    if getattr(torch_layer, "bias", False):
        keras_layer.bias_tensor.assign(torch_layer.kwargs["bias"].detach().numpy())


def _assert_close(torch_out_nchw, keras_out_nhwc, dims):
    t = torch_out_nchw.detach().cpu().numpy()
    k = keras.ops.convert_to_numpy(keras_out_nhwc)
    k = np.transpose(k, (0, 3, 1, 2)) if dims == 2 else np.transpose(k, (0, 4, 1, 2, 3))
    assert t.shape == k.shape
    np.testing.assert_allclose(t, k, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("kernel_size", [1, 2, 3])
@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("H,W,Cin,Cout", [(9, 8, 2, 3), (12, 7, 1, 4)])
def test_conv2d(kernel_size, stride, H, W, Cin, Cout):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, Cin, H, W)).astype(np.float32)
    tl = hexagdly.Conv2d(Cin, Cout, kernel_size=kernel_size, stride=stride, bias=True)
    kl = hgly.Conv2d(Cin, Cout, kernel_size=kernel_size, stride=stride, bias=True)
    _ = kl(keras.ops.zeros((1, H, W, Cin)))
    _copy_conv_weights_2d(tl, kl)
    to = tl(torch.from_numpy(x))
    ko = kl(keras.ops.convert_to_tensor(_to_nhwc(x)))
    _assert_close(to, ko, 2)


@pytest.mark.parametrize("kernel_size", [1, 2])
@pytest.mark.parametrize("stride", [1, 2])
def test_conv3d(kernel_size, stride):
    rng = np.random.default_rng(1)
    D, H, W, Cin, Cout = 5, 9, 8, 2, 3
    x = rng.standard_normal((2, Cin, D, H, W)).astype(np.float32)
    tl = hexagdly.Conv3d(Cin, Cout, kernel_size=kernel_size, stride=stride, bias=True)
    kl = hgly.Conv3d(Cin, Cout, kernel_size=kernel_size, stride=stride, bias=True)
    _ = kl(keras.ops.zeros((1, D, H, W, Cin)))
    _copy_conv_weights_3d(tl, kl)
    to = tl(torch.from_numpy(x))
    ko = kl(keras.ops.convert_to_tensor(_to_ndhwc(x)))
    _assert_close(to, ko, 3)


@pytest.mark.parametrize("kernel_size", [1, 2])
@pytest.mark.parametrize("stride", [1, 2])
def test_maxpool2d(kernel_size, stride):
    rng = np.random.default_rng(2)
    H, W, C = 10, 9, 3
    x = rng.standard_normal((2, C, H, W)).astype(np.float32)
    tl = hexagdly.MaxPool2d(kernel_size=kernel_size, stride=stride)
    kl = hgly.MaxPool2d(kernel_size=kernel_size, stride=stride)
    to = tl(torch.from_numpy(x))
    ko = kl(keras.ops.convert_to_tensor(_to_nhwc(x)))
    _assert_close(to, ko, 2)


@pytest.mark.parametrize("kernel_size", [1, 2])
@pytest.mark.parametrize("stride", [1, 2])
def test_maxpool3d(kernel_size, stride):
    rng = np.random.default_rng(14)
    D, H, W, C = 6, 10, 9, 3
    x = rng.standard_normal((2, C, D, H, W)).astype(np.float32)
    tl = hexagdly.MaxPool3d(kernel_size=kernel_size, stride=stride)
    kl = hgly.MaxPool3d(kernel_size=kernel_size, stride=stride)
    to = tl(torch.from_numpy(x))
    ko = kl(keras.ops.convert_to_tensor(_to_ndhwc(x)))
    _assert_close(to, ko, 3)


def test_custom_kernel_2d():
    rng = np.random.default_rng(3)
    n = 2
    subk = []
    for i in range(n + 1):
        kh = 2 * n + 1 - i
        kw = 1 if i == 0 else 2
        subk.append(rng.standard_normal((3, 2, kh, kw)).astype(np.float32))
    H, W = 11, 9
    x = rng.standard_normal((2, 2, H, W)).astype(np.float32)
    tl = hexagdly.Conv2d_CustomKernel(sub_kernels=[s.copy() for s in subk], stride=1)
    kl = hgly.Conv2d_CustomKernel(sub_kernels=[s.copy() for s in subk], stride=1)
    to = tl(torch.from_numpy(x))
    ko = kl(keras.ops.convert_to_tensor(_to_nhwc(x)))
    _assert_close(to, ko, 2)


def test_custom_kernel_3d():
    rng = np.random.default_rng(16)
    n, depth = 2, 3
    subk = []
    for i in range(n + 1):
        kh = 2 * n + 1 - i
        kw = 1 if i == 0 else 2
        subk.append(rng.standard_normal((3, 2, depth, kh, kw)).astype(np.float32))
    D, H, W = 5, 11, 9
    x = rng.standard_normal((2, 2, D, H, W)).astype(np.float32)
    tl = hexagdly.Conv3d_CustomKernel(sub_kernels=[s.copy() for s in subk], stride=1)
    kl = hgly.Conv3d_CustomKernel(sub_kernels=[s.copy() for s in subk], stride=1)
    to = tl(torch.from_numpy(x))
    ko = kl(keras.ops.convert_to_tensor(_to_ndhwc(x)))
    _assert_close(to, ko, 3)


@pytest.mark.parametrize("kernel_size", [1, 2, 3, 4])
@pytest.mark.parametrize(
    "H,W", [(9, 9), (9, 10), (8, 7), (8, 8), (13, 5), (5, 13), (7, 7), (15, 14)]
)
def test_conv2d_odd_even_widths(kernel_size, H, W):
    """The odd/even-column logic is the subtlest part: test both parities and
    several heights, including the awkward small/large kernel-vs-size cases."""
    if 2 * kernel_size + 1 > H:  # kernel taller than input: ill-defined
        pytest.skip("kernel taller than input")
    rng = np.random.default_rng(11)
    Cin, Cout = 2, 2
    x = rng.standard_normal((1, Cin, H, W)).astype(np.float32)
    tl = hexagdly.Conv2d(Cin, Cout, kernel_size=kernel_size, stride=1, bias=True)
    kl = hgly.Conv2d(Cin, Cout, kernel_size=kernel_size, stride=1, bias=True)
    _ = kl(keras.ops.zeros((1, H, W, Cin)))
    _copy_conv_weights_2d(tl, kl)
    to = tl(torch.from_numpy(x))
    ko = kl(keras.ops.convert_to_tensor(_to_nhwc(x)))
    _assert_close(to, ko, 2)


@pytest.mark.parametrize("kernel_size", [1, 2, 3])
@pytest.mark.parametrize("stride", [1, 2, 3, 4, 5])
def test_conv2d_large_strides(kernel_size, stride):
    rng = np.random.default_rng(12)
    H, W, Cin, Cout = 20, 17, 2, 3
    x = rng.standard_normal((2, Cin, H, W)).astype(np.float32)
    tl = hexagdly.Conv2d(Cin, Cout, kernel_size=kernel_size, stride=stride, bias=True)
    kl = hgly.Conv2d(Cin, Cout, kernel_size=kernel_size, stride=stride, bias=True)
    _ = kl(keras.ops.zeros((1, H, W, Cin)))
    _copy_conv_weights_2d(tl, kl)
    to = tl(torch.from_numpy(x))
    ko = kl(keras.ops.convert_to_tensor(_to_nhwc(x)))
    _assert_close(to, ko, 2)


@pytest.mark.parametrize("kd,kh", [(1, 2), (3, 1), (2, 2)])
@pytest.mark.parametrize("sd,sh", [(1, 1), (2, 1), (1, 2)])
def test_conv3d_asymmetric(kd, kh, sd, sh):
    """Depth vs hex-base kernel/stride differ -- exercises the (depth, hex)
    tuple handling that a symmetric int can't catch."""
    rng = np.random.default_rng(13)
    D, H, W, Cin, Cout = 7, 9, 8, 2, 3
    x = rng.standard_normal((2, Cin, D, H, W)).astype(np.float32)
    tl = hexagdly.Conv3d(Cin, Cout, kernel_size=(kd, kh), stride=(sd, sh), bias=True)
    kl = hgly.Conv3d(Cin, Cout, kernel_size=(kd, kh), stride=(sd, sh), bias=True)
    _ = kl(keras.ops.zeros((1, D, H, W, Cin)))
    _copy_conv_weights_3d(tl, kl)
    to = tl(torch.from_numpy(x))
    ko = kl(keras.ops.convert_to_tensor(_to_ndhwc(x)))
    _assert_close(to, ko, 3)


def test_gradients_2d():
    """Training equivalence: gradients wrt each sub-kernel and bias must match
    PyTorch's, so a model trains identically, not just infers identically."""
    if keras.backend.backend() != "tensorflow":
        pytest.skip("uses tf.GradientTape directly; only meaningful on the tensorflow backend")
    import tensorflow as tf

    rng = np.random.default_rng(17)
    for kernel_size in (1, 2, 3):
        H, W, Cin, Cout = 9, 8, 2, 3
        x = rng.standard_normal((2, Cin, H, W)).astype(np.float32)

        tl = hexagdly.Conv2d(Cin, Cout, kernel_size=kernel_size, stride=1, bias=True)
        kl = hgly.Conv2d(Cin, Cout, kernel_size=kernel_size, stride=1, bias=True)
        _ = kl(keras.ops.zeros((1, H, W, Cin)))
        _copy_conv_weights_2d(tl, kl)

        out_t = tl(torch.from_numpy(x))
        out_t.sum().backward()

        xk = keras.ops.convert_to_tensor(_to_nhwc(x))
        with tf.GradientTape() as tape:
            out_k = kl(xk)
            loss_k = tf.reduce_sum(out_k)
        kvars = [getattr(kl, f"kernel{i}") for i in range(kernel_size + 1)] + [kl.bias_tensor]
        grads_k = tape.gradient(loss_k, kvars)

        for i in range(kernel_size + 1):
            gt = getattr(tl, f"kernel{i}").grad.detach().cpu().numpy()  # (out,in,kh,kw)
            gk = np.transpose(grads_k[i].numpy(), (3, 2, 0, 1))  # ->(out,in,kh,kw)
            np.testing.assert_allclose(gt, gk, rtol=RTOL, atol=ATOL)
        gbt = tl.kwargs["bias"].grad.detach().cpu().numpy()
        gbk = grads_k[-1].numpy()
        np.testing.assert_allclose(gbt, gbk, rtol=RTOL, atol=ATOL)


def test_gradients_3d():
    """Train-time equivalence for the 3D path (separate transpose/backprop)."""
    if keras.backend.backend() != "tensorflow":
        pytest.skip("uses tf.GradientTape directly; only meaningful on the tensorflow backend")
    import tensorflow as tf

    rng = np.random.default_rng(20)
    for kernel_size in (1, 2):
        D, H, W, Cin, Cout = 5, 9, 8, 2, 3
        x = rng.standard_normal((2, Cin, D, H, W)).astype(np.float32)
        tl = hexagdly.Conv3d(Cin, Cout, kernel_size=kernel_size, stride=1, bias=True)
        kl = hgly.Conv3d(Cin, Cout, kernel_size=kernel_size, stride=1, bias=True)
        _ = kl(keras.ops.zeros((1, D, H, W, Cin)))
        _copy_conv_weights_3d(tl, kl)

        tl(torch.from_numpy(x)).sum().backward()
        xk = keras.ops.convert_to_tensor(_to_ndhwc(x))
        with tf.GradientTape() as tape:
            loss_k = tf.reduce_sum(kl(xk))
        kvars = [getattr(kl, f"kernel{i}") for i in range(kernel_size + 1)] + [kl.bias_tensor]
        grads_k = tape.gradient(loss_k, kvars)

        for i in range(kernel_size + 1):
            gt = getattr(tl, f"kernel{i}").grad.detach().cpu().numpy()  # (out,in,d,h,w)
            gk = np.transpose(grads_k[i].numpy(), (4, 3, 0, 1, 2))  # ->(out,in,d,h,w)
            np.testing.assert_allclose(gt, gk, rtol=RTOL, atol=ATOL)
        gbt = tl.kwargs["bias"].grad.detach().cpu().numpy()
        gbk = grads_k[-1].numpy()
        np.testing.assert_allclose(gbt, gbk, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("kernel_size", [1, 2, 3])
@pytest.mark.parametrize("stride", [1, 2])
def test_conv2d_no_bias(kernel_size, stride):
    rng = np.random.default_rng(10)
    H, W, Cin, Cout = 10, 9, 2, 3
    x = rng.standard_normal((2, Cin, H, W)).astype(np.float32)
    tl = hexagdly.Conv2d(Cin, Cout, kernel_size=kernel_size, stride=stride, bias=False)
    kl = hgly.Conv2d(Cin, Cout, kernel_size=kernel_size, stride=stride, bias=False)
    _ = kl(keras.ops.zeros((1, H, W, Cin)))
    _copy_conv_weights_2d(tl, kl)
    to = tl(torch.from_numpy(x))
    ko = kl(keras.ops.convert_to_tensor(_to_nhwc(x)))
    _assert_close(to, ko, 2)


@pytest.mark.parametrize("kernel_size", [1, 2])
def test_conv3d_no_bias(kernel_size):
    rng = np.random.default_rng(24)
    D, H, W, Cin, Cout = 5, 9, 8, 2, 3
    x = rng.standard_normal((2, Cin, D, H, W)).astype(np.float32)
    tl = hexagdly.Conv3d(Cin, Cout, kernel_size=kernel_size, stride=1, bias=False)
    kl = hgly.Conv3d(Cin, Cout, kernel_size=kernel_size, stride=1, bias=False)
    _ = kl(keras.ops.zeros((1, D, H, W, Cin)))
    _copy_conv_weights_3d(tl, kl)
    to = tl(torch.from_numpy(x))
    ko = kl(keras.ops.convert_to_tensor(_to_ndhwc(x)))
    _assert_close(to, ko, 3)


@pytest.mark.parametrize("Cin,Cout", [(8, 16), (16, 4), (1, 32)])
def test_many_channels(Cin, Cout):
    rng = np.random.default_rng(25)
    H, W = 10, 9
    x = rng.standard_normal((2, Cin, H, W)).astype(np.float32)
    tl = hexagdly.Conv2d(Cin, Cout, kernel_size=2, stride=1, bias=True)
    kl = hgly.Conv2d(Cin, Cout, kernel_size=2, stride=1, bias=True)
    _ = kl(keras.ops.zeros((1, H, W, Cin)))
    _copy_conv_weights_2d(tl, kl)
    to = tl(torch.from_numpy(x))
    ko = kl(keras.ops.convert_to_tensor(_to_nhwc(x)))
    _assert_close(to, ko, 2)


@pytest.mark.parametrize("stride", [2, 3])
def test_custom_kernel_2d_strided(stride):
    rng = np.random.default_rng(21)
    n = 2
    subk = []
    for i in range(n + 1):
        kh = 2 * n + 1 - i
        kw = 1 if i == 0 else 2
        subk.append(rng.standard_normal((3, 2, kh, kw)).astype(np.float32))
    H, W = 15, 13
    x = rng.standard_normal((2, 2, H, W)).astype(np.float32)
    tl = hexagdly.Conv2d_CustomKernel(sub_kernels=[s.copy() for s in subk], stride=stride)
    kl = hgly.Conv2d_CustomKernel(sub_kernels=[s.copy() for s in subk], stride=stride)
    to = tl(torch.from_numpy(x))
    ko = kl(keras.ops.convert_to_tensor(_to_nhwc(x)))
    _assert_close(to, ko, 2)


def test_batch_independence():
    """A batched call must equal stacking per-sample calls (no cross-sample leak)."""
    rng = np.random.default_rng(22)
    H, W, Cin, Cout = 11, 9, 2, 3
    kl = hgly.Conv2d(Cin, Cout, kernel_size=2, stride=1, bias=True)
    _ = kl(keras.ops.zeros((1, H, W, Cin)))
    x = rng.standard_normal((4, H, W, Cin)).astype(np.float32)
    batched = keras.ops.convert_to_numpy(kl(keras.ops.convert_to_tensor(x)))
    per_sample = np.concatenate(
        [
            keras.ops.convert_to_numpy(kl(keras.ops.convert_to_tensor(x[k : k + 1])))
            for k in range(4)
        ],
        axis=0,
    )
    np.testing.assert_allclose(batched, per_sample, rtol=RTOL, atol=ATOL)


# ----------------------------------------------------------------------------
# Forward-compatibility: if a FUTURE release of upstream PyTorch HexagDLy adds
# the same functionalities we introduced (share_neighbors, depth_padding),
# automatically cross-check against it instead of silently staying blind to
# it. These are no-ops (skipped) against every hexagdly release at the time
# of writing (2.0.2), which has neither.
# ----------------------------------------------------------------------------


def test_share_neighbors_forward_compat_with_upstream():
    import inspect

    sig = inspect.signature(hexagdly.Conv2d.__init__)
    if "share_neighbors" not in sig.parameters:
        pytest.skip("upstream hexagdly does not (yet) support share_neighbors")

    rng = np.random.default_rng(50)
    n, H, W, Cin, Cout = 2, 13, 11, 2, 3
    x = rng.standard_normal((2, Cin, H, W)).astype(np.float32)
    tl = hexagdly.Conv2d(Cin, Cout, kernel_size=n, stride=1, bias=True, share_neighbors=True)
    kl = hgly.Conv2d(Cin, Cout, kernel_size=n, stride=1, bias=True, share_neighbors=True)
    _ = kl(keras.ops.zeros((1, H, W, Cin)))
    rw = tl.ring_weights.detach().numpy()  # assumed (out,in,nr), our own torch-side convention
    kl.ring_weights.assign(np.transpose(rw, (2, 1, 0)))
    kl.bias_tensor.assign(tl.kwargs["bias"].detach().numpy())
    to = tl(torch.from_numpy(x))
    ko = kl(keras.ops.convert_to_tensor(_to_nhwc(x)))
    _assert_close(to, ko, 2)


def test_depth_padding_forward_compat_with_upstream():
    import inspect

    sig = inspect.signature(hexagdly.Conv3d.__init__)
    if "depth_padding" not in sig.parameters:
        pytest.skip("upstream hexagdly does not (yet) support depth_padding")

    rng = np.random.default_rng(51)
    D, kd, n, Cin, Cout, H, W = 9, 5, 1, 2, 3, 9, 8
    x = rng.standard_normal((2, Cin, D, H, W)).astype(np.float32)
    tl = hexagdly.Conv3d(Cin, Cout, kernel_size=(kd, n), stride=1, bias=True, depth_padding="same")
    kl = hgly.Conv3d(Cin, Cout, kernel_size=(kd, n), stride=1, bias=True, depth_padding="same")
    _ = kl(keras.ops.zeros((1, D, H, W, Cin)))
    _copy_conv_weights_3d(tl, kl)
    to = tl(torch.from_numpy(x))
    ko = kl(keras.ops.convert_to_tensor(_to_ndhwc(x)))
    assert to.shape[2] == D
    _assert_close(to, ko, 3)
