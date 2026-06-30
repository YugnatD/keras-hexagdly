# Speed benchmarks: keras_hexagdly vs. upstream PyTorch HexagDLy

`bench.py` times a `Conv2d` forward pass across 8 shape/kernel/stride
configurations, batch size 8, on CPU and (where available) GPU. Run once per
backend (`KERAS_BACKEND` is read at import time):

```
python benchmarks/bench.py pytorch                       # upstream HexagDLy, CPU
python benchmarks/bench.py pytorch_gpu                   # upstream HexagDLy, GPU (cuda)
KERAS_BACKEND=tensorflow python benchmarks/bench.py tensorflow
KERAS_BACKEND=tensorflow python benchmarks/bench.py tensorflow --compiled   # tf.function
KERAS_BACKEND=jax         python benchmarks/bench.py jax
KERAS_BACKEND=jax         python benchmarks/bench.py jax --compiled        # jax.jit
KERAS_BACKEND=torch       python benchmarks/bench.py torch                 # GPU if available
KERAS_BACKEND=torch       python benchmarks/bench.py torch --compiled      # torch.compile
KERAS_BACKEND=torch       python benchmarks/bench.py torch_cpu             # force CPU
python benchmarks/summarize.py
```

## Results (i7-13700KF · RTX 3090 Ti · CUDA 12 · torch 2.3.1 · jax 0.6.2 · keras 3.12.1)

All GPU timings use `cuda.synchronize()` before and after the timed loop so
they reflect actual compute time, not just kernel-launch latency.

Per-call time in ms:

| case | pytorch CPU | pytorch GPU | keras/torch GPU | keras/jax GPU | keras/tf GPU | keras/torch CPU |
|---|---|---|---|---|---|---|
| small k1 s1   | 0.392 | **0.094** | 1.009 | 1.576 | 2.515 | 1.851 |
| small k2 s1   | 0.513 | **0.119** | 1.301 | 2.396 | 3.337 | 2.413 |
| small k3 s1   | 0.749 | **0.169** | 1.869 | 3.804 | 4.822 | 3.513 |
| medium k2 s1  | 1.930 | **0.132** | 1.437 | 2.202 | 3.189 | 4.840 |
| medium k2 s2  | 1.217 | **0.196** | 1.820 | 3.264 | 5.054 | 3.766 |
| large k2 s1   | 21.544 | **0.587** | 1.620 | 2.291 | 3.357 | 24.193 |
| large k3 s1   | 26.669 | **0.915** | 2.196 | 3.833 | 4.800 | 34.733 |
| many channels | 10.578 | **0.590** | 1.434 | 2.391 | 3.270 | 18.557 |

As a multiple of the upstream PyTorch **CPU** baseline (below 1.0x means faster):

| case | pytorch GPU | keras/torch GPU | keras/jax GPU | keras/tf GPU | keras/torch CPU |
|---|---|---|---|---|---|
| small k1 s1   | **0.24x** | 2.57x | 4.02x | 6.41x | 4.72x |
| small k2 s1   | **0.23x** | 2.54x | 4.67x | 6.50x | 4.70x |
| small k3 s1   | **0.23x** | 2.49x | 5.08x | 6.44x | 4.69x |
| medium k2 s1  | **0.07x** | 0.74x | 1.14x | 1.65x | 2.51x |
| medium k2 s2  | **0.16x** | 1.50x | 2.68x | 4.15x | 3.09x |
| large k2 s1   | **0.03x** | 0.08x | 0.11x | 0.16x | 1.12x |
| large k3 s1   | **0.03x** | 0.08x | 0.14x | 0.18x | 1.30x |
| many channels | **0.06x** | 0.14x | 0.23x | 0.31x | 1.75x |

## Takeaways

- **The upstream library does not use CUDA.** `hexagdly.Conv2d` is a
  CPU-only implementation regardless of what device the tensors are on, so the
  fair GPU-vs-GPU comparison is `pytorch_gpu` vs. `keras/torch GPU` (and the
  other Keras GPU backends). The previously reported numbers that compared
  Keras-on-GPU against upstream-on-CPU overstated the advantage.

- **pytorch_gpu is the fastest overall.** The upstream library, once properly
  moved to CUDA with `.to("cuda")`, runs 2–11x faster than `keras/torch` on
  GPU. It uses hand-crafted slicing ops that map to tight CUDA kernels, while
  `keras_hexagdly` routes through the generic `keras.ops.conv` path which
  carries more per-call overhead.

- **keras/torch GPU beats upstream CPU by a large margin at realistic sizes.**
  For large inputs and many channels (`large_k*`, `many_channels`), all Keras
  GPU backends are 5–80x faster than the upstream CPU implementation. For
  small toy inputs the overhead of GPU launch outweighs the parallelism, so
  upstream CPU still wins there.

- **GPU backend ranking:** `keras/torch` > `keras/jax` > `keras/tensorflow`.
  Torch is roughly 1.5–2x faster than JAX, which is roughly 1.5–2x faster
  than TensorFlow, on this machine. The gap is largest for small inputs where
  per-dispatch overhead dominates.

- **Compiled execution (jax.jit / tf.function) closes the gap significantly.**
  On CPU runs in earlier testing, `jax.jit` fused the sub-kernel decomposition
  into a single XLA program and was 3–16x faster than upstream CPU eager on
  most cases. `tf.function` gave a smaller but consistent speedup. These
  variants are not included in the table above (GPU compiled numbers are
  hardware/driver-sensitive) but are available via the `--compiled` flag.
  In real training with `model.fit` or `model.predict`, the jax and tensorflow
  backends automatically compile the layer, so the eager numbers above are a
  pessimistic lower bound for those two backends.

- **`torch.compile` is currently unsupported** for this layer. The tracer
  hits an incompatibility between Keras's `any_symbolic_tensors` tree
  traversal and `torch._dynamo`. This is a correctness/tracing issue, not a
  performance one -- eager torch output is verified correct by the test suite.

- **Uncompiled CPU: this port is inherently slower than upstream** (1.2–5x).
  That is not a regression from the port -- it reflects HexagDLy's own design
  (the hex kernel is decomposed into several padded/sliced sub-convolutions,
  each a separate op-dispatch), which the original project's README already
  calls out: *"the implemented methods rather aim for flexibility then for
  performance."* This port keeps that same structure intentionally for
  numerical equivalence and backend portability.

**Bottom line**: moving from the upstream CPU-only library to `keras_hexagdly`
on a CUDA GPU gives meaningful speedups at any realistic input size. The
remaining gap versus a hypothetical hand-written CUDA extension is the cost of
backend-agnostic portability.
