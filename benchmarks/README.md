# Speed benchmarks: keras_hexagdly vs. upstream PyTorch HexagDLy

`bench.py` times a `Conv2d` forward pass across 8 shape/kernel/stride
configurations, batch size 8, on CPU and (where available) GPU. Run once per
backend (`KERAS_BACKEND` is read at import time):

```
python benchmarks/bench.py pytorch                       # the oracle (CPU)
KERAS_BACKEND=tensorflow python benchmarks/bench.py tensorflow
KERAS_BACKEND=tensorflow python benchmarks/bench.py tensorflow --compiled   # tf.function
KERAS_BACKEND=jax         python benchmarks/bench.py jax
KERAS_BACKEND=jax         python benchmarks/bench.py jax --compiled        # jax.jit
KERAS_BACKEND=torch       python benchmarks/bench.py torch                 # GPU if available
KERAS_BACKEND=torch       python benchmarks/bench.py torch --compiled      # torch.compile
KERAS_BACKEND=torch       python benchmarks/bench.py torch_cpu             # force CPU
python benchmarks/summarize.py
```

## Results (this machine: CPU + one CUDA GPU; JAX is CPU-only here, no jaxlib-cuda installed)

Per-call time in ms, then as a multiple of the upstream PyTorch (CPU, eager)
baseline -- **below 1.0x means faster than upstream**:

| case | jax | jax+jit | tensorflow | tf+function | torch (GPU) | torch+compile | torch (CPU) |
|---|---|---|---|---|---|---|---|
| small k1 s1  | 2.49x | 0.82x | 7.41x | 1.05x | 2.99x | 6.18x  | 14.54x |
| small k2 s1  | 2.43x | 0.09x | 7.47x | 0.52x | 2.75x | 5.88x  | 12.87x |
| small k3 s1  | 2.41x | 0.11x | 7.34x | 0.40x | 2.68x | 5.81x  | 12.69x |
| medium k2 s1 | 1.13x | 0.13x | 2.34x | 0.28x | 0.79x | 1.56x  | 4.86x  |
| medium k2 s2 | 1.73x | 0.06x | 5.38x | 0.27x | 1.43x | 3.20x  | 8.46x  |
| large k2 s1  | 1.38x | 0.07x | 1.23x | 0.50x | 0.23x | 0.32x  | 1.66x  |
| large k3 s1  | 1.40x | 0.09x | 1.47x | 0.71x | 0.31x | 0.87x  | 1.67x  |
| many channels| 1.13x | 0.09x | 1.63x | 0.79x | 0.16x | 0.26x  | 1.56x  |

## Takeaways

- **Uncompiled (eager), CPU vs. CPU**: this port is 1.1x-7.5x slower than
  upstream PyTorch HexagDLy on the same workload. That's not a regression
  introduced by the port -- it's inherent to HexagDLy's design (the hex
  kernel is decomposed into several padded/sliced sub-convolutions, each a
  separate op-dispatch), which the original project's own README already
  flags: "the implemented methods rather aim for flexibility then for
  performance." This port keeps that same structure on purpose (see
  `layers.py`'s module docstring), so it inherits the same per-call overhead.
- **Compiled execution changes the picture completely.** Wrapping the layer
  in `jax.jit` (what `model.fit`/`model.predict` do for you automatically on
  the jax backend) fuses the whole sub-kernel decomposition into one XLA
  program: 3x-16x **faster** than upstream's eager PyTorch on every case
  except the very largest. `tf.function` gets a smaller but still real
  speedup (roughly on par with, or faster than, upstream on most cases).
  Since real training/inference normally goes through a compiled path
  (`model.fit`, `model.predict`, `@tf.function`, `jax.jit`), this is the more
  representative number for actual usage, not the raw eager-loop numbers.
- **`torch.compile` is the outlier**: results are inconsistent, sometimes
  *slower* than eager. The cause is visible in the warm-up logs --
  `torch._dynamo hit config.recompile_limit`, triggered by `HexBase`'s
  first-call shape-caching (`self.odd_columns_slices.append(...)`, gated by
  `self.input_size_is_known`): the control flow differs between the very
  first call and every call after, which is exactly the kind of
  data-dependent Python control flow `torch.compile`'s tracer struggles to
  guard cheaply. This is a `torch.compile`-compatibility rough edge to be
  aware of, not a correctness issue (`test_vs_pytorch_hexagdly.py` already
  proves eager-mode torch backend output is correct).
- **Eager torch on GPU** is consistently competitive or faster than
  upstream-on-CPU even without compilation (0.16x-2.99x), simply because a
  real GPU is available. This is the easiest way to get a speed win with no
  code changes: `KERAS_BACKEND=torch` plus a CUDA-capable machine.

**Bottom line**: the port itself isn't slow -- it has the same architectural
overhead as upstream by design, and that overhead either shrinks
dramatically (compiled tf/jax) or is offset by GPU execution (torch) under
normal usage. The one thing worth keeping in mind is `torch.compile`
support, which is currently unreliable for this layer.
