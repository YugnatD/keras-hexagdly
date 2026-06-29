"""Forward-pass speed benchmark: keras_hexagdly (on each Keras 3 backend) vs
the upstream PyTorch hexagdly package.

Run once per backend (KERAS_BACKEND is read at import time, so each backend
needs its own process), then aggregate with summarize.py:

    KERAS_BACKEND=tensorflow python benchmarks/bench.py tensorflow
    KERAS_BACKEND=torch       python benchmarks/bench.py torch
    KERAS_BACKEND=torch       python benchmarks/bench.py torch_cpu   # force CPU
    KERAS_BACKEND=jax         python benchmarks/bench.py jax
                              python benchmarks/bench.py pytorch      # the oracle
    python benchmarks/summarize.py
"""

import json
import sys
import time

import numpy as np

CASES = [
    # name,            H,   W,  Cin, Cout, kernel_size, stride
    ("small_k1_s1", 20, 20, 4, 8, 1, 1),
    ("small_k2_s1", 20, 20, 4, 8, 2, 1),
    ("small_k3_s1", 20, 20, 4, 8, 3, 1),
    ("medium_k2_s1", 64, 64, 8, 16, 2, 1),
    ("medium_k2_s2", 64, 64, 8, 16, 2, 2),
    ("large_k2_s1", 128, 128, 16, 32, 2, 1),
    ("large_k3_s1", 128, 128, 16, 32, 3, 1),
    ("many_channels", 64, 64, 64, 64, 2, 1),
]

BATCH = 8
N_WARMUP = 5
N_REPS = 30


def bench_keras_hexagdly(compiled=False):
    import keras

    import keras_hexagdly as hgly

    backend = keras.backend.backend()
    results = []
    for name, h, w, cin, cout, k, s in CASES:
        layer = hgly.Conv2d(cin, cout, kernel_size=k, stride=s, bias=True)
        x = keras.ops.convert_to_tensor(np.random.randn(BATCH, h, w, cin).astype(np.float32))

        call = layer
        if compiled:
            if backend == "tensorflow":
                import tensorflow as tf

                call = tf.function(layer, jit_compile=False)
            elif backend == "jax":
                import jax

                call = jax.jit(layer)
            elif backend == "torch":
                import torch

                call = torch.compile(layer)

        for _ in range(N_WARMUP):
            call(x)
        t0 = time.perf_counter()
        for _ in range(N_REPS):
            call(x)
        t1 = time.perf_counter()
        results.append((name, (t1 - t0) / N_REPS * 1000))
    return results


def bench_keras_hexagdly_cpu():
    import keras

    import keras_hexagdly as hgly

    results = []
    with keras.device("cpu"):
        for name, h, w, cin, cout, k, s in CASES:
            layer = hgly.Conv2d(cin, cout, kernel_size=k, stride=s, bias=True)
            x = keras.ops.convert_to_tensor(np.random.randn(BATCH, h, w, cin).astype(np.float32))
            for _ in range(N_WARMUP):
                layer(x)
            t0 = time.perf_counter()
            for _ in range(N_REPS):
                layer(x)
            t1 = time.perf_counter()
            results.append((name, (t1 - t0) / N_REPS * 1000))
    return results


def bench_pytorch():
    import hexagdly
    import torch

    results = []
    with torch.no_grad():
        for name, h, w, cin, cout, k, s in CASES:
            layer = hexagdly.Conv2d(cin, cout, kernel_size=k, stride=s, bias=True)
            x = torch.from_numpy(np.random.randn(BATCH, cin, h, w).astype(np.float32))
            for _ in range(N_WARMUP):
                layer(x)
            t0 = time.perf_counter()
            for _ in range(N_REPS):
                layer(x)
            t1 = time.perf_counter()
            results.append((name, (t1 - t0) / N_REPS * 1000))
    return results


if __name__ == "__main__":
    which = sys.argv[1]
    compiled = "--compiled" in sys.argv
    if which == "pytorch":
        results = bench_pytorch()
    elif which == "torch_cpu":
        results = bench_keras_hexagdly_cpu()
    else:
        results = bench_keras_hexagdly(compiled=compiled)
    if compiled:
        which += "_compiled"

    out = {"backend": which, "results": results}
    path = f"benchmarks/results_{which}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}")
    for name, ms in results:
        print(f"  {name:>16s}: {ms:7.3f} ms")
