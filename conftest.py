import sys
from pathlib import Path

import pytest

# Allow running the test suite without an editable install (`pip install -e .`).
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "tests"))


@pytest.fixture(scope="session", autouse=True)
def full_float32_on_gpu():
    """Real float32 on Ampere+ GPUs for the whole session (TensorFlow, torch and JAX).

    They run convolutions in TF32 (a 10-bit mantissa, ~1e-3 relative error) by default, and the
    hexagonal kernels are wide, so comparing two computations at 1e-4 fails there for no reason
    that has to do with the code. The library itself never touches these global flags."""
    import keras

    backend = keras.backend.backend()
    if backend == "tensorflow":
        import tensorflow as tf

        was = tf.config.experimental.tensor_float_32_execution_enabled()
        tf.config.experimental.enable_tensor_float_32_execution(False)
        yield
        tf.config.experimental.enable_tensor_float_32_execution(was)
    elif backend == "torch":
        import torch

        was = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
        torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
        yield
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = was
    elif backend == "jax":
        import jax

        was = jax.config.jax_default_matmul_precision
        jax.config.update("jax_default_matmul_precision", "highest")
        yield
        jax.config.update("jax_default_matmul_precision", was)
    else:
        yield
