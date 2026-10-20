"""RTL co-simulation of the dataflow hex kernel against its C simulation (not collected by pytest).

    python tests/smoke_cosim_dataflow.py OUT_DIR [--dataflow 0|1] [--share ring] [--reuse 1,2]

A small CNN with the structure of the real one (1x1 conv, ReLU, hex conv, ReLU, hex conv with
stride 2, ReLU, global mean, Dense) goes through hls4ml (io_stream, Vitis) and `build(csim, synth,
cosim, validation)`: Vitis HLS runs the C simulation, the C synthesis and the RTL co-simulation and
compares the two result logs ("C/RTL VALIDATION ... Test PASSED"). The flag only changes the hex
layers' kernel, so running it with --dataflow 0 gives the reference behaviour of the same network.
Keep the types wide (--precision): once a type overflows, C simulation and RTL do not wrap alike and
the validation fails whatever the kernel does (this showed as a false "ring sharing fails in RTL").
Needs Vitis HLS (vitis-run on PATH) and its licence; one Vitis job at a time.
"""

import argparse
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import hls4ml  # noqa: E402
import keras  # noqa: E402
import numpy as np  # noqa: E402

import keras_hexagdly as hgly  # noqa: E402
from keras_hexagdly.hls4ml_ext import patch_model_for_hls  # noqa: E402
from keras_hexagdly.hls4ml_handler import register_hex_gather_layers  # noqa: E402

PART = "xc7vx485tffg1927-2"
HEIGHT, WIDTH, CHANNELS = 12, 10, 4


def build_model(share_a, share_b, single=False):
    grid = keras.Input((HEIGHT, WIDTH, CHANNELS), name="x")
    x = keras.layers.Conv2D(8, 1, name="pointwise")(grid)
    x = keras.layers.ReLU(name="pointwise_relu")(x)
    x = hgly.Conv2d(8, kernel_size=1, share_neighbors=share_a, name="hex_a")(x)
    if single:  # just the first hex layer, averaged: 8 outputs to compare channel by channel
        model = keras.Model(grid, keras.layers.GlobalAveragePooling2D(name="mean")(x))
    else:
        x = keras.layers.ReLU(name="hex_a_relu")(x)
        x = hgly.Conv2d(8, kernel_size=1, strides=2, share_neighbors=share_b, name="hex_b")(x)
        x = keras.layers.ReLU(name="hex_b_relu")(x)
        x = keras.layers.GlobalAveragePooling2D(name="mean")(x)
        model = keras.Model(grid, keras.layers.Dense(1, name="out")(x))
    rng = np.random.default_rng(3)
    for weight in model.trainable_variables:
        weight.assign((rng.integers(-8, 9, weight.shape) / 8).astype("float32"))
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output_dir")
    parser.add_argument("--dataflow", type=int, default=1, choices=[0, 1])
    parser.add_argument("--share", default="none", choices=["none", "ring"], help="both hex layers")
    parser.add_argument(
        "--share-a", choices=["none", "ring"], help="hex_a only (overrides --share)"
    )
    parser.add_argument(
        "--share-b", choices=["none", "ring"], help="hex_b only (overrides --share)"
    )
    parser.add_argument("--reuse", default="1,2", help="ReuseFactor of hex_a,hex_b")
    parser.add_argument("--events", type=int, default=6)
    parser.add_argument(
        "--single", action="store_true", help="only hex_a + global mean (8 outputs)"
    )
    parser.add_argument(
        "--mean-accum", help="accumulator type of the global mean, e.g. ap_fixed<32,22>"
    )
    parser.add_argument(
        "--precision",
        default="ap_fixed<28,16>",
        help="default type of every layer: wide enough that nothing overflows",
    )
    args = parser.parse_args()

    register_hex_gather_layers("Vitis")
    share = {"none": False, "ring": "ring"}
    patched = patch_model_for_hls(
        build_model(
            share[args.share_a or args.share], share[args.share_b or args.share], args.single
        )
    )
    config = hls4ml.utils.config_from_keras_model(
        patched, granularity="name", backend="Vitis", default_precision=args.precision
    )
    config["Model"].update(Strategy="Latency", ReuseFactor=1)
    for name, reuse in zip(
        ("hex_a_linebuffer", "hex_b_linebuffer"), map(int, args.reuse.split(","))
    ):
        config["LayerName"][name].update(ReuseFactor=reuse, Strategy="Latency")
        if args.dataflow:
            config["LayerName"][name]["Dataflow"] = True

    if args.mean_accum:
        config["LayerName"]["mean"] = {"Precision": {"accum": args.mean_accum}}

    x = np.random.default_rng(1).integers(0, 32, (args.events, HEIGHT, WIDTH, CHANNELS))
    os.makedirs(args.output_dir, exist_ok=True)
    data_path = os.path.join(args.output_dir, "tb_input.npy")
    np.save(data_path, x.astype("float32").reshape(args.events, -1))
    hls_model = hls4ml.converters.convert_from_keras_model(
        patched,
        hls_config=config,
        io_type="io_stream",
        backend="Vitis",
        output_dir=args.output_dir,
        part=PART,
        clock_period=5,
        project_name="m",
        input_data_tb=data_path,
    )
    hls_model.compile()
    hls_model.build(csim=True, synth=True, cosim=True, validation=True, export=False)


if __name__ == "__main__":
    main()
