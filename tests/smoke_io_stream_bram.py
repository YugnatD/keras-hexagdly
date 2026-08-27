"""Smoke test: io_stream synthesis produces non-zero BRAM usage.

Builds the smallest possible hex CNN (5×5 input, 1 conv + pool + 1 dense),
runs Vitis HLS csynth with io_stream + Strategy=Resource, and checks that the
csynth report shows BRAM_18K > 0.

Run manually (requires Vitis HLS on PATH):
    python tests/smoke_io_stream_bram.py

Exit 0 = BRAM non-zero and design fits.  Exit 1 = something went wrong.
"""

import os
import re
import shutil
import stat
import sys

import hls4ml
import keras

import keras_hexagdly as hgly
from keras_hexagdly.hls4ml_ext import patch_model_for_hls
from keras_hexagdly.hls4ml_handler import register_hex_gather_layers

# ── Vitis HLS shim (same as notebook) ────────────────────────────────────────
VITIS_BIN = "/tools/Xilinx/Vitis_HLS/2024.2/bin"
os.environ["PATH"] = VITIS_BIN + ":" + os.environ["PATH"]
shim = os.path.expanduser("~/.local/bin/vivado_hls")
if not os.path.isfile(shim):
    os.makedirs(os.path.dirname(shim), exist_ok=True)
    with open(shim, "w") as f:
        f.write('#!/bin/bash\nexec /tools/Xilinx/Vitis_HLS/2024.2/bin/vitis_hls "$@"\n')
    os.chmod(shim, os.stat(shim).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

# ── Tiny model: 5×5 grid, 1 input channel ────────────────────────────────────
# Small enough that csynth is fast; enough channels that weights qualify for BRAM.
H, W = 5, 5
N_CLASSES = 3

inp = keras.Input(shape=(H, W, 1), name="image")
x = hgly.Conv2d(
    filters=4, kernel_size=1, strides=1, share_neighbors=True, use_bias=True, name="conv1"
)(inp)
x = hgly.MaxPool2d(kernel_size=1, strides=2, name="pool1")(x)
x = keras.layers.Flatten(name="flat")(x)
x = keras.layers.Dense(8, activation="relu", name="fc1")(x)
out = keras.layers.Dense(N_CLASSES, name="logits")(x)
model = keras.Model(inp, out, name="tiny_hex")
model.summary()

# ── Patch for HLS gather strategy ────────────────────────────────────────────
register_hex_gather_layers()
patched = patch_model_for_hls(model, strategy="gather")

# ── HLS config: Resource + low ReuseFactor + io_stream ───────────────────────
OUT_DIR = "/tmp/smoke_io_stream_bram"
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)

cfg = hls4ml.utils.config_from_keras_model(patched, granularity="name", backend="Vivado")
cfg["Model"]["Precision"] = "ap_fixed<16,6>"
cfg["Model"]["Strategy"] = "Resource"
cfg["Model"]["ReuseFactor"] = 4
cfg["Model"]["BramFactor"] = 10  # push anything >10 elements to BRAM

hls_model = hls4ml.converters.convert_from_keras_model(
    patched,
    hls_config=cfg,
    backend="Vivado",
    output_dir=OUT_DIR,
    part="xc7vx485tffg1927-2",
    io_type="io_stream",
)

print("\n[smoke] Starting csynth (io_stream) — this takes a few minutes …")
hls_model.build(csim=False, synth=True, export=False)

# ── Parse the csynth report ──────────────────────────────────────────────────
rpt_path = os.path.join(OUT_DIR, "myproject_prj/solution1/syn/report/myproject_csynth.rpt")
if not os.path.exists(rpt_path):
    print(f"[smoke] FAIL — report not found at {rpt_path}")
    sys.exit(1)

rpt = open(rpt_path).read()

# Extract the summary utilization table
# Line looks like:  |Total            |        4|  1234|  56789|  23456|    0|
m = re.search(r"\|Total\s+\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|", rpt)
if not m:
    print("[smoke] FAIL — could not parse utilization table")
    sys.exit(1)

bram, dsp, ff, lut, uram = [int(m.group(i)) for i in range(1, 6)]

# Also grab % utilization line
m_pct = re.search(
    r"\|Utilization \(%\)\s+\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|", rpt
)
bram_pct, dsp_pct, ff_pct, lut_pct, uram_pct = (
    [int(m_pct.group(i)) for i in range(1, 6)] if m_pct else [0] * 5
)

print("\n" + "=" * 60)
print("  io_stream csynth utilization")
print("=" * 60)
print(f"  BRAM_18K : {bram:6d}  ({bram_pct:3d}%)   ← target > 0")
print(f"  DSP      : {dsp:6d}  ({dsp_pct:3d}%)")
print(f"  FF       : {ff:6d}  ({ff_pct:3d}%)")
print(f"  LUT      : {lut:6d}  ({lut_pct:3d}%)")
print(f"  URAM     : {uram:6d}  ({uram_pct:3d}%)")
print("=" * 60)

# Print timing too
m_clk = re.search(r"\|ap_clk\s+\|\s*[\d.]+\s*ns\|\s*([\d.]+)\s*ns", rpt)
if m_clk:
    print(f"  Timing   : estimated {m_clk.group(1)} ns")

m_lat = re.search(r"\|\s*(\d+)\|\s*(\d+)\|\s*[\d.]+\s*[nu]s\|.*\|\s*(yes|no)\|", rpt)
if m_lat:
    print(f"  Latency  : {m_lat.group(1)}–{m_lat.group(2)} cycles  pipelined={m_lat.group(3)}")

print()
if bram == 0:
    print("[smoke] FAIL — BRAM is still 0. io_stream / BramFactor not working as expected.")
    print(f"        Report: {rpt_path}")
    sys.exit(1)
else:
    print(f"[smoke] PASS — BRAM={bram} > 0. io_stream is using BRAM correctly.")
    print(f"        Report: {rpt_path}")
    sys.exit(0)
