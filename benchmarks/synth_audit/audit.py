"""Synthesis audit of the hls4ml kernels: every configuration through the C simulation, Vitis HLS
C synthesis and Vivado logic synthesis (out of context, no place and route).

What the test suite cannot see: whether the generated RTL meets the clock once Vivado has
synthesised it, and whether a layer reaches the rate its strategy promises. Vitis HLS's own clock
estimate is not enough (0.7.1: 3.0 ns estimated, 56 ns after logic synthesis, for the kernels that
left the folding of a reuse factor to the tool).

    python audit.py list
    python audit.py run OUT_DIR [--only NAME ...] [--shard I/N] [--stage hls|vivado]   # resumable
    python audit.py report OUT_DIR [OUT_DIR ...]                    # Markdown table, anomalies flagged

Needs hls4ml, g++, Vitis HLS and Vivado (settings64.sh of both: VITIS_SETTINGS / VIVADO_SETTINGS).
Small layers on purpose: one case takes a few minutes and stays under 12 GB.
"""

import argparse
import glob
import itertools
import json
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("KERAS_BACKEND", "torch")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

# Device of the logic synthesis (covered by the free Vivado licence) and, unless AUDIT_HLS_PART
# names another one, of the high-level synthesis. A machine without that device or without a
# Vivado licence can still do the first stage: see --stage.
PART = "xc7k160tffg676-2"
HLS_PART = os.environ.get("AUDIT_HLS_PART", PART)
CLOCK_NS = 4
PRECISION = "ap_fixed<24,12>"
VITIS_SETTINGS = os.environ.get("VITIS_SETTINGS", "/tools/Xilinx/Vitis_HLS/2024.2/settings64.sh")
VIVADO_SETTINGS = os.environ.get("VIVADO_SETTINGS", "/tools/Xilinx/Vivado/2024.2/settings64.sh")
MEMORY_KB = 12_000_000
CIN, COUT = 4, 8


def cases():
    """Every configuration audited, as dicts. The name is the directory of the case."""
    out = []

    def add(layer, kernel, stride, share, rf, strategy, dataflow, shape=(12, 12)):
        name = f"{layer}_k{kernel}_s{stride}_{share or 'none'}_rf{rf}_{strategy.lower()}_{'dataflow' if dataflow else 'fused'}"
        out.append(dict(name=name, layer=layer, kernel=kernel, stride=stride, share=share, rf=rf,
                        strategy=strategy, dataflow=dataflow, shape=shape))  # fmt: skip

    both = list(itertools.product(("Latency", "Resource"), (False, True)))
    # one-ring convolution: every strategy, kernel and reuse factor, with and without ring sharing
    for (strategy, dataflow), rf, share in itertools.product(both, (1, 2, 4, 8), (None, "ring")):
        add("conv2d", 1, 1, share, rf, strategy, dataflow)
    # two rings (19 cells), stride 2, the pair-sharing modes
    for (strategy, dataflow), rf, share in itertools.product(both, (1, 4), (None, "ring")):
        add("conv2d", 2, 1, share, rf, strategy, dataflow)
    for (strategy, dataflow), rf in itertools.product(both, (1, 4)):
        add("conv2d", 1, 2, None, rf, strategy, dataflow, (13, 12))
    for share in ("diag", "sym"):
        for strategy, dataflow in both:
            add("conv2d", 1, 1, share, 4, strategy, dataflow)
    # a reuse factor that does not divide the number of inputs (28 here)
    for strategy, dataflow in both:
        add("conv2d", 1, 1, None, 3, strategy, dataflow)
    # pooling
    for layer, stride in itertools.product(("maxpool2d", "avgpool2d"), (1, 2)):
        add(layer, 1, stride, None, 1, "Latency", False)
    # 3D layers (depth 3, hexagonal one ring)
    for (strategy, dataflow), rf in itertools.product(both, (1, 4)):
        add("conv3d", 1, 1, None, rf, strategy, dataflow, (6, 8, 8))
    add("maxpool3d", 1, 1, None, 1, "Latency", False, (6, 8, 8))
    return out


def build_model(case):
    import keras
    import numpy as np

    import keras_hexagdly as hgly

    shape = (*case["shape"], CIN)
    x = inputs = keras.Input(shape, name="x")
    layer = case["layer"]
    if layer == "conv2d":
        hex_layer = hgly.Conv2d(COUT, kernel_size=case["kernel"], strides=case["stride"], use_bias=True,
                                share_neighbors=case["share"] or False, name="c")  # fmt: skip
    elif layer == "conv3d":
        hex_layer = hgly.Conv3d(COUT, kernel_size=(3, case["kernel"]), strides=(1, case["stride"]),
                                use_bias=True, share_neighbors=case["share"] or False, name="c")  # fmt: skip
    elif layer == "maxpool2d":
        hex_layer = hgly.MaxPool2d(case["kernel"], case["stride"], name="c")
    elif layer == "avgpool2d":
        hex_layer = hgly.AvgPool2d(case["kernel"], case["stride"], name="c")
    else:
        hex_layer = hgly.MaxPool3d(
            kernel_size=(2, case["kernel"]), strides=(1, case["stride"]), name="c"
        )
    model = keras.Model(inputs, hex_layer(x))
    rng = np.random.default_rng(7)
    for weight in hex_layer.trainable_variables:  # values exact in the fixed-point types
        weight.assign((rng.integers(-8, 9, weight.shape) / 8).astype("float32"))
    return model


def export(case, directory):
    """hls4ml project of the case in directory/prj; returns the C-simulation check."""
    import hls4ml
    import numpy as np

    from keras_hexagdly.hls4ml_ext import patch_model_for_hls
    from keras_hexagdly.hls4ml_handler import register_hex_gather_layers

    model = build_model(case)
    x = (np.random.default_rng(1).integers(0, 64, (4, *case["shape"], CIN)) / 4).astype("float32")
    reference = model.predict(x, verbose=0)
    register_hex_gather_layers("Vitis")
    patched = patch_model_for_hls(model)
    config = hls4ml.utils.config_from_keras_model(
        patched, granularity="name", backend="Vitis", default_precision=PRECISION
    )
    config["Model"].update(Strategy=case["strategy"], ReuseFactor=case["rf"])
    for layer in patched.layers:
        config["LayerName"].setdefault(layer.name, {}).update(
            Strategy=case["strategy"], ReuseFactor=case["rf"], Dataflow=case["dataflow"]
        )
    # test bench of the RTL co-simulation: two of the inputs, with the outputs of the Keras model
    np.save(os.path.join(directory, "tb_input.npy"), x[:2])
    np.save(os.path.join(directory, "tb_output.npy"), reference[:2])
    hls_model = hls4ml.converters.convert_from_keras_model(
        patched, hls_config=config, io_type="io_stream", backend="Vitis",
        output_dir=os.path.join(directory, "prj"), part=HLS_PART, clock_period=CLOCK_NS,
        input_data_tb=os.path.join(directory, "tb_input.npy"),
        output_data_tb=os.path.join(directory, "tb_output.npy"),
    )  # fmt: skip
    hls_model.compile()
    got = np.asarray(hls_model.predict(np.ascontiguousarray(x))).reshape(reference.shape)
    np.save(os.path.join(directory, "csim_output.npy"), got)
    open(os.path.join(directory, "prj", "build_opt.tcl"), "w").write(
        # C simulation, C synthesis, RTL co-simulation, and the comparison of the two simulations:
        # an RTL that differs from the C model passes every C-simulation test (0.7.2: an explicit
        # reuse_factor:1 selection in the folded MAC gave constant RTL outputs)
        "array set opt {\n    reset      1\n    csim       1\n    synth      1\n    cosim      1\n"
        "    validation 1\n    export     0\n    vsynth     0\n    fifo_opt   0\n}\n"
    )
    out_pixels = int(np.prod(reference.shape[1:-1]))
    return {
        "max_abs_difference_with_keras": float(np.abs(got - reference).max()),
        "max_abs_keras": float(np.abs(reference).max()),
        "input_tokens": int(np.prod(case["shape"])), "output_pixels": out_pixels,
    }  # fmt: skip


def shell(command, cwd, timeout):
    start = time.time()
    done = subprocess.run(
        ["bash", "-c", f"ulimit -v {MEMORY_KB}; {command}"], cwd=cwd, capture_output=True, text=True,
        timeout=timeout,
    )  # fmt: skip
    return done.returncode, round(time.time() - start), done.stdout[-2000:] + done.stderr[-2000:]


def synthesise_hls(directory):
    """Vitis HLS C synthesis of the project of a case: {'hls': figures} or {'error': ...}."""
    project = os.path.join(directory, "prj")
    code, seconds, tail = shell(
        f"source {VITIS_SETTINGS} && vitis-run --mode hls --tcl build_prj.tcl > build.log 2>&1",
        project,
        3000,
    )
    reports = glob.glob(os.path.join(project, "*_prj/solution1/syn/report/myproject_csynth.xml"))
    if not reports:
        return {"error": f"no C synthesis report (exit {code}): see prj/build.log"}
    root = ET.parse(reports[0]).getroot()

    def get(path):
        return root.find(path).text

    log = open(os.path.join(project, "build.log"), errors="replace").read()
    result = {
        "hls": {
            "latency_cycles": int(get(".//Worst-caseLatency")), "interval_cycles": int(get(".//Interval-max")),
            "clock_estimate_ns": float(get(".//EstimatedClockPeriod")),
            "LUT": int(get(".//Resources/LUT")), "FF": int(get(".//Resources/FF")),
            "DSP": int(get(".//Resources/DSP")), "BRAM_18K": int(get(".//Resources/BRAM_18K")),
            "seconds": seconds,
            # an II the tool could not meet is only a warning in its log
            "ii_violations": sorted(set(re.findall(r"Unable to (?:enforce|schedule)[^\n]{0,160}", log)))[:4],
            # RTL against the C simulation on the test bench
            "rtl_equals_c_simulation": True if "INFO: Test PASSED" in log else False if "Test failed" in log else None,
        }
    }  # fmt: skip
    return result


def synthesise_vivado(directory):
    """Vivado logic synthesis (out of context) of the RTL that Vitis HLS wrote: the figures."""
    project = os.path.join(directory, "prj")
    open(os.path.join(project, "vivado_ooc.tcl"), "w").write(
        f"source project.tcl\nset part {PART}\nset_param general.maxThreads 4\n"
        "add_files ${project_name}_prj/solution1/syn/verilog\n"
        "synth_design -top ${project_name} -part $part -mode out_of_context\n"
        "create_clock -period $clock_period -name clk [get_ports ap_clk]\n"
        "report_utilization -file vivado_ooc_util.rpt\nreport_timing_summary -file vivado_ooc_timing.rpt\n"
    )
    code, seconds, tail = shell(
        f"source {VIVADO_SETTINGS} && vivado -mode batch -nojournal -log vivado_ooc.log -source vivado_ooc.tcl"
        " > /dev/null 2>&1", project, 3000,
    )  # fmt: skip
    try:
        util = open(os.path.join(project, "vivado_ooc_util.rpt")).read()
        timing = open(os.path.join(project, "vivado_ooc_timing.rpt")).read()
    except FileNotFoundError:
        return {"error": f"no Vivado report (exit {code}): see prj/vivado_ooc.log"}

    def used(name):
        return float(re.search(rf"\|\s*{name}\s*\|\s*([\d.]+)", util).group(1))

    wns = float(re.search(r"WNS\(ns\).*?\n\s*-+.*?\n\s*(-?[\d.]+)", timing, re.S).group(1))
    levels = re.search(r"Logic Levels:\s+(\d+)", timing)
    source = re.search(r"Max Delay Paths.*?Source:\s+(\S+)", timing, re.S)
    return {
        "LUT": int(used(r"Slice LUTs\*?")), "FF": int(used("Slice Registers")), "DSP": int(used("DSPs")),
        "BRAM_tiles": used("Block RAM Tile"), "wns_ns": wns, "critical_path_ns": round(CLOCK_NS - wns, 3),
        "logic_levels": int(levels.group(1)) if levels else None,
        "critical_source": source.group(1)[:160] if source else None, "seconds": seconds, "part": PART,
    }  # fmt: skip


def wait_for_the_tools():
    if os.environ.get("AUDIT_NO_WAIT"):  # the caller runs several shards side by side on purpose
        return
    while subprocess.run(["pgrep", "-f", "vitis-ru[n]|vivado -mod[e]"], capture_output=True).stdout:
        time.sleep(20)  # one synthesis at a time on the machine


def run(args):
    selected = [c for c in cases() if not args.only or c["name"] in args.only]
    if args.shard:
        index, count = map(int, args.shard.split("/"))
        selected = [c for i, c in enumerate(selected) if i % count == index]
    for case in selected:
        directory = os.path.join(args.out_dir, case["name"])
        path = os.path.join(directory, "result.json")
        result = json.load(open(path)) if os.path.exists(path) else None
        try:
            if result is None and args.stage in ("both", "hls"):
                os.makedirs(directory, exist_ok=True)
                wait_for_the_tools()
                result = dict(
                    case,
                    hls_part=HLS_PART,
                    clock_ns=CLOCK_NS,
                    precision=PRECISION,
                    channels=[CIN, COUT],
                )
                result["csim"] = export(case, directory)
                result.update(synthesise_hls(directory))
            elif result is None or "error" in result or "vivado" in result or args.stage == "hls":
                continue  # nothing to do for this stage (not exported yet, failed, or done)
            if "hls" in result and "vivado" not in result and args.stage in ("both", "vivado"):
                wait_for_the_tools()
                result["vivado"] = synthesise_vivado(directory)
        except Exception as error:  # a case that fails is a finding, not a reason to stop
            result = dict(result or case, error=f"{type(error).__name__}: {error}"[:600])
        json.dump(result, open(path, "w"), indent=1)
        print(
            json.dumps(
                {k: v for k, v in result.items() if k in ("name", "error", "hls", "vivado")}
            )[:400],
            flush=True,
        )


def flags(result, outputs):
    """What is wrong with a case, as short strings."""
    found = []
    if "error" in result:
        return ["ERROR " + result["error"][:80]]
    csim, hls, vivado = result["csim"], result.get("hls", {}), result.get("vivado", {})
    if csim["max_abs_difference_with_keras"] > 1e-6:
        found.append(f"csim differs from Keras by {csim['max_abs_difference_with_keras']:.3g}")
    twin = result["name"].replace("_dataflow", "_fused")
    if result["name"] != twin and twin in outputs and result["name"] in outputs:
        import numpy as np

        if not np.array_equal(outputs[twin], outputs[result["name"]]):
            found.append("dataflow and fused outputs differ")
    if "error" in vivado:
        found.append("VIVADO " + vivado["error"][:60])
    elif vivado:
        if vivado["wns_ns"] < 0:
            found.append(f"TIMING {vivado['critical_path_ns']} ns, {vivado['logic_levels']} levels")
        if (
            hls.get("clock_estimate_ns")
            and vivado["critical_path_ns"] > 1.5 * hls["clock_estimate_ns"]
        ):
            found.append("HLS clock estimate off by more than 1.5x")
    if hls:
        if hls.get("rtl_equals_c_simulation") is False:
            found.append("RTL DIFFERS from the C simulation")
        elif hls.get("rtl_equals_c_simulation") is None:
            found.append("no co-simulation verdict")
        tokens = csim["input_tokens"]
        per_token = hls["latency_cycles"] / tokens
        # cycles per input pixel: reuse_factor in the pipelined kernels; the unpipelined resource
        # kernel adds the steps of its loop (measured: 18 to 28 more). Virtual rows and columns
        # (n more columns per row, a few more rows) come on top, hence the margin.
        pipelined = result["dataflow"] or result["strategy"] == "Latency"
        expected = result["rf"] if pipelined else result["rf"] + 28
        if result["layer"].startswith("conv") and per_token > 1.6 * expected + 1:
            found.append(f"RATE {per_token:.1f} cycles per input token, expected about {expected}")
        if hls["ii_violations"]:
            found.append("II not met: " + hls["ii_violations"][0][:70])
    return found


def report(args):
    import numpy as np

    results, outputs = [], {}
    for out_dir in args.out_dirs:
        for path in sorted(glob.glob(os.path.join(out_dir, "*", "result.json"))):
            result = json.load(open(path))
            results = [r for r in results if r["name"] != result["name"]]  # a later directory wins
            results.append(result)
            npy = os.path.join(os.path.dirname(path), "csim_output.npy")
            if os.path.exists(npy):
                outputs[result["name"]] = np.load(npy)
    order = {c["name"]: i for i, c in enumerate(cases())}
    results.sort(key=lambda r: order.get(r["name"], 1e9))
    print(f"{len(results)} of {len(order)} cases. Vitis HLS 2024.2 then Vivado 2024.2 logic synthesis out of context,"
          f" {PART}, {CLOCK_NS} ns, {PRECISION}, {CIN} -> {COUT} channels.\n")  # fmt: skip
    print(
        "| Case | Cycles | Cycles/token | HLS clock | Vivado path | Levels | LUT (HLS / Vivado) | DSP (HLS / Vivado) | Flags |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    flagged = 0
    for result in results:
        found = flags(result, outputs)
        flagged += bool(found)
        hls, vivado = result.get("hls", {}), result.get("vivado", {})
        tokens = result.get("csim", {}).get("input_tokens", 0)
        print("| " + " | ".join(str(v) for v in (
            result["name"], hls.get("latency_cycles", ""),
            f"{hls['latency_cycles'] / tokens:.1f}" if hls and tokens else "",
            hls.get("clock_estimate_ns", ""), vivado.get("critical_path_ns", ""), vivado.get("logic_levels", ""),
            f"{hls.get('LUT', '')} / {vivado.get('LUT', '')}", f"{hls.get('DSP', '')} / {vivado.get('DSP', '')}",
            "; ".join(found),
        )) + " |")  # fmt: skip
    print(f"\n{flagged} case(s) flagged.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("list")
    p = sub.add_parser("run")
    p.add_argument("out_dir")
    p.add_argument("--only", nargs="*")
    p.add_argument("--shard", help="I/N: every N-th case starting at I, to split over machines")
    p.add_argument(
        "--stage",
        choices=["both", "hls", "vivado"],
        default="both",
        help="hls: export, C simulation and Vitis HLS only (then copy OUT_DIR to a machine with "
        "Vivado and run --stage vivado there on the same OUT_DIR)",
    )
    p = sub.add_parser("report")
    p.add_argument("out_dirs", nargs="+")
    arguments = parser.parse_args()
    if arguments.action == "list":
        for case in cases():
            print(case["name"])
    elif arguments.action == "run":
        run(arguments)
    else:
        report(arguments)
