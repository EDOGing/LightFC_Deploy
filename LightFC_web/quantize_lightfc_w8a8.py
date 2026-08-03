"""Static W8A8 PTQ for split LightFC ONNX (opset 12, CPU).

Pipeline: GOT-10k sampling -> PyTorch AdaRound -> split FP32 ONNX export ->
Percentile/MSE/hybrid activation calibration -> QLinearConv ONNX export.
Pixel-wise correlation MatMul is deliberately left in FP32.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

from export_onnx_split import TemplateBackbone, save_metadata
from quantization.activation import choose_ranges, collect_activation_values, save_range_cache
from quantization.adaround import apply_adaround
from quantization.config import QuantizationConfig
from quantization.got10k import GOT10kPairs, PairSpec
from web_demo import DEFAULT_CHECKPOINT, DEFAULT_CONFIG, LightFCCPU, LightFCONNXCPU


ROOT = Path(__file__).resolve().parent


class QuantizedDeploymentTrackingNetwork(nn.Module):
    """Export only the three tensors consumed by deployment box decoding."""

    def __init__(self, network: nn.Module):
        super().__init__()
        self.network = network

    def forward(self, template_features: torch.Tensor, search: torch.Tensor):
        output = self.network.forward_tracking(template_features, search)
        return output["score_map"], output["size_map"], output["offset_map"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize split LightFC to opset-12 W8A8 ONNX")
    parser.add_argument("--got10k", type=Path, default=Path(r"F:\dataset\got10k\train"))
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "quantized")
    parser.add_argument("--strategy", choices=["A", "B", "C", "a", "b", "c"], default="C")
    parser.add_argument("--calibration-pairs", type=int, default=None)
    parser.add_argument("--validation-pairs", type=int, default=None)
    parser.add_argument("--adaround-iterations", type=int, default=None)
    parser.add_argument("--skip-adaround", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume layer-wise AdaRound checkpoint")
    parser.add_argument("--evaluate-only", action="store_true", help="Compare existing FP32/INT8 models without requantizing")
    parser.add_argument("--benchmark-runs", type=int, default=200)
    parser.add_argument("--benchmark-warmup", type=int, default=20)
    parser.add_argument("--benchmark-threads", type=int, default=0, help="ORT intra-op threads; 0 uses its default")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Very small end-to-end engineering test")
    return parser.parse_args()


def _export(backbone, tracking, template, search, output_dir: Path) -> tuple[Path, Path]:
    backbone_path = output_dir / "lightfc_adaround_fp32_backbone_opset12.onnx"
    tracking_path = output_dir / "lightfc_adaround_fp32_tracking_opset12.onnx"
    # ONNX tracing may enable autograd internally; inference-mode tensors cannot
    # then be saved by MatMul, so create this boundary tensor under no_grad.
    with torch.no_grad():
        features = backbone(template)
    features = features.clone()
    torch.onnx.export(backbone, (template,), str(backbone_path), input_names=["template"], output_names=["template_features"], opset_version=12, do_constant_folding=True, dynamo=False)
    torch.onnx.export(tracking, (features, search), str(tracking_path), input_names=["template_features", "search"], output_names=["score_map", "size_map", "offset_map"], opset_version=12, do_constant_folding=True, dynamo=False)
    save_metadata(backbone_path, "template_backbone_adaround_fp32")
    save_metadata(tracking_path, "search_fusion_head_adaround_fp32")
    return backbone_path, tracking_path


def _quantize(source: Path, target: Path, cache: Path) -> None:
    quantize_static(
        source,
        target,
        calibration_data_reader=None,
        calibration_cache_path=cache,
        calibrate_method=CalibrationMethod.MinMax,
        quant_format=QuantFormat.QOperator,
        op_types_to_quantize=["Conv"],
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
    )


def _inspect(path: Path) -> dict[str, int]:
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    opset = next(item.version for item in model.opset_import if item.domain in {"", "ai.onnx"})
    counts: dict[str, int] = {"opset": opset}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    if opset != 12:
        raise RuntimeError(f"{path.name} is opset {opset}, expected 12")
    if counts.get("QLinearConv", 0) == 0:
        raise RuntimeError(f"No QLinearConv found in {path.name}")
    return counts


def _compare(float_path: Path, int8_path: Path, feed: dict[str, np.ndarray]) -> dict[str, float]:
    fp = ort.InferenceSession(str(float_path), providers=["CPUExecutionProvider"]).run(None, feed)
    q = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"]).run(None, feed)
    return {f"output_{i}_mae": float(np.mean(np.abs(a - b))) for i, (a, b) in enumerate(zip(fp, q))}


def _iou_xywh(a: list[float], b: list[float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2, bx2, by2 = ax1 + aw, ay1 + ah, bx1 + bw, by1 + bh
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = max(0.0, aw * ah) + max(0.0, bw * bh) - intersection
    return intersection / union if union > 0 else 0.0


def _center_error(a: list[float], b: list[float]) -> float:
    acx, acy = a[0] + 0.5 * a[2], a[1] + 0.5 * a[3]
    bcx, bcy = b[0] + 0.5 * b[2], b[1] + 0.5 * b[3]
    return float(np.hypot(acx - bcx, acy - bcy))


def _summarize_accuracy(ious: list[float], errors: list[float]) -> dict[str, float]:
    iou_values = np.asarray(ious, dtype=np.float32)
    error_values = np.asarray(errors, dtype=np.float32)
    thresholds = np.linspace(0.0, 1.0, 21)
    success_auc = float(np.mean([(iou_values >= threshold).mean() for threshold in thresholds]))
    return {
        "mean_iou": float(iou_values.mean()),
        "success_auc": success_auc,
        "precision_20px": float((error_values <= 20.0).mean()),
        "mean_center_error_px": float(error_values.mean()),
    }


def _tracking_accuracy(
    dataset: GOT10kPairs,
    specs,
    config_path: Path,
    fp_backbone: Path,
    fp_tracking: Path,
    int8_backbone: Path,
    int8_tracking: Path,
) -> dict:
    trackers = {
        "fp32": LightFCONNXCPU(fp_backbone, fp_tracking, config_path),
        "int8": LightFCONNXCPU(int8_backbone, int8_tracking, config_path),
    }
    measurements = {name: {"ious": [], "errors": []} for name in trackers}
    pairs = []
    for index, spec in enumerate(specs, 1):
        template, search, template_box, ground_truth = dataset.load_raw(spec)
        pair_result = {"sequence": spec.sequence, "template_index": spec.template_index, "search_index": spec.search_index}
        for name, tracker in trackers.items():
            tracker.initialize(template, template_box)
            predicted, confidence = tracker.track(search)
            iou = _iou_xywh(predicted, ground_truth)
            error = _center_error(predicted, ground_truth)
            measurements[name]["ious"].append(iou)
            measurements[name]["errors"].append(error)
            pair_result[name] = {"iou": iou, "center_error_px": error, "confidence": confidence}
        pairs.append(pair_result)
        print(f"Accuracy evaluation: pair {index}/{len(specs)}")
    summary = {
        name: _summarize_accuracy(values["ious"], values["errors"])
        for name, values in measurements.items()
    }
    summary["delta_int8_minus_fp32"] = {
        key: summary["int8"][key] - summary["fp32"][key]
        for key in summary["fp32"]
    }
    summary["evaluation_type"] = "held-out GOT-10k train frame pairs; not official GOT-10k test"
    summary["pair_count"] = len(specs)
    summary["pairs"] = pairs
    return summary


class _ProcessMemoryCountersEx(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def _windows_memory() -> tuple[int, int, int]:
    counters = _ProcessMemoryCountersEx()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_ProcessMemoryCountersEx),
        ctypes.c_ulong,
    ]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    handle = kernel32.GetCurrentProcess()
    ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    if not ok:
        raise ctypes.WinError()
    return counters.WorkingSetSize, counters.PeakWorkingSetSize, counters.PrivateUsage


def _benchmark_worker(queue, backbone_path: str, tracking_path: str, template, search, warmup: int, runs: int, threads: int) -> None:
    try:
        baseline_rss, baseline_peak, baseline_private = _windows_memory()
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads > 0:
            options.intra_op_num_threads = threads
        backbone = ort.InferenceSession(backbone_path, sess_options=options, providers=["CPUExecutionProvider"])
        tracking = ort.InferenceSession(tracking_path, sess_options=options, providers=["CPUExecutionProvider"])
        feature = backbone.run(["template_features"], {"template": template})[0]
        tracking_feed = {"template_features": feature, "search": search}
        for _ in range(warmup):
            backbone.run(None, {"template": template})
            tracking.run(None, tracking_feed)
        start = time.perf_counter()
        for _ in range(runs):
            backbone.run(None, {"template": template})
        backbone_seconds = time.perf_counter() - start
        start = time.perf_counter()
        for _ in range(runs):
            tracking.run(None, tracking_feed)
        tracking_seconds = time.perf_counter() - start
        rss, peak, private = _windows_memory()
        queue.put({
            "template_latency_ms": backbone_seconds * 1000.0 / runs,
            "tracking_latency_ms": tracking_seconds * 1000.0 / runs,
            "tracking_fps": runs / tracking_seconds,
            "working_set_increment_mib": max(0, rss - baseline_rss) / 1024**2,
            "peak_working_set_increment_mib": max(0, peak - baseline_peak) / 1024**2,
            "private_memory_increment_mib": max(0, private - baseline_private) / 1024**2,
        })
    except Exception as exc:
        queue.put({"error": repr(exc)})


def _benchmark_pair(
    fp_backbone: Path,
    fp_tracking: Path,
    int8_backbone: Path,
    int8_tracking: Path,
    template: np.ndarray,
    search: np.ndarray,
    warmup: int,
    runs: int,
    threads: int,
) -> dict:
    context = mp.get_context("spawn")
    results = {}
    for name, paths in {
        "fp32": (fp_backbone, fp_tracking),
        "int8": (int8_backbone, int8_tracking),
    }.items():
        queue = context.Queue()
        process = context.Process(
            target=_benchmark_worker,
            args=(queue, str(paths[0]), str(paths[1]), template, search, warmup, runs, threads),
        )
        process.start()
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"{name} benchmark process failed with exit code {process.exitcode}")
        result = queue.get()
        if "error" in result:
            raise RuntimeError(f"{name} benchmark failed: {result['error']}")
        result["model_size_mib"] = (paths[0].stat().st_size + paths[1].stat().st_size) / 1024**2
        results[name] = result
    fp, quant = results["fp32"], results["int8"]
    results["int8_vs_fp32"] = {
        "tracking_speedup": quant["tracking_fps"] / fp["tracking_fps"],
        "tracking_latency_reduction_percent": (1.0 - quant["tracking_latency_ms"] / fp["tracking_latency_ms"]) * 100.0,
        "model_size_reduction_percent": (1.0 - quant["model_size_mib"] / fp["model_size_mib"]) * 100.0,
        "working_set_reduction_percent": (1.0 - quant["working_set_increment_mib"] / fp["working_set_increment_mib"]) * 100.0 if fp["working_set_increment_mib"] else 0.0,
        "private_memory_reduction_percent": (1.0 - quant["private_memory_increment_mib"] / fp["private_memory_increment_mib"]) * 100.0 if fp["private_memory_increment_mib"] else 0.0,
    }
    results["settings"] = {"warmup": warmup, "runs": runs, "intra_op_threads": threads or "ORT default"}
    return results


def main() -> None:
    args = parse_args()
    cfg = QuantizationConfig(activation_strategy=args.strategy.upper())
    if args.calibration_pairs is not None:
        cfg.calibration_pairs = args.calibration_pairs
    if args.validation_pairs is not None:
        cfg.validation_pairs = args.validation_pairs
    if args.adaround_iterations is not None:
        cfg.adaround_iterations = args.adaround_iterations
    if args.smoke:
        cfg.calibration_sequences, cfg.validation_sequences = 4, 2
        cfg.calibration_pairs, cfg.validation_pairs = 2, 1
        cfg.adaround_pairs_per_layer, cfg.adaround_iterations = 1, 2
        cfg.activation_values_per_tensor, cfg.activation_values_per_batch = 4096, 4096
    cfg.validate()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_manifest_path = output_dir / "got10k_pair_manifest.json"
    saved_pair_manifest = None
    if args.evaluate_only and pair_manifest_path.is_file():
        saved_pair_manifest = json.loads(pair_manifest_path.read_text(encoding="utf-8"))
    if not args.evaluate_only:
        (output_dir / "quantization_config.json").write_text(json.dumps(cfg.__dict__, indent=2), encoding="utf-8")

    tracker = LightFCCPU(args.checkpoint.resolve(), args.config.resolve())
    dataset = GOT10kPairs(
        args.got10k,
        tracker.template_factor, tracker.template_size,
        tracker.search_factor, tracker.search_size,
        list(tracker.cfg.DATA.MEAN), list(tracker.cfg.DATA.STD),
        cfg.seed, cfg.calibration_sequences, cfg.validation_sequences,
        cfg.max_frame_gap, cfg.min_visible_ratio,
        output_dir / "got10k_split_manifest.json",
    )
    if saved_pair_manifest is not None:
        validation_specs = [PairSpec(**item) for item in saved_pair_manifest["validation"]]
        if args.validation_pairs is not None:
            validation_specs = validation_specs[:args.validation_pairs]
    else:
        validation_specs = dataset.specs("validation", cfg.validation_pairs, cfg.seed + 2)

    backbone_fp = output_dir / "lightfc_adaround_fp32_backbone_opset12.onnx"
    tracking_fp = output_dir / "lightfc_adaround_fp32_tracking_opset12.onnx"
    backbone_int8 = output_dir / "lightfc_w8a8_backbone_opset12.onnx"
    tracking_int8 = output_dir / "lightfc_w8a8_tracking_opset12.onnx"
    if args.evaluate_only:
        required = (backbone_fp, tracking_fp, backbone_int8, tracking_int8)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing models for --evaluate-only: " + ", ".join(missing))
        report_path = output_dir / "validation_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
        accuracy = _tracking_accuracy(
            dataset, validation_specs, args.config.resolve(),
            backbone_fp, tracking_fp, backbone_int8, tracking_int8,
        )
        pair_details = accuracy.pop("pairs")
        report["tracking_accuracy"] = accuracy
        if not args.skip_benchmark:
            benchmark_template, benchmark_search = dataset.load(validation_specs[0])
            report["performance_comparison"] = _benchmark_pair(
                backbone_fp, tracking_fp, backbone_int8, tracking_int8,
                benchmark_template, benchmark_search,
                args.benchmark_warmup, args.benchmark_runs, args.benchmark_threads,
            )
        (output_dir / "accuracy_pair_details.json").write_text(
            json.dumps(pair_details, indent=2), encoding="utf-8"
        )
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report["tracking_accuracy"], indent=2))
        print(f"Updated {report_path}")
        return

    calibration_specs = dataset.specs("calibration", cfg.calibration_pairs, cfg.seed + 1)
    pair_manifest_path.write_text(json.dumps({"calibration": [x.__dict__ for x in calibration_specs], "validation": [x.__dict__ for x in validation_specs]}, indent=2), encoding="utf-8")

    def torch_pairs():
        for template, search in dataset.iter_arrays(calibration_specs[:cfg.adaround_pairs_per_layer]):
            yield torch.from_numpy(template), torch.from_numpy(search)

    if not args.skip_adaround:
        apply_adaround(
            tracker.network,
            torch_pairs,
            cfg,
            checkpoint_path=output_dir / "adaround_progress.pth",
            resume=args.resume,
        )

    backbone = TemplateBackbone(tracker.network).cpu().eval()
    tracking = QuantizedDeploymentTrackingNetwork(tracker.network).cpu().eval()
    first_template, first_search = dataset.load(calibration_specs[0])
    backbone_fp, tracking_fp = _export(backbone, tracking, torch.from_numpy(first_template), torch.from_numpy(first_search), output_dir)

    calibration_arrays = list(dataset.iter_arrays(calibration_specs))
    backbone_values = collect_activation_values(
        backbone_fp, ({"template": t} for t, _ in calibration_arrays), cfg,
        output_dir / "_augmented_backbone.onnx",
    )
    backbone_ranges, backbone_choices = choose_ranges(backbone_values, cfg)
    backbone_cache = output_dir / "backbone_calibration_cache.json"
    save_range_cache(backbone_ranges, backbone_cache)

    # Quantize the template branch first, then calibrate tracking with the
    # actual INT8 template features it will receive in deployment.
    _quantize(backbone_fp, backbone_int8, backbone_cache)
    backbone_session = ort.InferenceSession(str(backbone_int8), providers=["CPUExecutionProvider"])
    tracking_feeds = (
        {"template_features": backbone_session.run(None, {"template": template})[0], "search": search}
        for template, search in calibration_arrays
    )
    tracking_values = collect_activation_values(
        tracking_fp, tracking_feeds, cfg, output_dir / "_augmented_tracking.onnx"
    )
    tracking_ranges, tracking_choices = choose_ranges(tracking_values, cfg)
    tracking_cache = output_dir / "tracking_calibration_cache.json"
    save_range_cache(tracking_ranges, tracking_cache)
    (output_dir / "activation_strategy_choices.json").write_text(json.dumps({"backbone": backbone_choices, "tracking": tracking_choices}, indent=2), encoding="utf-8")

    _quantize(tracking_fp, tracking_int8, tracking_cache)

    test_template, test_search = dataset.load(validation_specs[0])
    fp_feature = ort.InferenceSession(str(backbone_fp), providers=["CPUExecutionProvider"]).run(None, {"template": test_template})[0]
    int8_feature = ort.InferenceSession(str(backbone_int8), providers=["CPUExecutionProvider"]).run(None, {"template": test_template})[0]
    fp_tracking_outputs = ort.InferenceSession(str(tracking_fp), providers=["CPUExecutionProvider"]).run(None, {"template_features": fp_feature, "search": test_search})
    int8_tracking_outputs = ort.InferenceSession(str(tracking_int8), providers=["CPUExecutionProvider"]).run(None, {"template_features": int8_feature, "search": test_search})
    accuracy = _tracking_accuracy(
        dataset,
        validation_specs,
        args.config.resolve(),
        backbone_fp,
        tracking_fp,
        backbone_int8,
        tracking_int8,
    )
    pair_details = accuracy.pop("pairs")
    (output_dir / "accuracy_pair_details.json").write_text(
        json.dumps(pair_details, indent=2), encoding="utf-8"
    )
    report = {
        "backbone_graph": _inspect(backbone_int8),
        "tracking_graph": _inspect(tracking_int8),
        "backbone_error": _compare(backbone_fp, backbone_int8, {"template": test_template}),
        "tracking_error": _compare(tracking_fp, tracking_int8, {"template_features": fp_feature, "search": test_search}),
        "end_to_end_error": {
            f"output_{i}_mae": float(np.mean(np.abs(a - b)))
            for i, (a, b) in enumerate(zip(fp_tracking_outputs, int8_tracking_outputs))
        },
        "tracking_accuracy": accuracy,
    }
    if not args.skip_benchmark:
        benchmark_warmup = min(args.benchmark_warmup, 2) if args.smoke else args.benchmark_warmup
        benchmark_runs = min(args.benchmark_runs, 5) if args.smoke else args.benchmark_runs
        report["performance_comparison"] = _benchmark_pair(
            backbone_fp, tracking_fp, backbone_int8, tracking_int8,
            test_template, test_search,
            benchmark_warmup, benchmark_runs, args.benchmark_threads,
        )
    (output_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved {backbone_int8}")
    print(f"Saved {tracking_int8}")


if __name__ == "__main__":
    main()
