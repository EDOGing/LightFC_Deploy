"""Compare the original LightFC PyTorch checkpoint with the FP32 ncnn export.

The two runtimes are evaluated on the same held-out GOT-10k pairs.  Timing and
memory are measured in separate child processes to avoid cross-runtime residue.
This file is independent and does not modify either deployment pipeline.
"""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import torch

from quantization.config import QuantizationConfig
from quantization.got10k import GOT10kPairs, PairSpec
from quantize_lightfc_ncnn import (
    DEFAULT_OUTPUT,
    NcnnSegmentedModel,
    NcnnTracker,
    _iou,
    _memory,
    _segmented_file_group,
)
from web_demo import DEFAULT_CHECKPOINT, DEFAULT_CONFIG, LightFCCPU


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare original PTH and FP32 ncnn on CPU")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--got10k", type=Path, default=Path(r"F:\dataset\got10k\train"))
    parser.add_argument("--ncnn-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--validation-pairs", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--benchmark-repeats", type=int, default=3)
    return parser.parse_args()


def ncnn_paths(directory: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name in ("template", "search", "fusion", "score", "size", "offset"):
        paths[f"fp_{name}_param"] = directory / f"lightfc_{name}.ncnn.param"
        paths[f"fp_{name}_bin"] = directory / f"lightfc_{name}.ncnn.bin"
    return paths


def build_dataset(args: argparse.Namespace, tracker: LightFCCPU) -> tuple[GOT10kPairs, list[PairSpec]]:
    cfg = QuantizationConfig(validation_pairs=args.validation_pairs)
    manifest = args.ncnn_dir / "got10k_pair_manifest.json"
    dataset = GOT10kPairs(
        args.got10k,
        tracker.template_factor,
        tracker.template_size,
        tracker.search_factor,
        tracker.search_size,
        list(tracker.cfg.DATA.MEAN),
        list(tracker.cfg.DATA.STD),
        cfg.seed,
        cfg.calibration_sequences,
        cfg.validation_sequences,
        cfg.max_frame_gap,
        cfg.min_visible_ratio,
        args.ncnn_dir / "got10k_split_manifest.json",
    )
    if manifest.is_file():
        saved = json.loads(manifest.read_text(encoding="utf-8"))["validation"]
        specs = [PairSpec(**item) for item in saved[: args.validation_pairs]]
    else:
        specs = dataset.specs("validation", args.validation_pairs, cfg.seed + 2)
    if len(specs) != args.validation_pairs:
        raise ValueError(f"Requested {args.validation_pairs} pairs, but the manifest has {len(specs)}")
    return dataset, specs


def summarize(iou_values: list[float], errors: list[float]) -> dict[str, float]:
    ious = np.asarray(iou_values)
    center_errors = np.asarray(errors)
    return {
        "mean_iou": float(ious.mean()),
        "success_auc": float(np.mean([(ious >= threshold).mean() for threshold in np.linspace(0, 1, 21)])),
        "precision_20px": float((center_errors <= 20).mean()),
        "mean_center_error_px": float(center_errors.mean()),
    }


def evaluate_accuracy(
    checkpoint: Path,
    config: Path,
    ncnn_files: dict[str, tuple[Path, Path]],
    dataset: GOT10kPairs,
    specs: list[PairSpec],
    threads: int,
) -> dict:
    torch.set_num_threads(threads)
    pth = LightFCCPU(checkpoint, config)
    ncnn = NcnnTracker(NcnnSegmentedModel(ncnn_files, threads), pth)
    values = {name: {"iou": [], "error": []} for name in ("pth", "ncnn_fp32")}
    for index, spec in enumerate(specs, 1):
        template, search, initial, target = dataset.load_raw(spec)
        pth.initialize(template, initial)
        pth_box, _ = pth.track(search)
        ncnn_box = ncnn.run_pair(template, search, initial)
        for name, prediction in (("pth", pth_box), ("ncnn_fp32", ncnn_box)):
            values[name]["iou"].append(_iou(prediction, target))
            predicted_center = (prediction[0] + prediction[2] / 2, prediction[1] + prediction[3] / 2)
            target_center = (target[0] + target[2] / 2, target[1] + target[3] / 2)
            values[name]["error"].append(float(np.hypot(
                predicted_center[0] - target_center[0], predicted_center[1] - target_center[1]
            )))
        print(f"PTH vs ncnn accuracy {index}/{len(specs)}")
    result = {name: summarize(item["iou"], item["error"]) for name, item in values.items()}
    result["ncnn_minus_pth"] = {key: result["ncnn_fp32"][key] - result["pth"][key] for key in result["pth"]}
    result["pair_count"] = len(specs)
    return result


def benchmark_worker(queue, kind: str, files, checkpoint, config, template, search, warmup, runs, threads):
    try:
        torch.set_num_threads(threads)
        baseline = _memory()
        if kind == "pth":
            model = LightFCCPU(Path(checkpoint), Path(config))
            template_tensor = torch.from_numpy(template)
            search_tensor = torch.from_numpy(search)
            with torch.inference_mode():
                feature = model.network.forward_backbone(template_tensor)
                for _ in range(warmup):
                    model.network.forward_backbone(template_tensor)
                    model.network.forward_tracking(feature, search_tensor)
                start = time.perf_counter()
                for _ in range(runs):
                    model.network.forward_backbone(template_tensor)
                template_seconds = time.perf_counter() - start
                start = time.perf_counter()
                for _ in range(runs):
                    model.network.forward_tracking(feature, search_tensor)
                tracking_seconds = time.perf_counter() - start
            model_size = Path(checkpoint).stat().st_size
        else:
            model = NcnnSegmentedModel(
                {name: (Path(pair[0]), Path(pair[1])) for name, pair in files.items()}, threads
            )
            feature = model.template(template)
            for _ in range(warmup):
                model.template(template)
                model.track(feature, search)
            start = time.perf_counter()
            for _ in range(runs):
                model.template(template)
            template_seconds = time.perf_counter() - start
            start = time.perf_counter()
            for _ in range(runs):
                model.track(feature, search)
            tracking_seconds = time.perf_counter() - start
            model_size = sum(Path(path).stat().st_size for pair in files.values() for path in pair)
        current = _memory()
        queue.put({
            "template_latency_ms": template_seconds * 1000 / runs,
            "tracking_latency_ms": tracking_seconds * 1000 / runs,
            "tracking_fps": runs / tracking_seconds,
            "working_set_increment_mib": max(0, current[0] - baseline[0]) / 1024**2,
            "private_memory_increment_mib": max(0, current[1] - baseline[1]) / 1024**2,
            "model_size_mib": model_size / 1024**2,
        })
    except Exception as exc:
        queue.put({"error": repr(exc)})


def benchmark(args, files, template, search) -> dict:
    context = mp.get_context("spawn")
    serialized = {name: (str(pair[0]), str(pair[1])) for name, pair in files.items()}
    result = {}
    for name, file_argument in (("pth", {}), ("ncnn_fp32", serialized)):
        samples = []
        for _ in range(args.benchmark_repeats):
            queue = context.Queue()
            process = context.Process(target=benchmark_worker, args=(
                queue, name, file_argument, str(args.checkpoint), str(args.config), template, search,
                args.warmup, args.runs, args.threads,
            ))
            process.start()
            process.join()
            if process.exitcode != 0 or queue.empty():
                raise RuntimeError(f"{name} benchmark process exited with code {process.exitcode}")
            item = queue.get()
            if "error" in item:
                raise RuntimeError(f"{name} benchmark failed: {item['error']}")
            samples.append(item)
        keys = samples[0].keys()
        result[name] = {key: float(np.median([sample[key] for sample in samples])) for key in keys}
        result[name]["samples"] = samples
    pth, ncnn = result["pth"], result["ncnn_fp32"]
    result["ncnn_vs_pth"] = {
        "tracking_speedup": ncnn["tracking_fps"] / pth["tracking_fps"],
        "tracking_latency_reduction_percent": (1 - ncnn["tracking_latency_ms"] / pth["tracking_latency_ms"]) * 100,
        "checkpoint_to_ncnn_size_reduction_percent": (1 - ncnn["model_size_mib"] / pth["model_size_mib"]) * 100,
        "working_set_reduction_percent": (1 - ncnn["working_set_increment_mib"] / pth["working_set_increment_mib"]) * 100 if pth["working_set_increment_mib"] else 0,
        "private_memory_reduction_percent": (1 - ncnn["private_memory_increment_mib"] / pth["private_memory_increment_mib"]) * 100 if pth["private_memory_increment_mib"] else 0,
    }
    result["settings"] = {"warmup": args.warmup, "runs": args.runs, "threads": args.threads,
                          "independent_process_repeats": args.benchmark_repeats, "aggregate": "median"}
    return result


def main() -> None:
    args = parse_args()
    args.checkpoint, args.config, args.ncnn_dir = (path.resolve() for path in (args.checkpoint, args.config, args.ncnn_dir))
    report_path = args.report.resolve() if args.report else args.ncnn_dir / "pth_vs_fp32_ncnn_report.json"
    reference = LightFCCPU(args.checkpoint, args.config)
    dataset, specs = build_dataset(args, reference)
    paths = ncnn_paths(args.ncnn_dir)
    files = _segmented_file_group(paths, False)
    missing = [str(path) for pair in files.values() for path in pair if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing FP32 ncnn files: " + ", ".join(missing))
    template, search = dataset.load(specs[0])
    deployed_state = io.BytesIO()
    torch.save(reference.network.state_dict(), deployed_state)
    minimal_ncnn_bytes = sum(
        path.stat().st_size for name, pair in files.items() if name != "search" for path in pair
    )
    report = {
        "comparison_scope": {
            "pth": "Original training checkpoint and original PyTorch inference graph",
            "ncnn_fp32": "AdaRound-adjusted weights exported by pnnx with fp16=0",
            "storage_note": "PTH size includes optimizer and training metadata; ncnn size includes deployment param/bin only.",
        },
        "storage": {
            "original_training_checkpoint_mib": args.checkpoint.stat().st_size / 1024**2,
            "pytorch_deployed_state_dict_mib": len(deployed_state.getvalue()) / 1024**2,
            "ncnn_six_graph_package_mib": sum(path.stat().st_size for pair in files.values() for path in pair) / 1024**2,
            "ncnn_minimal_shared_backbone_package_mib": minimal_ncnn_bytes / 1024**2,
            "minimal_package_note": "template and search backbone files are byte-identical, so C++ may share one pair.",
        },
        "accuracy": evaluate_accuracy(args.checkpoint, args.config, files, dataset, specs, args.threads),
        "performance": benchmark(args, files, template, search),
        "note": "Held-out GOT-10k train frame pairs; not the official GOT-10k test set.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Saved:", report_path)


if __name__ == "__main__":
    mp.freeze_support()
    main()
