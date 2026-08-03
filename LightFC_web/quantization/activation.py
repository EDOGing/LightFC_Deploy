from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, shape_inference
from onnxruntime.quantization.calibrate import CalibrationMethod, TensorData, TensorsData
from onnxruntime.quantization.quantize import save_tensors_data

from .config import QuantizationConfig


def _activation_names(model: onnx.ModelProto) -> list[str]:
    initializers = {value.name for value in model.graph.initializer}
    names: list[str] = []
    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        if node.input and node.input[0] not in initializers:
            names.append(node.input[0])
        names.extend(node.output)
    return list(dict.fromkeys(names))


def _augment_outputs(model_path: Path, output_path: Path) -> list[str]:
    model = shape_inference.infer_shapes(onnx.load(str(model_path)))
    names = _activation_names(model)
    known = {value.name: value for value in [*model.graph.input, *model.graph.value_info, *model.graph.output]}
    existing = {value.name for value in model.graph.output}
    for name in names:
        if name not in existing:
            model.graph.output.append(known.get(name, helper.make_tensor_value_info(name, TensorProto.FLOAT, None)))
    onnx.save(model, str(output_path))
    return names


def collect_activation_values(
    model_path: Path,
    feeds: Iterable[dict[str, np.ndarray]],
    cfg: QuantizationConfig,
    augmented_path: Path,
) -> dict[str, np.ndarray]:
    names = _augment_outputs(model_path, augmented_path)
    session = ort.InferenceSession(str(augmented_path), providers=["CPUExecutionProvider"])
    chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    totals: dict[str, int] = defaultdict(int)
    rng = np.random.default_rng(cfg.seed)
    for sample_index, feed in enumerate(feeds, 1):
        values = session.run(names, feed)
        for name, value in zip(names, values):
            flat = np.asarray(value, dtype=np.float32).reshape(-1)
            if flat.size > cfg.activation_values_per_batch:
                flat = flat[rng.choice(flat.size, cfg.activation_values_per_batch, replace=False)]
            remaining = cfg.activation_values_per_tensor - totals[name]
            if remaining > 0:
                kept = flat if flat.size <= remaining else flat[rng.choice(flat.size, remaining, replace=False)]
                chunks[name].append(kept.copy())
                totals[name] += kept.size
        print(f"Activation calibration: sample {sample_index}")
    return {name: np.concatenate(parts) for name, parts in chunks.items() if parts}


def _range_percentile(values: np.ndarray, percentile: float) -> tuple[float, float]:
    tail = (100.0 - percentile) / 2.0
    low, high = np.percentile(values, [tail, 100.0 - tail])
    return min(float(low), 0.0), max(float(high), 0.0)


def _quant_mse(values: np.ndarray, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    scale = (high - low) / 255.0
    zero = np.clip(np.rint(-low / scale), 0, 255)
    restored = (np.clip(np.rint(values / scale) + zero, 0, 255) - zero) * scale
    return float(np.mean((values - restored) ** 2))


def _range_mse(values: np.ndarray, cfg: QuantizationConfig) -> tuple[float, float]:
    best_range, best_error = None, float("inf")
    for percentile in np.linspace(cfg.mse_percentile_min, cfg.mse_percentile_max, cfg.mse_candidates):
        candidate = _range_percentile(values, float(percentile))
        error = _quant_mse(values, *candidate)
        if error < best_error:
            best_range, best_error = candidate, error
    assert best_range is not None
    return best_range


def choose_ranges(values: dict[str, np.ndarray], cfg: QuantizationConfig) -> tuple[dict[str, tuple[float, float]], dict[str, str]]:
    ranges, choices = {}, {}
    for name, samples in values.items():
        a = _range_percentile(samples, cfg.percentile)
        b = _range_mse(samples, cfg)
        if cfg.activation_strategy == "A":
            selected, choice = a, "Percentile"
        elif cfg.activation_strategy == "B":
            selected, choice = b, "MSE"
        else:
            error_a, error_b = _quant_mse(samples, *a), _quant_mse(samples, *b)
            selected, choice = (a, "Percentile") if error_a <= error_b else (b, "MSE")
        # Avoid a zero-width range rejected by some execution providers.
        low, high = selected
        if high - low < 1e-8:
            high = low + 1e-8
        ranges[name], choices[name] = (low, high), choice
    return ranges, choices


def save_range_cache(ranges: dict[str, tuple[float, float]], path: Path) -> None:
    data = {
        name: TensorData(lowest=np.asarray(low, dtype=np.float32), highest=np.asarray(high, dtype=np.float32))
        for name, (low, high) in ranges.items()
    }
    save_tensors_data(TensorsData(CalibrationMethod.MinMax, data), path)

