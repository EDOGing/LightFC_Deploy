"""Independent LightFC deployment pipeline: PyTorch AdaRound -> pnnx -> ncnn INT8.

This script does not modify the existing ONNX pipeline.  Every generated file is
written below --output-dir (default: quantized/ncnn).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Iterable

import cv2
import numpy as np
import torch

from export_onnx_split import TemplateBackbone, TrackingNetwork
from quantization.adaround import apply_adaround
from quantization.config import QuantizationConfig
from quantization.got10k import GOT10kPairs, PairSpec
from web_demo import DEFAULT_CHECKPOINT, DEFAULT_CONFIG, LightFCCPU, sample_target
from lib.test.utils.hann import hann2d
from lib.utils.box_ops import clip_box


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "quantized" / "ncnn"


class FusionOnly(torch.nn.Module):
    def __init__(self, network): super().__init__(); self.fusion = network.fusion
    def forward(self, template_features, search_features): return self.fusion(template_features, search_features)


class HeadOnly(torch.nn.Module):
    def __init__(self, network): super().__init__(); self.head = network.head
    def forward(self, fused):
        output = self.head(fused)
        return output["score_map"], output["size_map"], output["offset_map"]


class HeadBranch(torch.nn.Module):
    def __init__(self, network, kind: str):
        super().__init__(); head = network.head; self.kind = kind
        suffix = {"score": "ctr", "size": "size", "offset": "offset"}[kind]
        self.conv1 = getattr(head, f"conv1_{suffix}"); self.se = getattr(head, f"se_{suffix}")
        self.conv2 = getattr(head, f"conv2_{suffix}"); self.conv3 = getattr(head, f"conv3_{suffix}")
        self.conv4 = getattr(head, f"conv4_{suffix}"); self.conv5 = getattr(head, f"conv5_{suffix}")

    def forward(self, x):
        x = self.conv1(x); x = self.se(x); x = self.conv2(x); x = self.conv3(x); x = self.conv4(x); x = self.conv5(x)
        return torch.clamp(torch.sigmoid(x), 1e-4, 1 - 1e-4) if self.kind in {"score", "size"} else x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LightFC AdaRound + pnnx + ncnn INT8 pipeline")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--got10k", type=Path, default=Path(r"F:\dataset\got10k\train"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pnnx", type=Path, default=None, help="pnnx executable; auto-detected when omitted")
    parser.add_argument("--ncnn-tools", type=Path, default=None, help="Directory containing ncnn2table and ncnn2int8")
    parser.add_argument("--ncnn2table", type=Path, default=None)
    parser.add_argument("--ncnn2int8", type=Path, default=None)
    parser.add_argument("--calibration-method", choices=["kl", "aciq"], default="kl")
    parser.add_argument("--calibration-pairs", type=int, default=256)
    parser.add_argument("--validation-pairs", type=int, default=64)
    parser.add_argument("--calibration-sequences", type=int, default=256)
    parser.add_argument("--validation-sequences", type=int, default=64)
    parser.add_argument("--adaround-pairs-per-layer", type=int, default=16)
    parser.add_argument("--adaround-iterations", type=int, default=2000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse-adaround", type=Path, default=None, help="Load a completed AdaRound checkpoint instead of optimizing")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--benchmark-warmup", type=int, default=20)
    parser.add_argument("--benchmark-runs", type=int, default=200)
    parser.add_argument("--evaluate-only", action="store_true", help="Evaluate existing FP32/INT8 ncnn files")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _find_executable(explicit: Path | None, directory: Path | None, name: str) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    if directory is not None:
        candidates.extend((directory / name, directory / f"{name}.exe"))
        if directory.is_dir():
            discovered = list(directory.rglob(f"{name}.exe"))
            # The official Windows archive contains arm64 and x64 tools.  This
            # project's Python environment is x64, so prefer that toolchain.
            discovered.sort(key=lambda path: ("x64" not in {part.lower() for part in path.parts}, str(path)))
            candidates.extend(discovered)
    found = shutil.which(name) or shutil.which(f"{name}.exe")
    if found:
        candidates.append(Path(found))
    if name == "pnnx":
        try:
            import pnnx  # type: ignore
            package = Path(pnnx.__file__).resolve().parent
            candidates.extend((package / "pnnx", package / "pnnx.exe"))
        except ImportError:
            pass
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate.is_file():
            return candidate
    option = "--pnnx" if name == "pnnx" else f"--{name} or --ncnn-tools"
    raise FileNotFoundError(f"Cannot find {name}. Install/build it, then pass {option}.")


def _run(command: list[str], cwd: Path) -> None:
    print("RUN:", subprocess.list2cmdline(command))
    subprocess.run(command, cwd=cwd, check=True)


def _save_torchscript(network: torch.nn.Module, template: torch.Tensor, search: torch.Tensor, output: Path) -> tuple[Path, Path]:
    backbone = TemplateBackbone(network).cpu().eval()
    tracking = TrackingNetwork(network).cpu().eval()
    with torch.no_grad():
        feature = backbone(template).clone()
        backbone_ts = torch.jit.freeze(torch.jit.trace(backbone, template, strict=False))
        tracking_ts = torch.jit.freeze(torch.jit.trace(tracking, (feature, search), strict=False))
    backbone_path, tracking_path = output / "lightfc_template.pt", output / "lightfc_tracking.pt"
    torch.jit.save(backbone_ts, str(backbone_path))
    torch.jit.save(tracking_ts, str(tracking_path))
    return backbone_path, tracking_path


def _save_segmented_torchscript(network, template, search, output: Path) -> dict[str, Path]:
    template_net = TemplateBackbone(network).cpu().eval()
    search_net = TemplateBackbone(network).cpu().eval()
    fusion_net = FusionOnly(network).cpu().eval()
    head_nets = {name: HeadBranch(network, name).cpu().eval() for name in ("score", "size", "offset")}
    with torch.no_grad():
        z = template_net(template).clone(); x = search_net(search).clone(); fused = fusion_net(z, x).clone()
        modules = {
            "template": torch.jit.freeze(torch.jit.trace(template_net, template, strict=False)),
            "search": torch.jit.freeze(torch.jit.trace(search_net, search, strict=False)),
            "fusion": torch.jit.freeze(torch.jit.trace(fusion_net, (z, x), strict=False)),
            **{name: torch.jit.freeze(torch.jit.trace(module, fused, strict=False)) for name, module in head_nets.items()},
        }
    result = {}
    for name, module in modules.items():
        path = output / f"lightfc_{name}.pt"; torch.jit.save(module, str(path)); result[name] = path
    return result


def _convert_pnnx(pnnx: Path, model: Path, input_shapes: list[list[int]]) -> tuple[Path, Path]:
    shape_arg = "inputshape=" + ",".join("[" + ",".join(str(v) for v in shape) + "]" for shape in input_shapes)
    # Disable pnnx's default FP16 weight storage so the unquantized baseline is
    # genuinely FP32 and model-size/memory comparisons have an honest baseline.
    _run([str(pnnx), str(model), shape_arg, "fp16=0"], model.parent)
    param, binary = model.with_suffix(".ncnn.param"), model.with_suffix(".ncnn.bin")
    if not param.is_file() or not binary.is_file():
        raise RuntimeError(f"pnnx did not generate {param.name} and {binary.name}")
    return param, binary


def _ncnn_io_names(param_path: Path) -> tuple[list[str], list[str]]:
    lines = [line.strip() for line in param_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    produced, consumed, inputs = [], set(), []
    for line in lines[2:]:
        fields = line.split()
        if len(fields) < 4:
            continue
        bottom_count, top_count = int(fields[2]), int(fields[3])
        bottoms = fields[4:4 + bottom_count]
        tops = fields[4 + bottom_count:4 + bottom_count + top_count]
        consumed.update(bottoms)
        produced.extend(tops)
        if fields[0] == "Input":
            inputs.extend(tops)
    outputs = [name for name in produced if name not in consumed]

    def order(name: str):
        suffix = name[3:] if name.startswith(("in", "out")) else ""
        return (0, int(suffix)) if suffix.isdigit() else (1, name)

    return sorted(dict.fromkeys(inputs), key=order), sorted(dict.fromkeys(outputs), key=order)


def _load_ncnn_module():
    try:
        import ncnn  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Python package 'ncnn' is required for feature generation, accuracy, speed and memory evaluation") from exc
    return ncnn


class NcnnSplitModel:
    def __init__(self, backbone_param: Path, backbone_bin: Path, tracking_param: Path, tracking_bin: Path, threads: int):
        ncnn = _load_ncnn_module()
        self.ncnn = ncnn
        self.backbone_inputs, self.backbone_outputs = _ncnn_io_names(backbone_param)
        self.tracking_inputs, self.tracking_outputs = _ncnn_io_names(tracking_param)
        if len(self.backbone_inputs) != 1 or len(self.backbone_outputs) != 1:
            raise RuntimeError("Unexpected template ncnn input/output graph")
        if len(self.tracking_inputs) != 2 or len(self.tracking_outputs) != 3:
            raise RuntimeError("Unexpected tracking ncnn input/output graph")
        self.backbone = ncnn.Net()
        self.tracking = ncnn.Net()
        self.backbone.opt.use_vulkan_compute = False
        self.tracking.opt.use_vulkan_compute = False
        self.backbone.opt.num_threads = threads
        self.tracking.opt.num_threads = threads
        self.backbone.load_param(str(backbone_param))
        self.backbone.load_model(str(backbone_bin))
        self.tracking.load_param(str(tracking_param))
        self.tracking.load_model(str(tracking_bin))

    @staticmethod
    def _extract(extractor, name: str):
        value = extractor.extract(name)
        if isinstance(value, tuple):
            code, value = value
            if code != 0:
                raise RuntimeError(f"ncnn extract {name} failed: {code}")
        return value

    def template(self, array: np.ndarray) -> np.ndarray:
        ex = self.backbone.create_extractor()
        ex.input(self.backbone_inputs[0], self.ncnn.Mat(np.ascontiguousarray(array[0], dtype=np.float32)))
        return np.asarray(self._extract(ex, self.backbone_outputs[0]), dtype=np.float32).copy()

    def track(self, feature: np.ndarray, search: np.ndarray) -> list[np.ndarray]:
        ex = self.tracking.create_extractor()
        ex.input(self.tracking_inputs[0], self.ncnn.Mat(np.ascontiguousarray(feature, dtype=np.float32)))
        ex.input(self.tracking_inputs[1], self.ncnn.Mat(np.ascontiguousarray(search[0], dtype=np.float32)))
        return [np.asarray(self._extract(ex, name), dtype=np.float32).copy() for name in self.tracking_outputs]


class NcnnSegmentedModel:
    """Six physical ncnn graphs with explicit FP32 boundaries around correlation."""
    def __init__(self, files: dict[str, tuple[Path, Path]], threads: int):
        self.ncnn = _load_ncnn_module(); self.nets = {}; self.io = {}
        for name, (param, binary) in files.items():
            inputs, outputs = _ncnn_io_names(param); net = self.ncnn.Net()
            net.opt.use_vulkan_compute = False; net.opt.num_threads = threads
            net.load_param(str(param)); net.load_model(str(binary))
            self.nets[name] = net; self.io[name] = (inputs, outputs)

    @staticmethod
    def _extract(ex, name):
        value = ex.extract(name)
        if isinstance(value, tuple):
            code, value = value
            if code != 0: raise RuntimeError(f"ncnn extract {name} failed: {code}")
        return np.asarray(value, dtype=np.float32).copy()

    def _single(self, name, arrays):
        ex = self.nets[name].create_extractor(); inputs, outputs = self.io[name]
        for input_name, array in zip(inputs, arrays):
            ex.input(input_name, self.ncnn.Mat(np.ascontiguousarray(array, dtype=np.float32)))
        return [self._extract(ex, output_name) for output_name in outputs]

    def template(self, array): return self._single("template", [array[0]])[0]
    def search(self, array): return self._single("search", [array[0]])[0]
    def fuse(self, z, x): return self._single("fusion", [z, x])[0]
    def head(self, fused): return [self._single(name, [fused])[0] for name in ("score", "size", "offset")]
    def track(self, feature, search): return self.head(self.fuse(feature, self.search(search)))


def _write_list(path: Path, values: Iterable[Path]) -> None:
    path.write_text("\n".join(str(value.resolve()) for value in values) + "\n", encoding="utf-8")


def _prepare_calibration(model: NcnnSplitModel, dataset: GOT10kPairs, specs: list[PairSpec], output: Path) -> tuple[Path, Path, Path]:
    directory = output / "calibration_npy"
    directory.mkdir(parents=True, exist_ok=True)
    template_files, feature_files, search_files = [], [], []
    for index, (template, search) in enumerate(dataset.iter_arrays(specs)):
        template_path = directory / f"{index:05d}_template.npy"
        feature_path = directory / f"{index:05d}_template_features.npy"
        search_path = directory / f"{index:05d}_search.npy"
        np.save(template_path, template[0])
        np.save(feature_path, model.template(template))
        np.save(search_path, search[0])
        template_files.append(template_path)
        feature_files.append(feature_path)
        search_files.append(search_path)
        print(f"ncnn calibration data {index + 1}/{len(specs)}")
    template_list, feature_list, search_list = output / "template_npy.list", output / "feature_npy.list", output / "search_npy.list"
    _write_list(template_list, template_files)
    _write_list(feature_list, feature_files)
    _write_list(search_list, search_files)
    return template_list, feature_list, search_list


def _prepare_segmented_calibration(model: NcnnSegmentedModel, dataset, specs, output):
    directory = output / "calibration_npy"; directory.mkdir(parents=True, exist_ok=True)
    groups = {name: [] for name in ("template", "search", "template_features", "search_features", "fused")}
    for index, (template, search) in enumerate(dataset.iter_arrays(specs)):
        z, x = model.template(template), model.search(search); fused = model.fuse(z, x)
        arrays = {"template": template[0], "search": search[0], "template_features": z, "search_features": x, "fused": fused}
        for name, array in arrays.items():
            path = directory / f"{index:05d}_{name}.npy"; np.save(path, array); groups[name].append(path)
        print(f"ncnn calibration data {index + 1}/{len(specs)}")
    lists = {}
    for name, files in groups.items():
        path = output / f"{name}_npy.list"; _write_list(path, files); lists[name] = path
    return lists


def _quantize_segmented(args, tools, fp_model, dataset, specs, paths):
    ncnn2table, ncnn2int8 = tools
    lists = _prepare_segmented_calibration(fp_model, dataset, specs, paths["output"])
    configs = {
        "template": (lists["template"], "shape=[128,128,3]"),
        "search": (lists["search"], "shape=[256,256,3]"),
        "score": (lists["fused"], "shape=[16,16,192]"),
        "size": (lists["fused"], "shape=[16,16,192]"),
        "offset": (lists["fused"], "shape=[16,16,192]"),
    }
    for name, (input_list, shape) in configs.items():
        _run([str(ncnn2table), str(paths[f"fp_{name}_param"]), str(paths[f"fp_{name}_bin"]), str(input_list),
              str(paths[f"{name}_table"]), shape, f"thread={args.threads}", f"method={args.calibration_method}", "type=1"], paths["output"])
        fallback = _sanitize_nonfinite_scales(paths[f"{name}_table"])
        print(f"{name} FP32 fallback for invalid scales:", fallback)
        boundary = _preserve_head_boundaries(paths[f"fp_{name}_param"], paths[f"{name}_table"]) if name in {"score", "size", "offset"} else []
        if boundary:
            print(f"{name} FP32 boundary layers:", boundary)
        _run([str(ncnn2int8), str(paths[f"fp_{name}_param"]), str(paths[f"fp_{name}_bin"]),
              str(paths[f"int8_{name}_param"]), str(paths[f"int8_{name}_bin"]), str(paths[f"{name}_table"])], paths["output"])


def _preserve_head_boundaries(param_path: Path, table_path: Path) -> list[str]:
    """Keep SE/residual mixing and the final predictor in FP32.

    Current ncnn Windows builds can terminate inside a head when INT8 tensors cross
    the SE BinaryOp path.  Quantizing the three middle 3x3 convolutions is stable,
    while the explicit float boundaries also avoid quantizing sigmoid/box outputs.
    """
    layers = []
    for line in param_path.read_text(encoding="utf-8").splitlines()[2:]:
        fields = line.split()
        if len(fields) >= 4:
            layers.append((fields[0], fields[1]))
    mul_index = next((i for i, (kind, _) in enumerate(layers) if kind == "BinaryOp"), -1)
    convolution_indices = [i for i, (kind, _) in enumerate(layers) if kind in {"Convolution", "ConvolutionDepthWise"}]
    if mul_index < 0 or not convolution_indices:
        return []
    keep = {name for i, (kind, name) in enumerate(layers)
            if kind in {"Convolution", "ConvolutionDepthWise"} and i < mul_index}
    keep.add(layers[convolution_indices[-1]][1])
    updated = []
    for line in table_path.read_text(encoding="utf-8").splitlines():
        token = line.split(maxsplit=1)[0] if line.strip() else ""
        if any(token == name or token.startswith(name + "_") for name in keep):
            line = "# " + line
        updated.append(line)
    table_path.write_text("\n".join(updated) + "\n", encoding="utf-8")
    return sorted(keep)


def _preserve_dynamic_matmul(param_path: Path, table_path: Path) -> list[str]:
    nodes = []
    producer, consumers = {}, {}
    for line in param_path.read_text(encoding="utf-8").splitlines()[2:]:
        fields = line.split()
        if len(fields) < 4:
            continue
        bc, tc = int(fields[2]), int(fields[3])
        node = {"type": fields[0], "name": fields[1], "bottoms": fields[4:4 + bc], "tops": fields[4 + bc:4 + bc + tc]}
        nodes.append(node)
        for top in node["tops"]: producer[top] = node
        for bottom in node["bottoms"]: consumers.setdefault(bottom, []).append(node)
    matmuls = [node for node in nodes if node["type"].lower() in {"matmul", "gemm"}]
    float_layers = [node["name"] for node in matmuls]
    convolution_types = {"convolution", "convolutiondepthwise", "deconvolution", "deconvolutiondepthwise"}

    def walk_backward(blob: str, seen: set[str]):
        node = producer.get(blob)
        if node is None or node["name"] in seen: return
        seen.add(node["name"])
        if node["type"].lower() in convolution_types:
            # ncnn2int8 may otherwise propagate packed int8 through reshape /
            # residual paths into a dynamic MatMul. Keep its complete dynamic
            # input subgraph floating, not only the immediately adjacent conv.
            float_layers.append(node["name"])
        for bottom in node["bottoms"]: walk_backward(bottom, seen)

    def walk_forward(blob: str, seen: set[str]):
        for node in consumers.get(blob, []):
            if node["name"] in seen: continue
            seen.add(node["name"])
            if node["type"].lower() in convolution_types:
                float_layers.append(node["name"]); continue
            for top in node["tops"]: walk_forward(top, seen)

    for node in matmuls:
        for bottom in node["bottoms"]: walk_backward(bottom, set())
        for top in node["tops"]: walk_forward(top, set())
    float_layers = list(dict.fromkeys(float_layers))
    if not float_layers:
        return []
    updated = []
    for line in table_path.read_text(encoding="utf-8").splitlines():
        stripped = line.lstrip()
        if any(stripped.startswith(name + "_") or stripped.startswith(name + " ") for name in float_layers):
            line = "# " + line
        updated.append(line)
    table_path.write_text("\n".join(updated) + "\n", encoding="utf-8")
    return float_layers


def _sanitize_nonfinite_scales(table_path: Path) -> list[str]:
    """Keep layers with inf/nan calibration scales in FP32 instead of emitting a broken INT8 graph."""
    lines = table_path.read_text(encoding="utf-8").splitlines()
    invalid_layers: set[str] = set()
    for line in lines:
        fields = line.split()
        if fields and any(value.lower() in {"inf", "+inf", "-inf", "nan", "+nan", "-nan"} for value in fields[1:]):
            invalid_layers.add(fields[0].removesuffix("_param_0"))
    if invalid_layers:
        updated = []
        for line in lines:
            token = line.split(maxsplit=1)[0] if line.strip() else ""
            if any(token == name or token.startswith(name + "_") for name in invalid_layers):
                line = "# " + line
            updated.append(line)
        table_path.write_text("\n".join(updated) + "\n", encoding="utf-8")
    return sorted(invalid_layers)


def _quantize_ncnn(args, tools: tuple[Path, Path], fp_model: NcnnSplitModel, dataset: GOT10kPairs, specs: list[PairSpec], paths: dict[str, Path]) -> None:
    ncnn2table, ncnn2int8 = tools
    template_list, feature_list, search_list = _prepare_calibration(fp_model, dataset, specs, paths["output"])
    _run([
        str(ncnn2table), str(paths["fp_backbone_param"]), str(paths["fp_backbone_bin"]), str(template_list),
        str(paths["backbone_table"]), "shape=[128,128,3]", f"thread={args.threads}",
        f"method={args.calibration_method}", "type=1",
    ], paths["output"])
    _run([
        str(ncnn2table), str(paths["fp_tracking_param"]), str(paths["fp_tracking_bin"]),
        f"{feature_list},{search_list}", str(paths["tracking_table"]),
        "shape=[8,8,96],[256,256,3]", f"thread={args.threads}",
        f"method={args.calibration_method}", "type=1",
    ], paths["output"])
    float_layers = _preserve_dynamic_matmul(paths["fp_tracking_param"], paths["tracking_table"])
    print("FP32 correlation layers:", float_layers or "dynamic MatMul not represented by a table weight row")
    fallback_backbone = _sanitize_nonfinite_scales(paths["backbone_table"])
    fallback_tracking = _sanitize_nonfinite_scales(paths["tracking_table"])
    print("FP32 fallback for invalid scales:", {"template": fallback_backbone, "tracking": fallback_tracking})
    _run([str(ncnn2int8), str(paths["fp_backbone_param"]), str(paths["fp_backbone_bin"]), str(paths["int8_backbone_param"]), str(paths["int8_backbone_bin"]), str(paths["backbone_table"])], paths["output"])
    _run([str(ncnn2int8), str(paths["fp_tracking_param"]), str(paths["fp_tracking_bin"]), str(paths["int8_tracking_param"]), str(paths["int8_tracking_bin"]), str(paths["tracking_table"])], paths["output"])


def _iou(a, b) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    inter = max(0.0, min(ax2, bx2) - max(a[0], b[0])) * max(0.0, min(ay2, by2) - max(a[1], b[1]))
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


class NcnnTracker:
    def __init__(self, model: NcnnSplitModel, torch_tracker: LightFCCPU):
        self.model, self.reference = model, torch_tracker
        self.mean = np.asarray(torch_tracker.cfg.DATA.MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.asarray(torch_tracker.cfg.DATA.STD, dtype=np.float32).reshape(1, 3, 1, 1)
        self.window = hann2d(torch.tensor([torch_tracker.feat_size, torch_tracker.feat_size]), centered=True).numpy()

    def _preprocess(self, rgb):
        value = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32)
        return (value / 255.0 - self.mean) / self.std

    def run_pair(self, template_bgr, search_bgr, initial_box):
        template_rgb = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2RGB)
        patch, _ = sample_target(template_rgb, initial_box, self.reference.template_factor, self.reference.template_size)
        feature = self.model.template(self._preprocess(patch))
        search_rgb = cv2.cvtColor(search_bgr, cv2.COLOR_BGR2RGB)
        patch, resize = sample_target(search_rgb, initial_box, self.reference.search_factor, self.reference.search_size)
        outputs = self.model.track(feature, self._preprocess(patch))
        score = outputs[0].reshape(1, 1, self.reference.feat_size, self.reference.feat_size)
        size = outputs[1].reshape(1, 2, self.reference.feat_size, self.reference.feat_size)
        offset = outputs[2].reshape(1, 2, self.reference.feat_size, self.reference.feat_size)
        response = score * self.window
        iy, ix = divmod(int(np.argmax(response)), self.reference.feat_size)
        cx = (ix + float(offset[0, 0, iy, ix])) / self.reference.feat_size
        cy = (iy + float(offset[0, 1, iy, ix])) / self.reference.feat_size
        bw, bh = float(size[0, 0, iy, ix]), float(size[0, 1, iy, ix])
        cx, cy, bw, bh = [v * self.reference.search_size / resize for v in (cx, cy, bw, bh)]
        pcx, pcy = initial_box[0] + initial_box[2] / 2, initial_box[1] + initial_box[3] / 2
        half = 0.5 * self.reference.search_size / resize
        mapped = [cx + pcx - half - bw / 2, cy + pcy - half - bh / 2, bw, bh]
        return [float(v) for v in clip_box(mapped, search_bgr.shape[0], search_bgr.shape[1], margin=2)]


def _accuracy(fp_model, int8_model, tracker, dataset, specs):
    models = {"fp32": NcnnTracker(fp_model, tracker), "int8": NcnnTracker(int8_model, tracker)}
    values = {name: {"iou": [], "error": []} for name in models}
    for index, spec in enumerate(specs, 1):
        template, search, initial, target = dataset.load_raw(spec)
        for name, model in models.items():
            prediction = model.run_pair(template, search, initial)
            values[name]["iou"].append(_iou(prediction, target))
            pc = (prediction[0] + prediction[2] / 2, prediction[1] + prediction[3] / 2)
            tc = (target[0] + target[2] / 2, target[1] + target[3] / 2)
            values[name]["error"].append(float(np.hypot(pc[0] - tc[0], pc[1] - tc[1])))
        print(f"ncnn accuracy {index}/{len(specs)}")
    result = {}
    for name, item in values.items():
        ious, errors = np.asarray(item["iou"]), np.asarray(item["error"])
        result[name] = {
            "mean_iou": float(ious.mean()),
            "success_auc": float(np.mean([(ious >= t).mean() for t in np.linspace(0, 1, 21)])),
            "precision_20px": float((errors <= 20).mean()),
            "mean_center_error_px": float(errors.mean()),
        }
    result["int8_minus_fp32"] = {key: result["int8"][key] - result["fp32"][key] for key in result["fp32"]}
    result["pair_count"] = len(specs)
    return result


class _Memory(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong), ("peak_rss", ctypes.c_size_t), ("rss", ctypes.c_size_t),
                ("qpp", ctypes.c_size_t), ("qp", ctypes.c_size_t), ("qnpp", ctypes.c_size_t), ("qnp", ctypes.c_size_t),
                ("pagefile", ctypes.c_size_t), ("peak_pagefile", ctypes.c_size_t), ("private", ctypes.c_size_t)]


def _memory() -> tuple[int, int]:
    value = _Memory(); value.cb = ctypes.sizeof(value)
    kernel, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Memory), ctypes.c_ulong]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(value), value.cb):
        raise ctypes.WinError()
    return value.rss, value.private


def _benchmark_worker(queue, files, template, search, warmup, runs, threads):
    try:
        baseline = _memory()
        model = NcnnSplitModel(*(Path(x) for x in files), threads)
        feature = model.template(template)
        for _ in range(warmup): model.template(template); model.track(feature, search)
        start = time.perf_counter()
        for _ in range(runs): model.template(template)
        template_time = time.perf_counter() - start
        start = time.perf_counter()
        for _ in range(runs): model.track(feature, search)
        tracking_time = time.perf_counter() - start
        current = _memory()
        queue.put({"template_latency_ms": template_time * 1000 / runs, "tracking_latency_ms": tracking_time * 1000 / runs,
                   "tracking_fps": runs / tracking_time, "working_set_increment_mib": max(0, current[0] - baseline[0]) / 1024**2,
                   "private_memory_increment_mib": max(0, current[1] - baseline[1]) / 1024**2})
    except Exception as exc:
        queue.put({"error": repr(exc)})


def _benchmark(paths, template, search, warmup, runs, threads):
    context, result = mp.get_context("spawn"), {}
    groups = {"fp32": [paths[k] for k in ("fp_backbone_param", "fp_backbone_bin", "fp_tracking_param", "fp_tracking_bin")],
              "int8": [paths[k] for k in ("int8_backbone_param", "int8_backbone_bin", "int8_tracking_param", "int8_tracking_bin")]}
    for name, files in groups.items():
        queue = context.Queue(); process = context.Process(target=_benchmark_worker, args=(queue, [str(x) for x in files], template, search, warmup, runs, threads))
        process.start(); process.join()
        item = queue.get()
        if process.exitcode or "error" in item: raise RuntimeError(f"{name} benchmark failed: {item}")
        item["model_size_mib"] = sum(path.stat().st_size for path in files) / 1024**2
        result[name] = item
    fp, q = result["fp32"], result["int8"]
    result["int8_vs_fp32"] = {"tracking_speedup": q["tracking_fps"] / fp["tracking_fps"],
        "model_size_reduction_percent": (1 - q["model_size_mib"] / fp["model_size_mib"]) * 100,
        "working_set_reduction_percent": (1 - q["working_set_increment_mib"] / fp["working_set_increment_mib"]) * 100 if fp["working_set_increment_mib"] else 0,
        "private_memory_reduction_percent": (1 - q["private_memory_increment_mib"] / fp["private_memory_increment_mib"]) * 100 if fp["private_memory_increment_mib"] else 0}
    result["settings"] = {"warmup": warmup, "runs": runs, "threads": threads}
    return result


def _segmented_file_group(paths, quantized: bool):
    prefix = "int8" if quantized else "fp"
    return {
        "template": (paths[f"{prefix}_template_param"], paths[f"{prefix}_template_bin"]),
        "search": (paths[f"{prefix}_search_param"], paths[f"{prefix}_search_bin"]),
        "fusion": (paths["fp_fusion_param"], paths["fp_fusion_bin"]),
        "score": (paths[f"{prefix}_score_param"], paths[f"{prefix}_score_bin"]),
        "size": (paths[f"{prefix}_size_param"], paths[f"{prefix}_size_bin"]),
        "offset": (paths[f"{prefix}_offset_param"], paths[f"{prefix}_offset_bin"]),
    }


def _segmented_benchmark_worker(queue, serialized, template, search, warmup, runs, threads):
    try:
        baseline = _memory(); files = {k: (Path(v[0]), Path(v[1])) for k, v in serialized.items()}
        model = NcnnSegmentedModel(files, threads); feature = model.template(template)
        for _ in range(warmup): model.template(template); model.track(feature, search)
        start = time.perf_counter()
        for _ in range(runs): model.template(template)
        template_time = time.perf_counter() - start; start = time.perf_counter()
        for _ in range(runs): model.track(feature, search)
        tracking_time = time.perf_counter() - start; current = _memory()
        queue.put({"template_latency_ms": template_time * 1000 / runs, "tracking_latency_ms": tracking_time * 1000 / runs,
                   "tracking_fps": runs / tracking_time, "working_set_increment_mib": max(0, current[0] - baseline[0]) / 1024**2,
                   "private_memory_increment_mib": max(0, current[1] - baseline[1]) / 1024**2})
    except Exception as exc: queue.put({"error": repr(exc)})


def _benchmark_segmented(paths, template, search, warmup, runs, threads):
    context, result = mp.get_context("spawn"), {}
    for name, quantized in (("fp32", False), ("int8", True)):
        files = _segmented_file_group(paths, quantized); serialized = {k: (str(v[0]), str(v[1])) for k, v in files.items()}
        queue = context.Queue(); process = context.Process(target=_segmented_benchmark_worker, args=(queue, serialized, template, search, warmup, runs, threads))
        process.start(); process.join(); item = queue.get()
        if process.exitcode or "error" in item: raise RuntimeError(f"{name} benchmark failed: {item}")
        item["model_size_mib"] = sum(p.stat().st_size for pair in files.values() for p in pair) / 1024**2; result[name] = item
    fp, q = result["fp32"], result["int8"]
    result["int8_vs_fp32"] = {"tracking_speedup": q["tracking_fps"] / fp["tracking_fps"],
        "model_size_reduction_percent": (1 - q["model_size_mib"] / fp["model_size_mib"]) * 100,
        "working_set_reduction_percent": (1 - q["working_set_increment_mib"] / fp["working_set_increment_mib"]) * 100 if fp["working_set_increment_mib"] else 0,
        "private_memory_reduction_percent": (1 - q["private_memory_increment_mib"] / fp["private_memory_increment_mib"]) * 100 if fp["private_memory_increment_mib"] else 0}
    result["settings"] = {"warmup": warmup, "runs": runs, "threads": threads}; return result


def main() -> None:
    args = parse_args(); output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    tracker = LightFCCPU(args.checkpoint.resolve(), args.config.resolve())
    cfg = QuantizationConfig(calibration_pairs=args.calibration_pairs, validation_pairs=args.validation_pairs,
        calibration_sequences=args.calibration_sequences, validation_sequences=args.validation_sequences,
        adaround_pairs_per_layer=args.adaround_pairs_per_layer, adaround_iterations=args.adaround_iterations)
    if args.smoke:
        cfg.calibration_sequences, cfg.validation_sequences, cfg.calibration_pairs, cfg.validation_pairs = 4, 2, 2, 1
        cfg.adaround_pairs_per_layer, cfg.adaround_iterations = 1, 2
        args.benchmark_warmup, args.benchmark_runs = 2, 5
    cfg.validate()
    dataset = GOT10kPairs(args.got10k, tracker.template_factor, tracker.template_size, tracker.search_factor, tracker.search_size,
        list(tracker.cfg.DATA.MEAN), list(tracker.cfg.DATA.STD), cfg.seed, cfg.calibration_sequences, cfg.validation_sequences,
        cfg.max_frame_gap, cfg.min_visible_ratio, output / "got10k_split_manifest.json")
    pair_manifest = output / "got10k_pair_manifest.json"
    if args.evaluate_only and pair_manifest.is_file():
        saved = json.loads(pair_manifest.read_text(encoding="utf-8")); calibration_specs = [PairSpec(**x) for x in saved["calibration"]]; validation_specs = [PairSpec(**x) for x in saved["validation"]]
    else:
        calibration_specs = dataset.specs("calibration", cfg.calibration_pairs, cfg.seed + 1); validation_specs = dataset.specs("validation", cfg.validation_pairs, cfg.seed + 2)
        pair_manifest.write_text(json.dumps({"calibration": [x.__dict__ for x in calibration_specs], "validation": [x.__dict__ for x in validation_specs]}, indent=2), encoding="utf-8")
    paths = {"output": output}
    for name in ("template", "search", "fusion", "score", "size", "offset"):
        paths[f"fp_{name}_param"] = output / f"lightfc_{name}.ncnn.param"
        paths[f"fp_{name}_bin"] = output / f"lightfc_{name}.ncnn.bin"
    for name in ("template", "search", "score", "size", "offset"):
        paths[f"int8_{name}_param"] = output / f"lightfc_{name}_int8.ncnn.param"
        paths[f"int8_{name}_bin"] = output / f"lightfc_{name}_int8.ncnn.bin"
        paths[f"{name}_table"] = output / f"lightfc_{name}.table"
    if not args.evaluate_only:
        if args.reuse_adaround:
            saved = torch.load(args.reuse_adaround.resolve(), map_location="cpu", weights_only=True); tracker.network.load_state_dict(saved["model"], strict=True)
        else:
            def pairs():
                for t, s in dataset.iter_arrays(calibration_specs[:cfg.adaround_pairs_per_layer]): yield torch.from_numpy(t), torch.from_numpy(s)
            apply_adaround(tracker.network, pairs, cfg, output / "adaround_progress.pth", args.resume)
        template, search = dataset.load(calibration_specs[0]); template_t, search_t = torch.from_numpy(template), torch.from_numpy(search)
        scripts = _save_segmented_torchscript(tracker.network, template_t, search_t, output)
        pnnx = _find_executable(args.pnnx, None, "pnnx")
        shapes = {"template": [[1, 3, 128, 128]], "search": [[1, 3, 256, 256]],
                  "fusion": [[1, 96, 8, 8], [1, 96, 16, 16]],
                  "score": [[1, 192, 16, 16]], "size": [[1, 192, 16, 16]], "offset": [[1, 192, 16, 16]]}
        for name, script in scripts.items():
            generated = _convert_pnnx(pnnx, script, shapes[name])
            for source, suffix in zip(generated, ("param", "bin")):
                target = paths[f"fp_{name}_{suffix}"]
                if source.resolve() != target.resolve(): shutil.copy2(source, target)
        fp_model = NcnnSegmentedModel(_segmented_file_group(paths, False), args.threads)
        tools = (_find_executable(args.ncnn2table, args.ncnn_tools, "ncnn2table"), _find_executable(args.ncnn2int8, args.ncnn_tools, "ncnn2int8"))
        _quantize_segmented(args, tools, fp_model, dataset, calibration_specs, paths)
    required = [path for key, path in paths.items() if key != "output" and not key.endswith("_table")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing: raise FileNotFoundError("Missing ncnn model files: " + ", ".join(missing))
    fp_model = NcnnSegmentedModel(_segmented_file_group(paths, False), args.threads)
    int8_model = NcnnSegmentedModel(_segmented_file_group(paths, True), args.threads)
    template, search = dataset.load(validation_specs[0])
    report = {
        "deployment": {
            "baseline": "FP32 ncnn (pnnx fp16=0)",
            "quantized": "native ncnn INT8 with explicit FP32 safety boundaries",
            "int8_graphs": ["template backbone", "search backbone", "score middle convolutions",
                            "size middle convolutions", "offset middle convolutions"],
            "fp32_graphs_or_boundaries": ["pixel-wise correlation/fusion", "SE residual mixing",
                                           "final score/size/offset predictors"],
        },
        "accuracy": _accuracy(fp_model, int8_model, tracker, dataset, validation_specs),
        "performance": _benchmark_segmented(paths, template, search, args.benchmark_warmup, args.benchmark_runs, args.threads),
        "note": "Held-out GOT-10k train frame pairs; not the official GOT-10k test set.",
    }
    (output / "ncnn_comparison_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2)); print("Saved:", output)


if __name__ == "__main__":
    mp.freeze_support()
    main()
