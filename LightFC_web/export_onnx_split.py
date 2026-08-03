"""Export LightFC as a cached template backbone and per-frame tracking ONNX pair."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from web_demo import DEFAULT_CHECKPOINT, DEFAULT_CONFIG, LightFCCPU


ROOT = Path(__file__).resolve().parent
DEFAULT_BACKBONE_OUTPUT = ROOT / "lightfc_ep0400_backbone.onnx"
DEFAULT_TRACKING_OUTPUT = ROOT / "lightfc_ep0400_tracking.onnx"


class TemplateBackbone(nn.Module):
    def __init__(self, network: nn.Module):
        super().__init__()
        self.network = network

    def forward(self, template: torch.Tensor) -> torch.Tensor:
        return self.network.forward_backbone(template)


class TrackingNetwork(nn.Module):
    def __init__(self, network: nn.Module):
        super().__init__()
        self.network = network

    def forward(self, template_features: torch.Tensor, search: torch.Tensor):
        output = self.network.forward_tracking(template_features, search)
        return output["score_map"], output["size_map"], output["offset_map"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export split LightFC ONNX models")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--backbone-output", default=str(DEFAULT_BACKBONE_OUTPUT))
    parser.add_argument("--tracking-output", default=str(DEFAULT_TRACKING_OUTPUT))
    parser.add_argument("--opset", type=int, default=12)
    return parser.parse_args()


def save_metadata(path: Path, kind: str) -> None:
    model = onnx.load(str(path))
    values = {
        "model": "LightFC MobileNetV2",
        "part": kind,
        "device": "CPU compatible",
        "normalization": "RGB float32 NCHW; mean=0.485,0.456,0.406; std=0.229,0.224,0.225",
    }
    for key, value in values.items():
        item = model.metadata_props.add()
        item.key, item.value = key, value
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def verify(
    backbone_path: Path,
    tracking_path: Path,
    backbone: nn.Module,
    tracking: nn.Module,
    template: torch.Tensor,
    search: torch.Tensor,
) -> None:
    with torch.inference_mode():
        expected_features = backbone(template).numpy()
        expected_outputs = [value.numpy() for value in tracking(torch.from_numpy(expected_features), search)]

    backbone_session = ort.InferenceSession(str(backbone_path), providers=["CPUExecutionProvider"])
    tracking_session = ort.InferenceSession(str(tracking_path), providers=["CPUExecutionProvider"])
    actual_features = backbone_session.run(None, {"template": template.numpy()})[0]
    feature_error = float(np.max(np.abs(expected_features - actual_features)))
    if not np.allclose(expected_features, actual_features, rtol=1e-3, atol=1e-5):
        raise RuntimeError(f"Backbone verification failed: max error={feature_error}")
    print(f"verified template_features shape={list(actual_features.shape)} max_abs_error={feature_error:.8g}")

    actual_outputs = tracking_session.run(
        None, {"template_features": actual_features, "search": search.numpy()}
    )
    for name, expected, actual in zip(
        ("score_map", "size_map", "offset_map"), expected_outputs, actual_outputs
    ):
        max_error = float(np.max(np.abs(expected - actual)))
        if not np.allclose(expected, actual, rtol=1e-3, atol=1e-5):
            raise RuntimeError(f"Tracking verification failed for {name}: max error={max_error}")
        print(f"verified {name:12s} shape={list(actual.shape)} max_abs_error={max_error:.8g}")


def main() -> None:
    args = parse_args()
    backbone_path = Path(args.backbone_output).resolve()
    tracking_path = Path(args.tracking_output).resolve()
    backbone_path.parent.mkdir(parents=True, exist_ok=True)
    tracking_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(2026)
    tracker = LightFCCPU(Path(args.checkpoint), Path(args.config))
    backbone = TemplateBackbone(tracker.network).cpu().eval()
    tracking = TrackingNetwork(tracker.network).cpu().eval()
    template = torch.randn(1, 3, tracker.template_size, tracker.template_size)
    search = torch.randn(1, 3, tracker.search_size, tracker.search_size)
    with torch.no_grad():
        template_features = backbone(template)

    torch.onnx.export(
        backbone,
        (template,),
        str(backbone_path),
        input_names=["template"],
        output_names=["template_features"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    torch.onnx.export(
        tracking,
        (template_features, search),
        str(tracking_path),
        input_names=["template_features", "search"],
        output_names=["score_map", "size_map", "offset_map"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    save_metadata(backbone_path, "template_backbone")
    save_metadata(tracking_path, "search_backbone_fusion_head")
    verify(backbone_path, tracking_path, backbone, tracking, template, search)
    print(f"saved: {backbone_path} ({backbone_path.stat().st_size / 1024 / 1024:.2f} MiB)")
    print(f"saved: {tracking_path} ({tracking_path.stat().st_size / 1024 / 1024:.2f} MiB)")


if __name__ == "__main__":
    main()
