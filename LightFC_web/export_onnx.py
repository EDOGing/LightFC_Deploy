"""Export the LightFC checkpoint to a single, CPU-compatible ONNX model."""

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
DEFAULT_OUTPUT = ROOT / "lightfc_ep0400.onnx"


class LightFCOnnxWrapper(nn.Module):
    """Expose LightFC's two image tensors and four inference outputs."""

    def __init__(self, network: nn.Module):
        super().__init__()
        self.network = network

    def forward(self, template: torch.Tensor, search: torch.Tensor):
        template_features = self.network.forward_backbone(template)
        output = self.network.forward_tracking(template_features, search)
        return (
            output["score_map"],
            output["size_map"],
            output["offset_map"],
            output["pred_boxes"],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export LightFC checkpoint to ONNX")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def add_metadata(model_path: Path) -> None:
    model = onnx.load(str(model_path))
    metadata = {
        "model": "LightFC MobileNetV2",
        "device": "CPU compatible",
        "template_input": "float32 NCHW [1,3,128,128], RGB ImageNet-normalized",
        "search_input": "float32 NCHW [1,3,256,256], RGB ImageNet-normalized",
        "normalization_mean": "0.485,0.456,0.406",
        "normalization_std": "0.229,0.224,0.225",
        "bbox_format": "pred_boxes: normalized cx,cy,width,height in search crop",
    }
    del model.metadata_props[:]
    for key, value in metadata.items():
        item = model.metadata_props.add()
        item.key, item.value = key, value
    onnx.checker.check_model(model)
    onnx.save(model, str(model_path))


def verify_runtime(
    model_path: Path,
    wrapper: nn.Module,
    template: torch.Tensor,
    search: torch.Tensor,
) -> None:
    with torch.inference_mode():
        torch_outputs = [value.detach().cpu().numpy() for value in wrapper(template, search)]
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    ort_outputs = session.run(
        None,
        {"template": template.numpy(), "search": search.numpy()},
    )
    names = ("score_map", "size_map", "offset_map", "pred_boxes")
    for name, expected, actual in zip(names, torch_outputs, ort_outputs):
        max_error = float(np.max(np.abs(expected - actual)))
        if not np.allclose(expected, actual, rtol=1e-3, atol=1e-5):
            raise RuntimeError(f"ONNX Runtime verification failed for {name}: max error={max_error}")
        print(f"verified {name:12s} shape={list(actual.shape)} max_abs_error={max_error:.8g}")


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(2026)
    tracker = LightFCCPU(Path(args.checkpoint), Path(args.config))
    wrapper = LightFCOnnxWrapper(tracker.network).cpu().eval()
    template = torch.randn(1, 3, tracker.template_size, tracker.template_size)
    search = torch.randn(1, 3, tracker.search_size, tracker.search_size)

    torch.onnx.export(
        wrapper,
        (template, search),
        str(output_path),
        input_names=["template", "search"],
        output_names=["score_map", "size_map", "offset_map", "pred_boxes"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    add_metadata(output_path)
    verify_runtime(output_path, wrapper, template, search)
    print(f"saved: {output_path}")
    print(f"size: {output_path.stat().st_size / 1024 / 1024:.2f} MiB")


if __name__ == "__main__":
    main()
