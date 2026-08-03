from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .config import QuantizationConfig


def _get_module(root: nn.Module, name: str) -> nn.Module:
    module = root
    for part in name.split("."):
        module = getattr(module, part)
    return module


def _capture_layer(
    model: nn.Module,
    layer: nn.Conv2d,
    pair_factory: Callable[[], Iterable[tuple[torch.Tensor, torch.Tensor]]],
    count: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    cache: list[tuple[torch.Tensor, torch.Tensor]] = []

    def hook(_module, inputs, output):
        if len(cache) < count:
            cache.append((inputs[0].detach().cpu(), output.detach().cpu()))

    handle = layer.register_forward_hook(hook)
    try:
        with torch.no_grad():
            for template, search in pair_factory():
                z = model.forward_backbone(template)
                model.forward_tracking(z, search)
                if len(cache) >= count:
                    break
    finally:
        handle.remove()
    return cache


def _adaround_layer(layer: nn.Conv2d, cache: list[tuple[torch.Tensor, torch.Tensor]], cfg: QuantizationConfig) -> None:
    weight = layer.weight.detach().float()
    axes = tuple(range(1, weight.ndim))
    scale = weight.abs().amax(dim=axes, keepdim=True).clamp_min(1e-8) / 127.0
    scaled = weight / scale
    floor = torch.floor(scaled).clamp(-128, 127)
    rest = (scaled - floor).clamp(0, 1)
    gamma, zeta = -0.1, 1.1
    p = ((rest - gamma) / (zeta - gamma)).clamp(1e-6, 1 - 1e-6)
    alpha = nn.Parameter(torch.log(p / (1 - p)))
    optimizer = torch.optim.Adam([alpha], lr=cfg.adaround_learning_rate)
    warmup = int(cfg.adaround_iterations * cfg.adaround_warmup_fraction)

    for step in range(cfg.adaround_iterations):
        h = (torch.sigmoid(alpha) * (zeta - gamma) + gamma).clamp(0, 1)
        rounded = (floor + h).clamp(-128, 127) * scale
        x, target = cache[step % len(cache)]
        output = F.conv2d(x, rounded, layer.bias, layer.stride, layer.padding, layer.dilation, layer.groups)
        reconstruction = F.mse_loss(output, target)
        if step >= warmup:
            progress = (step - warmup) / max(1, cfg.adaround_iterations - warmup - 1)
            beta = cfg.adaround_beta_start + progress * (cfg.adaround_beta_end - cfg.adaround_beta_start)
            regularizer = (1 - (2 * h - 1).abs().pow(beta)).mean()
            loss = reconstruction + cfg.adaround_regularization * regularizer
        else:
            loss = reconstruction
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    hard = (alpha.detach() >= 0).to(weight.dtype)
    layer.weight.data.copy_((floor + hard).clamp(-128, 127) * scale)


def apply_adaround(
    model: nn.Module,
    pair_factory: Callable[[], Iterable[tuple[torch.Tensor, torch.Tensor]]],
    cfg: QuantizationConfig,
    checkpoint_path: Path | None = None,
    resume: bool = False,
) -> list[str]:
    """Optimize rounding for every Conv2d. Correlation/MatMul is untouched."""
    names = [name for name, module in model.named_modules() if isinstance(module, nn.Conv2d)]
    completed: list[str] = []
    if resume and checkpoint_path is not None and checkpoint_path.is_file():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"], strict=True)
        completed = list(saved.get("completed", []))
        print(f"Resumed AdaRound after {len(completed)} layers")
    for index, name in enumerate(names, 1):
        if name in completed:
            continue
        layer = _get_module(model, name)
        cache = _capture_layer(model, layer, pair_factory, cfg.adaround_pairs_per_layer)
        if not cache:
            raise RuntimeError(f"No calibration activation captured for {name}")
        print(f"AdaRound [{index}/{len(names)}] {name}, samples={len(cache)}")
        _adaround_layer(layer, cache, cfg)
        completed.append(name)
        if checkpoint_path is not None:
            torch.save({"model": model.state_dict(), "completed": completed}, checkpoint_path)
    return names
