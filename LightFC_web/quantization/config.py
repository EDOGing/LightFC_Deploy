from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QuantizationConfig:
    """All tunable PTQ parameters. Change A/B/C here or use CLI flags."""

    # Activation strategy: A=Percentile, B=MSE, C=per-tensor hybrid.
    activation_strategy: str = "C"
    percentile: float = 99.99
    mse_percentile_min: float = 99.0
    mse_percentile_max: float = 100.0
    mse_candidates: int = 41

    seed: int = 2026
    calibration_sequences: int = 256
    validation_sequences: int = 64
    calibration_pairs: int = 256
    validation_pairs: int = 64
    max_frame_gap: int = 100
    min_visible_ratio: float = 0.25

    # AdaRound. 2000 is a practical CPU default; use 10000 for the paper-style run.
    adaround_pairs_per_layer: int = 16
    adaround_iterations: int = 2000
    adaround_learning_rate: float = 1e-3
    adaround_regularization: float = 0.01
    adaround_warmup_fraction: float = 0.2
    adaround_beta_start: float = 20.0
    adaround_beta_end: float = 2.0

    activation_values_per_tensor: int = 262_144
    activation_values_per_batch: int = 32_768
    opset: int = 12
    quantize_ops: list[str] = field(default_factory=lambda: ["Conv"])

    def validate(self) -> None:
        self.activation_strategy = self.activation_strategy.upper()
        if self.activation_strategy not in {"A", "B", "C"}:
            raise ValueError("activation_strategy must be A, B, or C")
        if self.opset != 12:
            raise ValueError("This pipeline intentionally targets ONNX opset 12")
        if not 0 < self.percentile <= 100:
            raise ValueError("percentile must be in (0, 100]")
        if self.calibration_pairs < 1 or self.adaround_pairs_per_layer < 1:
            raise ValueError("sample counts must be positive")

