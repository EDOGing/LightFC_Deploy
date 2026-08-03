# LightFC W8A8 静态量化

入口脚本：`quantize_lightfc_w8a8.py`。执行设备固定为 CPU，最终生成两个 ONNX opset 12 模型：

- `lightfc_w8a8_backbone_opset12.onnx`：初始化时运行一次的模板分支
- `lightfc_w8a8_tracking_opset12.onnx`：每帧运行的搜索/融合/head 分支

卷积采用 W8A8：激活为 per-tensor `uint8`，权重为 per-output-channel 对称 `int8`。opset 12 使用 `QLinearConv`；pixel-wise correlation 的 `MatMul`、Sigmoid 和框解码相关算子保持 FP32。

## 正式运行

```powershell
D:\anaconda3\envs\SwinTrack\python.exe -u quantize_lightfc_w8a8.py `
  --got10k "F:\dataset\got10k\train" `
  --strategy C `
  --output-dir "quantized"
```

AdaRound 是逐层优化，在 CPU 上可能运行很久。脚本会在每层完成后更新 `adaround_progress.pth`；中断后使用同一输出目录并增加 `--resume`：

```powershell
D:\anaconda3\envs\SwinTrack\python.exe -u quantize_lightfc_w8a8.py `
  --got10k "F:\dataset\got10k\train" `
  --strategy C `
  --output-dir "quantized" `
  --resume
```

## A / B / C 参数

集中修改 `quantization/config.py` 中的 `QuantizationConfig`：

- A：固定 Percentile；主要参数 `percentile`，默认 99.99。
- B：以量化重建 MSE 搜索裁剪阈值；参数为 `mse_percentile_min`、`mse_percentile_max`、`mse_candidates`。
- C：对每个激活张量分别计算 A/B，并选择重建误差更小者；选择结果保存到 `activation_strategy_choices.json`。
- AdaRound：`adaround_pairs_per_layer`、`adaround_iterations`、学习率、正则强度、warmup 和 beta 调度均可修改。
- 数据：默认 256 个标定 pair、64 个留出验证 pair；序列和帧对会写入 manifest，随机种子固定为 2026。

`--smoke` 仅用于检查工程链路，样本过少，不能用于评估量化精度。`--skip-adaround` 仅用于排错，不是最终方案。

## Web 使用

现有两个 Web 程序已经支持指定一对 ONNX 路径。正式量化完成后可传入：

```powershell
D:\anaconda3\envs\SwinTrack\python.exe web_demo.py `
  --onnx-backbone "quantized\lightfc_w8a8_backbone_opset12.onnx" `
  --onnx-tracking "quantized\lightfc_w8a8_tracking_opset12.onnx"
```

RTSP 页面同样把入口换为 `rtsp_web_demo.py`。页面内选择 ONNX 后即使用量化模型。

## 输出与验收

除两个部署模型外，还会生成：

- GOT-10k split/pair manifest
- AdaRound 断点
- 两个激活标定缓存
- C 策略逐张量选择记录
- `validation_report.json`（ONNX 合法性、算子计数、FP32/INT8 输出 MAE）

