# LightFC CPU Web、ONNX 与 ncnn 部署

本项目基于 LightFC，面向 CPU 单目标跟踪，主要包含三个部分：PyTorch Web Demo、ONNX 转换与 W8A8 量化，以及 ncnn/C++ 部署与量化。模板特征只在初始化目标时计算一次，之后每帧只处理搜索区域。

> 当前测试环境：Windows、Python `D:\anaconda3\envs\SwinTrack\python.exe`、CPU 推理。

## 1. PTH 单目标跟踪 Web Demo

### 离线视频

[web_demo.py](./web_demo.py) 使用 `lightfc_ep0400.pth.tar` 在 CPU 上完成：

- 上传或选择本地视频；
- 在首帧手动框选目标；
- 单目标跟踪、实时预览和推理速率显示；
- 取消任务；
- 导出结果视频和 bbox CSV。

启动：

```powershell
D:\anaconda3\envs\SwinTrack\python.exe web_demo.py --open-browser
```

也可以双击 `run_web_demo.bat`，默认页面为 <http://127.0.0.1:7860>。

### RTSP 实时视频流

[rtsp_web_demo.py](./rtsp_web_demo.py) 用于摄像机实时跟踪，支持：

- 默认地址 `rtsp://192.168.6.116:554/live/av1`，也可在页面填写其他 RTSP 地址；
- 在实时画面中手动框选、重新框选和取消目标；
- 断线重连、截图、置信度、视频帧率和推理速率显示；
- PyTorch、ONNX FP32 和 ONNX INT8 后端切换。

启动：

```powershell
D:\anaconda3\envs\SwinTrack\python.exe rtsp_web_demo.py --open-browser
```

也可以双击 `run_rtsp_web_demo.bat`，默认页面为 <http://127.0.0.1:7861>。

## 2. PTH → ONNX 与 W8A8 静态量化

ONNX 部署采用“模板模型 + 跟踪模型”拆分结构：

```text
初始化：template image → backbone → 缓存 template_features
逐帧：  template_features + search image → tracking → score/size/offset
```

推理图不输出 `pred_boxes`，只输出 `score_map`、`size_map` 和 `offset_map`，最终 bbox 在模型外解码。导出目标为 ONNX opset 12。

FP32 导出：

```powershell
D:\anaconda3\envs\SwinTrack\python.exe export_onnx_split.py --opset 12
```

W8A8 静态量化方案：

```text
Percentile/MSE 激活标定
    └─ 每层分别计算 Percentile 与 MSE 候选范围，选择量化误差更小者
+ AdaRound 权重舍入
    └─ 使用 GOT-10k train 标定样本逐层优化权重舍入
```

运行量化：

```powershell
D:\anaconda3\envs\SwinTrack\python.exe quantize_lightfc_w8a8.py `
  --got10k F:\dataset\got10k\train `
  --output-dir quantized
```

量化脚本同时比较 ONNX FP32 与 ONNX INT8 的精度、跟踪延迟/FPS、内存和模型大小。结果见 [quantized/validation_report.json](./quantized/validation_report.json)。该精度结果使用隔离的 GOT-10k train 帧对，不是 GOT-10k 官方 test 评测结果。

## 3. PTH → TorchScript/pnnx → ncnn

ncnn 路线为：

```text
PTH → PyTorch AdaRound → TorchScript → pnnx → FP32 ncnn → ncnn native INT8
```

为明确控制精度和兼容性，物理上拆为 template、search、fusion、score、size、offset 六个 ncnn 图；逻辑上仍然是模板初始化和逐帧跟踪两个阶段。Pixel-wise correlation/fusion 保持 FP32，INT8 图还为 SE 混合与最终预测层保留 FP32 安全边界。

生成模型：

```powershell
D:\anaconda3\envs\SwinTrack\python.exe quantize_lightfc_ncnn.py `
  --checkpoint lightfc_ep0400.pth.tar `
  --got10k F:\dataset\got10k\train `
  --output-dir quantized\ncnn `
  --ncnn-tools quantized\ncnn\tools
```

ncnn FP32/INT8 对比见 [ncnn_comparison_report.json](./quantized/ncnn/ncnn_comparison_report.json)，原始 PTH/FP32 ncnn 对比见 [pth_vs_fp32_ncnn_report.json](./quantized/ncnn/pth_vs_fp32_ncnn_report.json)。C++ 串并行关系和 bbox 解码参考 [lightfc_fp32_ncnn_cpp_pseudocode.cpp](./quantized/ncnn/lightfc_fp32_ncnn_cpp_pseudocode.cpp)。

## 模型位置与下载

以下均为相对于项目根目录的路径。若本项目上传到 GitHub/GitLab，点击文件名即可打开或下载；体积较大的文件建议通过 Git LFS 或 Release 附件发布。

### 原始 PyTorch 模型

| 模型 | 仓库路径/下载 | 说明 |
|---|---|---|
| LightFC PTH | [lightfc_ep0400.pth.tar](./lightfc_ep0400.pth.tar) | 本项目 Web Demo 的默认模型 |
| 官方 LightFC checkpoint | [Google Drive](https://drive.google.com/file/d/1ns7NQJCt078547X483skqjX1qM1rBqLP/view) | 原项目公开下载地址；下载后按需要重命名或修改启动参数 |

### ONNX 模型

| 类型 | 模板/Backbone | 逐帧 Tracking |
|---|---|---|
| ONNX FP32 | [lightfc_ep0400_backbone.onnx](./lightfc_ep0400_backbone.onnx) | [lightfc_ep0400_tracking.onnx](./lightfc_ep0400_tracking.onnx) |
| AdaRound FP32 对照 | [lightfc_adaround_fp32_backbone_opset12.onnx](./quantized/lightfc_adaround_fp32_backbone_opset12.onnx) | [lightfc_adaround_fp32_tracking_opset12.onnx](./quantized/lightfc_adaround_fp32_tracking_opset12.onnx) |
| ONNX W8A8 INT8 | [lightfc_w8a8_backbone_opset12.onnx](./quantized/lightfc_w8a8_backbone_opset12.onnx) | [lightfc_w8a8_tracking_opset12.onnx](./quantized/lightfc_w8a8_tracking_opset12.onnx) |

早期整图模型 `lightfc_ep0400.onnx` 已不再保留。实时部署请使用上表中的 backbone/tracking 拆分模型，以缓存模板特征。

仓库仅保留最终部署模型和验证报告。量化 smoke 输出、标定缓存、AdaRound 进度文件及 TorchScript/pnnx 转换中间产物可通过脚本重新生成，因此不纳入版本管理。

### ncnn FP32 模型（C++ 推荐，12 个文件）

| 子图 | `.param` | `.bin` |
|---|---|---|
| Template | [lightfc_template.ncnn.param](./quantized/ncnn/lightfc_template.ncnn.param) | [lightfc_template.ncnn.bin](./quantized/ncnn/lightfc_template.ncnn.bin) |
| Search | [lightfc_search.ncnn.param](./quantized/ncnn/lightfc_search.ncnn.param) | [lightfc_search.ncnn.bin](./quantized/ncnn/lightfc_search.ncnn.bin) |
| Fusion | [lightfc_fusion.ncnn.param](./quantized/ncnn/lightfc_fusion.ncnn.param) | [lightfc_fusion.ncnn.bin](./quantized/ncnn/lightfc_fusion.ncnn.bin) |
| Score | [lightfc_score.ncnn.param](./quantized/ncnn/lightfc_score.ncnn.param) | [lightfc_score.ncnn.bin](./quantized/ncnn/lightfc_score.ncnn.bin) |
| Size | [lightfc_size.ncnn.param](./quantized/ncnn/lightfc_size.ncnn.param) | [lightfc_size.ncnn.bin](./quantized/ncnn/lightfc_size.ncnn.bin) |
| Offset | [lightfc_offset.ncnn.param](./quantized/ncnn/lightfc_offset.ncnn.param) | [lightfc_offset.ncnn.bin](./quantized/ncnn/lightfc_offset.ncnn.bin) |

### ncnn INT8 模型

INT8 部署使用以下量化模型，并继续使用 FP32 fusion：

| 子图 | `.param` | `.bin` |
|---|---|---|
| Template INT8 | [lightfc_template_int8.ncnn.param](./quantized/ncnn/lightfc_template_int8.ncnn.param) | [lightfc_template_int8.ncnn.bin](./quantized/ncnn/lightfc_template_int8.ncnn.bin) |
| Search INT8 | [lightfc_search_int8.ncnn.param](./quantized/ncnn/lightfc_search_int8.ncnn.param) | [lightfc_search_int8.ncnn.bin](./quantized/ncnn/lightfc_search_int8.ncnn.bin) |
| Score INT8 | [lightfc_score_int8.ncnn.param](./quantized/ncnn/lightfc_score_int8.ncnn.param) | [lightfc_score_int8.ncnn.bin](./quantized/ncnn/lightfc_score_int8.ncnn.bin) |
| Size INT8 | [lightfc_size_int8.ncnn.param](./quantized/ncnn/lightfc_size_int8.ncnn.param) | [lightfc_size_int8.ncnn.bin](./quantized/ncnn/lightfc_size_int8.ncnn.bin) |
| Offset INT8 | [lightfc_offset_int8.ncnn.param](./quantized/ncnn/lightfc_offset_int8.ncnn.param) | [lightfc_offset_int8.ncnn.bin](./quantized/ncnn/lightfc_offset_int8.ncnn.bin) |
| Fusion FP32 | [lightfc_fusion.ncnn.param](./quantized/ncnn/lightfc_fusion.ncnn.param) | [lightfc_fusion.ncnn.bin](./quantized/ncnn/lightfc_fusion.ncnn.bin) |

根据当前 CPU 实测，FP32 ncnn 的速度优于当前混合 INT8 ncnn，因此 C++ 实时部署优先推荐上面的 12 文件 FP32 方案。

## 数据集

ONNX 和 ncnn 的标定/验证均使用 GOT-10k train。默认路径：

```text
F:\dataset\got10k\train
```

目录下应包含 `list.txt`、序列目录、`groundtruth.txt`、`absence.label` 和 `cover.label`。可通过各脚本的 `--got10k` 参数修改路径。

## 参考

- LightFC 论文：[LightFC: A Lightweight Fully-Convolutional Transformer for Visual Tracking](https://arxiv.org/abs/2310.05392)
- 原始 LightFC 项目提供的 checkpoint 和实验结果：[Google Drive](https://drive.google.com/file/d/1ns7NQJCt078547X483skqjX1qM1rBqLP/view)
