# LightFC Deploy

本仓库用于集中存放 LightFC 的相关部署项目。每个项目均位于独立的子文件夹中。

---

## TrackingApp

一款在 Android 设备上离线运行的轻量级目标跟踪应用，使用 ncnn 部署 LightFC 模型。

主要功能：

- 支持静态图片目标跟踪
- 支持本地视频目标跟踪
- 支持手机摄像头实时目标跟踪
- 显示跟踪置信度、推理耗时和 FPS
- 支持 LightFC 模型的导入、校验与切换

apk文件在“./TrackingApp/TrackingApp.apk”

### 演示视频

[点击查看 TrackingApp 演示视频](./TrackingApp/sample_video.mp4)

[查看项目详细说明](./TrackingApp/README.md)

---

## LightFC_cpp

一款面向 Windows x64 的 LightFC C++ 实时目标跟踪 Web 应用，使用 ncnn 进行模型推理。

主要功能：

- 支持 RTSP 实时视频流和本地视频
- 支持在浏览器中框选并初始化跟踪目标
- 通过网页显示 MJPEG 实时跟踪画面
- 输出目标框、置信度及逐帧跟踪数据
- 支持下游云台控制等模块读取跟踪结果

[查看项目详细说明](./LightFC_cpp/README.md)

---

## LightFC_web

一套面向 CPU 的 LightFC 单目标跟踪 Web Demo、模型转换与量化部署项目。

主要功能：

- 支持本地视频和 RTSP 实时视频流目标跟踪
- 支持在网页中框选目标、实时预览及导出跟踪结果
- 支持 PyTorch、ONNX FP32 和 ONNX INT8 推理后端
- 支持 PTH 模型转换为 ONNX 和 ncnn
- 提供 ONNX W8A8 量化及 ncnn FP32/INT8 量化与性能对比

[查看项目详细说明](./LightFC_web/README.md)

---

后续项目将在此处分别介绍。
