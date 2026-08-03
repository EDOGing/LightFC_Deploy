# LightFC C++ 完成状态（2026-07-30）

## 已完成

- 已将 search、fusion、score、size、offset 合并为：
  - `model/lightfc_tracking.ncnn.param`
  - `model/lightfc_tracking.ncnn.bin`
- 合并图输入：`template_features[96,8,8]`、`search[3,256,256]`。
- 合并图输出：`score_map[1,16,16]`、`size_map[2,16,16]`、`offset_map[2,16,16]`。
- fusion 输出后已显式插入三路 Split，解决 NCNN 多输出分支的整数除零问题。
- C++ 正式 tracker 现在只加载 template 和 tracking 两张图；每帧只创建一个 tracking Extractor。
- CTest `lightfc_tracking_model_equivalence` 验证三张 map 与原 5 图的 `max_abs_error=0`。
- OpenCV 4.14.0 的 core/imgproc/imgcodecs/videoio 已构建为静态 Release 库，使用 OpenMP。
- OpenCV 静态构建安装目录：`third_party/opencv-static/install`。
- 最终 exe 不依赖 `opencv_world4140.dll`，发布模型目录仅包含 4 个模型文件。
- FFmpeg wrapper `opencv_videoio_ffmpeg4140_64.dll` 被保留，用于 Windows RTSP。

## 最终验证

- GOT-10k `GOT-10k_Train_000001` 端到端跟踪：通过。
- HTTP 首页：200。
- 合并图 + 静态 OpenCV 中位推理速度：约 182 FPS（当前机器、4 个 NCNN 线程）。
- 最终 Release 程序：`build/vs2022-x64/Release/lightfc_web.exe`。

## 仍为动态运行时的部分

现有预编译 `ncnn.lib` 使用 `/MD` 和 OpenMP，因此 exe 仍依赖 MSVCP/VCRUNTIME/VCOMP；Windows FFmpeg wrapper 也是 DLL。若以后要求真正零第三方 DLL，需要从源码以匹配配置重新编译 NCNN，并准备静态 FFmpeg。

