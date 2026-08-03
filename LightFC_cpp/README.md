# LightFC NCNN C++ / RTSP Web Demo

这是 Windows x64 + VS2022 的 C++ 部署版本。程序读取 RTSP 或本地视频，通过浏览器框选目标并输出 MJPEG 实时跟踪画面。

## 模型结构

运行时只加载两张 NCNN 图（4 个文件）：

```text
lightfc_template.ncnn.param/bin
lightfc_tracking.ncnn.param/bin
```

- 初始化：`template[3,128,128] -> template_features[96,8,8]`。
- 每帧：合并图输入 `template_features` 与 `search[3,256,256]`，直接输出：
  - `score_map[1,16,16]`
  - `size_map[2,16,16]`
  - `offset_map[2,16,16]`
- `pred_boxes` 不在模型中，bbox 完全由 C++ 解码，因此没有 Cast 算子。

合并模型由以下命令生成：

```powershell
D:\anaconda3\python.exe tools\merge_tracking_ncnn.py
```

脚本使用项目内 `ncnnmerge.exe`，并在 fusion 后显式插入三路 `Split`。`lightfc_compare_models` 已验证合并图三张输出与原 5 图的最大绝对误差均为 0。

## 与 Python LightFC 对齐

- 输入框是上一帧 `xywh`，中心为 `(x + 0.5w, y + 0.5h)`。
- 裁剪边长为 `ceil(sqrt(w*h) * factor)`；模板 factor=2、输出 128×128，搜索区域 factor=4、输出 256×256。
- 裁剪起点使用 Python `round` 的 ties-to-even 规则。
- 越界采用每通道 0 的 `BORDER_CONSTANT`；右侧/下侧补边保留 `+1`。
- resize 显式使用 `INTER_LINEAR`。
- BGR 按 RGB 顺序转换为 CHW float，并执行 `(pixel / 255 - mean) / std`；mean=`[0.485,0.456,0.406]`，std=`[0.229,0.224,0.225]`。
- 响应为 `score * hann`；Hann 对应 `torch.hann_window(16, periodic=False)` 的二维外积。
- 行优先展平后取第一个最大值，随后用 size/offset map 解码并执行 `clip_box(..., margin=2)`。
- 置信度为未乘 Hann 的 `score_map.max()`。

实现位于 `src/preprocess.cpp` 与 `src/tracker.cpp`。

## 逐帧跟踪数据接口

`include/lightfc/tracking_output.h` 定义了面向云台（yaw/pitch）控制或其他下游模块的版本化数据：

```cpp
struct TrackingOutput {
    std::uint32_t schema_version;  // 当前为 1
    std::uint64_t sequence;        // 跟踪输出连续序号
    std::uint64_t frame_id;        // 当前视频连接中的原始帧序号
    std::int64_t timestamp_unix_ms;
    int image_width;
    int image_height;
    BBox bbox;                     // 原图像素坐标 xywh
    float confidence;              // score_map.max()，未乘 Hann window
};
```

这不是 Web 接口，也不依赖 `RtspTrackingService`。同事只需要拿到 `LightFCNcnnTracker`，在每次 `initialize()` 或 `track()` 成功后调用一个函数：

```cpp
lightfc::TrackResult result;
if (tracker.track(frame, result, &error)) {
    lightfc::TrackingOutput output;
    if (tracker.get_tracking_output(output)) {
        const auto& box = output.bbox;
        send_to_ptz_module(box.x, box.y, box.width, box.height,
                           output.confidence);
    }
}
```

唯一需要交给同事的结果函数是：

```cpp
bool LightFCNcnnTracker::get_tracking_output(TrackingOutput& output) const;
```

成功初始化后，该函数返回用户框和置信度 `1.0`；以后每次成功推理后返回最新的模型 bbox 和置信度。尚未初始化、初始化失败或调用 `cancel()` 后返回 `false`。后续需要增加目标中心、丢失标志、yaw/pitch 建议值等数据时，可在 `TrackingOutput` 尾部追加字段并提升 `schema_version`，函数签名不需要改变。

Web 页面中的逐帧日志框只是演示和检查工具，内部通过以下 HTTP 地址按 `sequence` 获取数据，它不是给移植代码调用的正式接口：

```text
GET /api/tracking-output?after=0&limit=200
```

服务端最多缓存最近 512 条，网页每 100 ms 拉取一次并按顺序打印；浏览器文本框最多保留最近 300 帧，防止长时间运行持续占用内存。

## 静态 OpenCV

项目不使用 vcpkg。OpenCV 4.14.0 已从 `F:\opencv4.14.0\opencv\sources` 构建为静态模块，安装在：

```text
third_party/opencv-static/install/x64/vc17/staticlib/
```

静态模块包括 core、imgproc、imgcodecs、videoio，以及静态 zlib、libpng、libjpeg-turbo、ittnotify。重新构建它们可运行：

```powershell
powershell -ExecutionPolicy Bypass -File tools\build_opencv_static.ps1
```

最终 exe 不依赖 `opencv_world4140.dll`。为保留 Windows RTSP/FFmpeg 支持，仍需随程序分发：

```text
opencv_videoio_ffmpeg4140_64.dll
```

现有预编译 `ncnn.lib` 使用 `/MD` 和 OpenMP，所以 exe 仍依赖 MSVC Runtime、CONCRT 与 VCOMP。若要求真正零第三方运行时 DLL，需要用 `/MT` 重新编译 NCNN，并自行准备静态 FFmpeg；仅修改 LightFC 的链接选项无法绕过二进制库的 RuntimeLibrary 标记。

## VS2022 构建与运行

```powershell
cd F:\work\LightFC-main\LightFC-main0\LightFC_cpp
& 'D:\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --preset vs2022-x64
& 'D:\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build --preset release
```

也可直接打开：

```text
build/vs2022-x64/LightFCNcnnWeb.sln
```

选择 `Release | x64`，启动项目选择 `lightfc_web`，然后按 Ctrl+F5。

发布目录：

```text
build/vs2022-x64/Release/
  lightfc_web.exe
  opencv_videoio_ffmpeg4140_64.dll
  model/
    lightfc_template.ncnn.param/bin
    lightfc_tracking.ncnn.param/bin
  web/index.html
```

命令行参数：

- `--model DIR`、`--web DIR`：覆盖模型和网页目录。
- `--url URL`：启动时连接视频源。
- `--threads N`：NCNN 线程数，当前机器建议 4。
- `--no-open`：不自动打开浏览器。
- `--run-seconds N`：运行 N 秒后自动退出，用于自动化测试。

HTTP 服务只监听 `127.0.0.1`。程序启动后访问 `http://127.0.0.1:8080/`。
