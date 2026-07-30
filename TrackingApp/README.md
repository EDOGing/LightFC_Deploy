# LightFC TrackingApp

这是一个在 Android 设备上运行的轻量级目标跟踪应用。项目使用 **ncnn** 部署 LightFC 模型，通过 Kotlin、JNI 和 C++ 完成图像预处理、模型推理及目标框更新，可在没有网络连接的情况下运行。

## 技术方案

- 推理框架：ncnn（ARM64 CPU 推理）
- 跟踪模型：LightFC
- 模型结构：模板网络和跟踪网络共同工作
  - `lightfc_template.ncnn.param/.bin`
  - `lightfc_tracking.ncnn.param/.bin`
- Android：Kotlin + Camera2 + MediaCodec
- Native：C++17 + JNI
- 图像处理：OpenCV Android SDK
- 支持 ABI：`arm64-v8a`
- 最低 Android 版本：Android 7.0（API 24）

模板网络在用户选择初始目标后提取一次模板特征；跟踪网络在后续图像帧中持续预测目标位置、尺寸和置信度。

## 主要功能

- 静态图片跟踪
  - 选择模板图片并拖动画出初始目标框
  - 选择搜索图片执行单帧跟踪

- 本地视频跟踪
  - 从手机中选择视频
  - 在第一帧上绘制初始目标框
  - 使用 MediaExtractor 和 MediaCodec 连续解码
  - 处理速度落后时自动跳过过期帧
  - 支持暂停、继续和停止

- 摄像头实时跟踪
  - Camera2 实时取流
  - 支持前置和后置摄像头切换
  - 冻结当前画面并绘制初始目标框
  - 只保留最新相机帧，避免处理队列积压

- 性能信息显示
  - 跟踪置信度
  - 单帧推理耗时
  - 推理 FPS
  - 解码/图像转换与推理的完整处理 FPS

- 模型管理
  - APK 内置完整 LightFC 双网络模型
  - 首次启动时将模型复制到应用私有目录
  - 支持导入、校验、切换和删除兼容的四文件模型包

## APK 打包

模型文件、Java/Kotlin 代码和 ARM64 原生库均打包在同一个 APK 中，安装时不需要额外复制模型或动态库。应用卸载后，其私有目录中的模型副本也会由 Android 一并删除。

## 构建环境

- Android Studio Quail 2
- Android SDK 36
- Android NDK 30
- CMake 4.1.2
- OpenCV Android SDK 4.14.0

构建前请检查 `app/src/main/cpp/CMakeLists.txt` 中的 `OpenCV_DIR`，将其修改为本机 OpenCV Android SDK 的实际路径。ncnn ARM64 静态库位于 `app/src/main/cpp/ncnn/arm64-v8a`。

```powershell
.\gradlew.bat :app:assembleDebug
```

生成的调试 APK 位于：

```text
app/build/outputs/apk/debug/app-debug.apk
```

如果需要对外发布，建议创建正式签名的 Release APK，并在 GitHub Releases 中单独上传 APK 文件。

## 使用流程

1. 启动应用并等待内置 LightFC 模型加载完成。
2. 选择静态图片、本地视频或手机摄像头模式。
3. 在模板帧或冻结画面上拖动绘制目标框。
4. 初始化目标模板并开始跟踪。
5. 查看目标框、置信度、推理耗时和 FPS。

## 说明

本项目当前以 ARM64 Android 手机的 CPU 推理为主。模型文件及相关第三方库的使用和再发布应遵守其各自的许可证。
