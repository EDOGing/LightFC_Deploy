#include <jni.h>

#include <android/bitmap.h>
#include <android/log.h>
#include <cpu.h>
#include <c_api.h>

#include <cstdio>
#include <algorithm>
#include <cstdint>
#include <opencv2/imgproc.hpp>
#include <string>

#include "runtime/ncnn_runtime.h"

#define LOG_TAG "NcnnTracker"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

bool read_string(JNIEnv* env, jstring value, const char* label, std::string& output) {
    if (value == nullptr) {
        LOGE("%s is null", label);
        return false;
    }
    const char* chars = env->GetStringUTFChars(value, nullptr);
    if (chars == nullptr) {
        LOGE("GetStringUTFChars failed for %s", label);
        return false;
    }
    output.assign(chars);
    env->ReleaseStringUTFChars(value, chars);
    return true;
}

bool bitmap_to_bgr(JNIEnv* env, jobject bitmap, cv::Mat& bgr, std::string& error) {
    if (bitmap == nullptr) {
        error = "bitmap is null";
        return false;
    }
    AndroidBitmapInfo info{};
    int code = AndroidBitmap_getInfo(env, bitmap, &info);
    if (code != ANDROID_BITMAP_RESULT_SUCCESS) {
        error = "AndroidBitmap_getInfo failed (code " + std::to_string(code) + ')';
        return false;
    }
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888 || info.width == 0 || info.height == 0) {
        error = "bitmap must use non-empty RGBA_8888 format";
        return false;
    }
    void* pixels = nullptr;
    code = AndroidBitmap_lockPixels(env, bitmap, &pixels);
    if (code != ANDROID_BITMAP_RESULT_SUCCESS || pixels == nullptr) {
        error = "AndroidBitmap_lockPixels failed (code " + std::to_string(code) + ')';
        return false;
    }
    cv::Mat rgba(static_cast<int>(info.height),
                 static_cast<int>(info.width),
                 CV_8UC4,
                 pixels,
                 static_cast<std::size_t>(info.stride));
    cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
    AndroidBitmap_unlockPixels(env, bitmap);
    return !bgr.empty();
}

jobject make_track_result(JNIEnv* env,
                          const bool success,
                          const std::string& error,
                          const trackingapp::TrackOutput& output) {
    jclass result_class = env->FindClass("com/example/trackingapp/TrackerResult");
    if (result_class == nullptr) return nullptr;
    jmethodID constructor = env->GetMethodID(
        result_class,
        "<init>",
        "(ZLjava/lang/String;FFFFFF)V");
    if (constructor == nullptr) return nullptr;
    jstring error_string = env->NewStringUTF(error.c_str());
    if (error_string == nullptr) return nullptr;
    jvalue args[8]{};
    args[0].z = success ? JNI_TRUE : JNI_FALSE;
    args[1].l = error_string;
    args[2].f = static_cast<jfloat>(output.box.x);
    args[3].f = static_cast<jfloat>(output.box.y);
    args[4].f = static_cast<jfloat>(output.box.width);
    args[5].f = static_cast<jfloat>(output.box.height);
    args[6].f = output.confidence;
    args[7].f = output.inference_ms;
    jobject result = env->NewObjectA(result_class, constructor, args);
    env->DeleteLocalRef(error_string);
    env->DeleteLocalRef(result_class);
    return result;
}

}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_trackingapp_NcnnTracker_nativeConvertYuv420ToArgb(
    JNIEnv* env,
    jobject /* thiz */,
    jobject y_buffer,
    jobject u_buffer,
    jobject v_buffer,
    jint y_offset,
    jint u_offset,
    jint v_offset,
    jint y_row_stride,
    jint u_row_stride,
    jint v_row_stride,
    jint u_pixel_stride,
    jint v_pixel_stride,
    jint crop_left,
    jint crop_top,
    jint crop_width,
    jint crop_height,
    jint rotation_degrees,
    jobject output_bitmap) {
    auto* y_data = static_cast<const std::uint8_t*>(env->GetDirectBufferAddress(y_buffer));
    auto* u_data = static_cast<const std::uint8_t*>(env->GetDirectBufferAddress(u_buffer));
    auto* v_data = static_cast<const std::uint8_t*>(env->GetDirectBufferAddress(v_buffer));
    const jlong y_capacity = env->GetDirectBufferCapacity(y_buffer);
    const jlong u_capacity = env->GetDirectBufferCapacity(u_buffer);
    const jlong v_capacity = env->GetDirectBufferCapacity(v_buffer);
    if (y_data == nullptr || u_data == nullptr || v_data == nullptr ||
        y_capacity <= 0 || u_capacity <= 0 || v_capacity <= 0) {
        return env->NewStringUTF("MediaCodec returned non-direct YUV buffers");
    }
    if (crop_width <= 0 || crop_height <= 0 || y_row_stride <= 0 ||
        u_row_stride <= 0 || v_row_stride <= 0 ||
        u_pixel_stride <= 0 || v_pixel_stride <= 0) {
        return env->NewStringUTF("Invalid YUV plane layout");
    }

    AndroidBitmapInfo bitmap_info{};
    if (AndroidBitmap_getInfo(env, output_bitmap, &bitmap_info) != ANDROID_BITMAP_RESULT_SUCCESS ||
        bitmap_info.format != ANDROID_BITMAP_FORMAT_RGBA_8888 ||
        bitmap_info.width == 0 || bitmap_info.height == 0) {
        return env->NewStringUTF("YUV output bitmap must use RGBA_8888");
    }

    void* raw_pixels = nullptr;
    if (AndroidBitmap_lockPixels(env, output_bitmap, &raw_pixels) != ANDROID_BITMAP_RESULT_SUCCESS ||
        raw_pixels == nullptr) {
        return env->NewStringUTF("Unable to lock YUV output bitmap");
    }

    const int rotation = ((rotation_degrees % 360) + 360) % 360;
    const int output_width = static_cast<int>(bitmap_info.width);
    const int output_height = static_cast<int>(bitmap_info.height);
    bool valid = true;
    for (int output_y = 0; output_y < output_height && valid; ++output_y) {
        auto* row = static_cast<std::uint8_t*>(raw_pixels) +
                    static_cast<std::size_t>(output_y) * bitmap_info.stride;
        for (int output_x = 0; output_x < output_width; ++output_x) {
            int local_x = 0;
            int local_y = 0;
            if (rotation == 90) {
                local_x = output_y * crop_width / output_height;
                local_y = crop_height - 1 - output_x * crop_height / output_width;
            } else if (rotation == 180) {
                local_x = crop_width - 1 - output_x * crop_width / output_width;
                local_y = crop_height - 1 - output_y * crop_height / output_height;
            } else if (rotation == 270) {
                local_x = crop_width - 1 - output_y * crop_width / output_height;
                local_y = output_x * crop_height / output_width;
            } else {
                local_x = output_x * crop_width / output_width;
                local_y = output_y * crop_height / output_height;
            }
            const int source_x = crop_left + std::clamp(local_x, 0, crop_width - 1);
            const int source_y = crop_top + std::clamp(local_y, 0, crop_height - 1);
            const jlong y_index = static_cast<jlong>(y_offset) +
                                  static_cast<jlong>(source_y) * y_row_stride + source_x;
            const jlong u_index = static_cast<jlong>(u_offset) +
                                  static_cast<jlong>(source_y / 2) * u_row_stride +
                                  static_cast<jlong>(source_x / 2) * u_pixel_stride;
            const jlong v_index = static_cast<jlong>(v_offset) +
                                  static_cast<jlong>(source_y / 2) * v_row_stride +
                                  static_cast<jlong>(source_x / 2) * v_pixel_stride;
            if (y_index < 0 || y_index >= y_capacity ||
                u_index < 0 || u_index >= u_capacity ||
                v_index < 0 || v_index >= v_capacity) {
                valid = false;
                break;
            }

            const int y_value = std::max(0, static_cast<int>(y_data[y_index]) - 16);
            const int u_value = static_cast<int>(u_data[u_index]) - 128;
            const int v_value = static_cast<int>(v_data[v_index]) - 128;
            const int red = (298 * y_value + 409 * v_value + 128) >> 8;
            const int green = (298 * y_value - 100 * u_value - 208 * v_value + 128) >> 8;
            const int blue = (298 * y_value + 516 * u_value + 128) >> 8;
            auto* pixel = row + static_cast<std::size_t>(output_x) * 4U;
            pixel[0] = static_cast<std::uint8_t>(std::clamp(red, 0, 255));
            pixel[1] = static_cast<std::uint8_t>(std::clamp(green, 0, 255));
            pixel[2] = static_cast<std::uint8_t>(std::clamp(blue, 0, 255));
            pixel[3] = 255;
        }
    }
    AndroidBitmap_unlockPixels(env, output_bitmap);
    if (!valid) return env->NewStringUTF("YUV plane buffer is smaller than its stride metadata");
    return env->NewStringUTF("");
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_trackingapp_NcnnTracker_nativeGetVersion(JNIEnv* env, jobject /* thiz */) {
    char runtime_info[160]{};
    std::snprintf(runtime_info,
                  sizeof(runtime_info),
                  "ncnn %s | CPU-only | arm64-v8a | logical cores: %d",
                  ncnn_version(),
                  ncnn::get_cpu_count());
    LOGI("nativeGetVersion: %s", runtime_info);
    return env->NewStringUTF(runtime_info);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_trackingapp_NcnnTracker_nativeLoadModel(
    JNIEnv* env,
    jobject /* thiz */,
    jstring template_param,
    jstring template_bin,
    jstring tracking_param,
    jstring tracking_bin) {
    trackingapp::ModelPaths paths;
    if (!read_string(env, template_param, "templateParamPath", paths.template_param) ||
        !read_string(env, template_bin, "templateBinPath", paths.template_bin) ||
        !read_string(env, tracking_param, "trackingParamPath", paths.tracking_param) ||
        !read_string(env, tracking_bin, "trackingBinPath", paths.tracking_bin)) {
        return env->NewStringUTF("JNI string conversion failed");
    }

    std::string error;
    bool reused = false;
    if (!trackingapp::NcnnRuntime::instance().load(paths, error, reused)) {
        return env->NewStringUTF(error.c_str());
    }
    return env->NewStringUTF(reused ? "REUSED" : "");
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_trackingapp_NcnnTracker_nativeUnloadModel(JNIEnv* /* env */, jobject /* thiz */) {
    trackingapp::NcnnRuntime::instance().unload();
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_trackingapp_NcnnTracker_nativeIsModelLoaded(JNIEnv* /* env */, jobject /* thiz */) {
    return trackingapp::NcnnRuntime::instance().is_loaded() ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_trackingapp_NcnnTracker_nativeGetModelStatus(JNIEnv* env, jobject /* thiz */) {
    const std::string status = trackingapp::NcnnRuntime::instance().status();
    return env->NewStringUTF(status.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_trackingapp_NcnnTracker_nativeInitialize(
    JNIEnv* env,
    jobject /* thiz */,
    jobject bitmap,
    jfloat x,
    jfloat y,
    jfloat width,
    jfloat height) {
    cv::Mat bgr;
    std::string error;
    if (!bitmap_to_bgr(env, bitmap, bgr, error) ||
        !trackingapp::NcnnRuntime::instance().initialize(
            bgr,
            trackingapp::Box{x, y, width, height},
            error)) {
        LOGE("tracker initialize failed: %s", error.c_str());
        return env->NewStringUTF(error.c_str());
    }
    return env->NewStringUTF("");
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_example_trackingapp_NcnnTracker_nativeTrack(
    JNIEnv* env,
    jobject /* thiz */,
    jobject bitmap) {
    cv::Mat bgr;
    std::string error;
    trackingapp::TrackOutput output;
    if (!bitmap_to_bgr(env, bitmap, bgr, error) ||
        !trackingapp::NcnnRuntime::instance().track(bgr, output, error)) {
        LOGE("track failed: %s", error.c_str());
        return make_track_result(env, false, error, output);
    }
    return make_track_result(env, true, "", output);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_trackingapp_NcnnTracker_nativeResetTracker(JNIEnv* /* env */, jobject /* thiz */) {
    trackingapp::NcnnRuntime::instance().reset_tracker();
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_trackingapp_NcnnTracker_nativeIsTrackerInitialized(
    JNIEnv* /* env */,
    jobject /* thiz */) {
    return trackingapp::NcnnRuntime::instance().is_tracker_initialized()
               ? JNI_TRUE
               : JNI_FALSE;
}
