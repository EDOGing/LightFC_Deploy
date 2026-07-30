#include "runtime/ncnn_runtime.h"

#include <android/log.h>
#include <cpu.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <sys/stat.h>
#include <utility>
#include <vector>

#define LOG_TAG "NcnnTracker"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace trackingapp {
namespace {

constexpr int kTemplateSize = 128;
constexpr int kSearchSize = 256;
constexpr int kFeatureSize = 16;
constexpr double kTemplateFactor = 2.0;
constexpr double kSearchFactor = 4.0;
constexpr float kPi = 3.14159265358979323846f;

bool contains_name(const std::vector<const char*>& names, const char* expected) {
    for (const char* name : names) {
        if (name != nullptr && std::strcmp(name, expected) == 0) return true;
    }
    return false;
}

bool require_names(const std::vector<const char*>& actual,
                   const std::vector<const char*>& expected,
                   const char* label,
                   std::string& error) {
    for (const char* name : expected) {
        if (!contains_name(actual, name)) {
            std::ostringstream stream;
            stream << label << " missing required node: " << name;
            error = stream.str();
            return false;
        }
    }
    return true;
}

bool python_round_to_int(const double value, int& result) {
    if (!std::isfinite(value)) return false;
    const double lower = std::floor(value);
    const double fraction = value - lower;
    double rounded = lower;
    if (fraction > 0.5) {
        rounded = lower + 1.0;
    } else if (fraction == 0.5) {
        const auto lower_integer = static_cast<long long>(lower);
        rounded = (lower_integer % 2LL == 0LL) ? lower : lower + 1.0;
    }
    if (rounded < static_cast<double>(std::numeric_limits<int>::min()) ||
        rounded > static_cast<double>(std::numeric_limits<int>::max())) {
        return false;
    }
    result = static_cast<int>(rounded);
    return true;
}

Box clip_box(const Box& box, const int image_height, const int image_width) {
    constexpr double margin = 2.0;
    double x1 = std::min(std::max(0.0, box.x), image_width - margin);
    double x2 = std::min(std::max(margin, box.x + box.width),
                         static_cast<double>(image_width));
    double y1 = std::min(std::max(0.0, box.y), image_height - margin);
    double y2 = std::min(std::max(margin, box.y + box.height),
                         static_cast<double>(image_height));
    return Box{x1, y1, std::max(margin, x2 - x1), std::max(margin, y2 - y1)};
}

}  // namespace

bool ModelPaths::operator==(const ModelPaths& other) const {
    return template_param == other.template_param && template_bin == other.template_bin &&
           tracking_param == other.tracking_param && tracking_bin == other.tracking_bin;
}

NcnnRuntime::NcnnRuntime() {
    float one_dimensional[kFeatureSize]{};
    for (int i = 0; i < kFeatureSize; ++i) {
        const float angle =
            (2.0f * kPi / static_cast<float>(kFeatureSize + 1)) * static_cast<float>(i + 1);
        one_dimensional[i] = 0.5f * (1.0f - std::cos(angle));
    }
    for (int y = 0; y < kFeatureSize; ++y) {
        for (int x = 0; x < kFeatureSize; ++x) {
            hann_[y * kFeatureSize + x] = one_dimensional[y] * one_dimensional[x];
        }
    }
}

NcnnRuntime& NcnnRuntime::instance() {
    static NcnnRuntime runtime;
    return runtime;
}

void NcnnRuntime::configure(ncnn::Net& net) {
    net.opt.use_vulkan_compute = false;
    net.opt.num_threads = std::max(1, std::min(4, ncnn::get_cpu_count()));
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_fp16_packed = false;
    net.opt.use_bf16_storage = false;
}

bool NcnnRuntime::validate_file(const std::string& path,
                                const char* label,
                                std::string& error) {
    if (path.empty()) {
        error = std::string(label) + " path is empty";
        return false;
    }
    struct stat info {};
    if (stat(path.c_str(), &info) != 0) {
        error = std::string(label) + " is not readable: " + std::strerror(errno);
        return false;
    }
    if (!S_ISREG(info.st_mode) || info.st_size <= 0) {
        error = std::string(label) + " is not a non-empty regular file";
        return false;
    }
    LOGI("%s path=%s size=%lld", label, path.c_str(), static_cast<long long>(info.st_size));
    return true;
}

bool NcnnRuntime::load_pair(ncnn::Net& net,
                            const std::string& param_path,
                            const std::string& bin_path,
                            const char* label,
                            std::string& error) {
    int code = net.load_param(param_path.c_str());
    LOGI("%s load_param returned %d", label, code);
    if (code != 0) {
        std::ostringstream stream;
        stream << label << " load_param failed (ncnn code " << code << ')';
        error = stream.str();
        return false;
    }
    code = net.load_model(bin_path.c_str());
    LOGI("%s load_model returned %d", label, code);
    if (code != 0) {
        std::ostringstream stream;
        stream << label << " load_model failed (ncnn code " << code << ')';
        error = stream.str();
        return false;
    }
    return true;
}

bool NcnnRuntime::validate_protocol(const ncnn::Net& template_net,
                                    const ncnn::Net& tracking_net,
                                    std::string& error) {
    return require_names(template_net.input_names(), {"in0"}, "template inputs", error) &&
           require_names(template_net.output_names(), {"out0"}, "template outputs", error) &&
           require_names(tracking_net.input_names(),
                         {"template_features", "search"},
                         "tracking inputs",
                         error) &&
           require_names(tracking_net.output_names(),
                         {"score_map", "size_map", "offset_map"},
                         "tracking outputs",
                         error);
}

bool NcnnRuntime::load(const ModelPaths& paths, std::string& error, bool& reused) {
    std::lock_guard<std::mutex> guard(mutex_);
    reused = false;
    if (template_net_ && tracking_net_ && paths_ == paths) {
        reused = true;
        LOGI("model load reused existing nets");
        return true;
    }

    if (!validate_file(paths.template_param, "template param", error) ||
        !validate_file(paths.template_bin, "template bin", error) ||
        !validate_file(paths.tracking_param, "tracking param", error) ||
        !validate_file(paths.tracking_bin, "tracking bin", error)) {
        LOGE("model file validation failed: %s", error.c_str());
        return false;
    }

    std::unique_ptr<ncnn::Net> candidate_template(new (std::nothrow) ncnn::Net());
    std::unique_ptr<ncnn::Net> candidate_tracking(new (std::nothrow) ncnn::Net());
    if (!candidate_template || !candidate_tracking) {
        error = "failed to allocate ncnn networks";
        LOGE("%s", error.c_str());
        return false;
    }
    configure(*candidate_template);
    configure(*candidate_tracking);

    if (!load_pair(*candidate_template,
                   paths.template_param,
                   paths.template_bin,
                   "lightfc_template",
                   error) ||
        !load_pair(*candidate_tracking,
                   paths.tracking_param,
                   paths.tracking_bin,
                   "lightfc_tracking",
                   error) ||
        !validate_protocol(*candidate_template, *candidate_tracking, error)) {
        LOGE("model load failed: %s", error.c_str());
        return false;
    }

    template_net_ = std::move(candidate_template);
    tracking_net_ = std::move(candidate_tracking);
    paths_ = paths;
    template_feature_ = ncnn::Mat();
    tracker_initialized_ = false;
    LOGI("LightFC template/tracking model package loaded successfully");
    return true;
}

void NcnnRuntime::unload() {
    std::lock_guard<std::mutex> guard(mutex_);
    tracking_net_.reset();
    template_net_.reset();
    paths_ = ModelPaths{};
    template_feature_ = ncnn::Mat();
    tracker_initialized_ = false;
    LOGI("model package unloaded");
}

bool NcnnRuntime::is_loaded() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return template_net_ != nullptr && tracking_net_ != nullptr;
}

std::string NcnnRuntime::status() const {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!template_net_ || !tracking_net_) return "unloaded";
    return "loaded: LightFC template + tracking (CPU)";
}

bool NcnnRuntime::sample_target(const cv::Mat& frame,
                                const Box& box,
                                const double factor,
                                const int output_size,
                                cv::Mat& patch,
                                double& resize_factor,
                                std::string& error) {
    if (frame.empty() || frame.type() != CV_8UC3 ||
        !std::isfinite(box.x) || !std::isfinite(box.y) ||
        !std::isfinite(box.width) || !std::isfinite(box.height) ||
        box.width <= 0.0 || box.height <= 0.0) {
        error = "invalid BGR frame or target box";
        return false;
    }
    const int crop_size = static_cast<int>(std::ceil(std::sqrt(box.width * box.height) * factor));
    if (crop_size < 1) {
        error = "target box is too small";
        return false;
    }
    int x1 = 0;
    int y1 = 0;
    if (!python_round_to_int(box.x + 0.5 * box.width - 0.5 * crop_size, x1) ||
        !python_round_to_int(box.y + 0.5 * box.height - 0.5 * crop_size, y1)) {
        error = "crop coordinate is out of range";
        return false;
    }
    const int x2 = x1 + crop_size;
    const int y2 = y1 + crop_size;
    const int left = std::max(0, -x1);
    const int top = std::max(0, -y1);
    const int right = std::max(x2 - frame.cols + 1, 0);
    const int bottom = std::max(y2 - frame.rows + 1, 0);
    const int source_x1 = x1 + left;
    const int source_y1 = y1 + top;
    const int source_x2 = x2 - right;
    const int source_y2 = y2 - bottom;
    if (source_x1 < 0 || source_y1 < 0 || source_x2 > frame.cols ||
        source_y2 > frame.rows || source_x2 <= source_x1 || source_y2 <= source_y1) {
        error = "crop lies outside the source frame";
        return false;
    }

    const cv::Rect roi(source_x1, source_y1, source_x2 - source_x1, source_y2 - source_y1);
    cv::Mat padded;
    cv::copyMakeBorder(frame(roi), padded, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    if (padded.cols != crop_size || padded.rows != crop_size) {
        error = "crop padding size mismatch";
        return false;
    }
    cv::resize(padded, patch, cv::Size(output_size, output_size), 0.0, 0.0, cv::INTER_LINEAR);
    resize_factor = static_cast<double>(output_size) / crop_size;
    return !patch.empty();
}

bool NcnnRuntime::preprocess(const cv::Mat& patch_bgr,
                             ncnn::Mat& output,
                             std::string& error) {
    if (patch_bgr.empty() || patch_bgr.type() != CV_8UC3) {
        error = "preprocess expects CV_8UC3";
        return false;
    }
    static constexpr float mean[3] = {0.485f, 0.456f, 0.406f};
    static constexpr float stddev[3] = {0.229f, 0.224f, 0.225f};
    output = ncnn::Mat(patch_bgr.cols, patch_bgr.rows, 3);
    if (output.empty()) {
        error = "failed to allocate ncnn input";
        return false;
    }
    float* out_r = output.channel(0);
    float* out_g = output.channel(1);
    float* out_b = output.channel(2);
    for (int y = 0; y < patch_bgr.rows; ++y) {
        const cv::Vec3b* row = patch_bgr.ptr<cv::Vec3b>(y);
        for (int x = 0; x < patch_bgr.cols; ++x) {
            const int index = y * patch_bgr.cols + x;
            out_r[index] = (static_cast<float>(row[x][2]) / 255.0f - mean[0]) / stddev[0];
            out_g[index] = (static_cast<float>(row[x][1]) / 255.0f - mean[1]) / stddev[1];
            out_b[index] = (static_cast<float>(row[x][0]) / 255.0f - mean[2]) / stddev[2];
        }
    }
    return true;
}

bool NcnnRuntime::validate_shape(const ncnn::Mat& value,
                                 const int width,
                                 const int height,
                                 const int channels,
                                 const char* name,
                                 std::string& error) {
    if (value.empty() || value.w != width || value.h != height ||
        value.c * value.elempack != channels || value.elempack != 1) {
        std::ostringstream stream;
        stream << name << " shape mismatch; expected " << channels << 'x' << height << 'x'
               << width << ", got c=" << value.c << " pack=" << value.elempack
               << " h=" << value.h << " w=" << value.w;
        error = stream.str();
        return false;
    }
    return true;
}

bool NcnnRuntime::initialize(const cv::Mat& bgr_frame,
                             const Box& initial_box,
                             std::string& error) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!template_net_ || !tracking_net_) {
        error = "models are not loaded";
        return false;
    }
    cv::Mat patch;
    double resize_factor = 0.0;
    if (!sample_target(bgr_frame, initial_box, kTemplateFactor, kTemplateSize,
                       patch, resize_factor, error)) {
        return false;
    }
    ncnn::Mat input;
    if (!preprocess(patch, input, error)) return false;
    ncnn::Mat candidate_feature;
    ncnn::Extractor extractor = template_net_->create_extractor();
    extractor.set_light_mode(true);
    int code = extractor.input("in0", input);
    if (code == 0) code = extractor.extract("out0", candidate_feature);
    if (code != 0) {
        error = "template inference failed (ncnn code " + std::to_string(code) + ')';
        return false;
    }
    if (!validate_shape(candidate_feature, 8, 8, 96, "template_feature", error)) return false;
    template_feature_ = candidate_feature;
    state_ = initial_box;
    tracker_initialized_ = true;
    LOGI("tracker initialized box=[%.2f %.2f %.2f %.2f]",
         state_.x, state_.y, state_.width, state_.height);
    return true;
}

TrackOutput NcnnRuntime::decode(const ncnn::Mat& score,
                                const ncnn::Mat& size,
                                const ncnn::Mat& offset,
                                const double resize_factor,
                                const int image_width,
                                const int image_height,
                                const float inference_ms) const {
    const float* score_values = score.channel(0);
    float best_response = -std::numeric_limits<float>::infinity();
    float raw_confidence = -std::numeric_limits<float>::infinity();
    int best_index = 0;
    for (int index = 0; index < kFeatureSize * kFeatureSize; ++index) {
        raw_confidence = std::max(raw_confidence, score_values[index]);
        const float response = score_values[index] * hann_[index];
        if (response > best_response) {
            best_response = response;
            best_index = index;
        }
    }
    const int best_y = best_index / kFeatureSize;
    const int best_x = best_index % kFeatureSize;
    const float* width_values = size.channel(0);
    const float* height_values = size.channel(1);
    const float* dx_values = offset.channel(0);
    const float* dy_values = offset.channel(1);
    const double normalized_cx = (best_x + static_cast<double>(dx_values[best_index])) /
                                 kFeatureSize;
    const double normalized_cy = (best_y + static_cast<double>(dy_values[best_index])) /
                                 kFeatureSize;
    const double scale = static_cast<double>(kSearchSize) / resize_factor;
    const double width = width_values[best_index] * scale;
    const double height = height_values[best_index] * scale;
    const double previous_cx = state_.x + 0.5 * state_.width;
    const double previous_cy = state_.y + 0.5 * state_.height;
    const double half_side = 0.5 * scale;
    const Box mapped{normalized_cx * scale + previous_cx - half_side - 0.5 * width,
                     normalized_cy * scale + previous_cy - half_side - 0.5 * height,
                     width,
                     height};
    return TrackOutput{clip_box(mapped, image_height, image_width),
                       raw_confidence,
                       inference_ms};
}

bool NcnnRuntime::track(const cv::Mat& bgr_frame,
                        TrackOutput& output,
                        std::string& error) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!template_net_ || !tracking_net_ || !tracker_initialized_) {
        error = "tracker is not initialized";
        return false;
    }
    cv::Mat patch;
    double resize_factor = 0.0;
    if (!sample_target(bgr_frame, state_, kSearchFactor, kSearchSize,
                       patch, resize_factor, error)) {
        return false;
    }
    ncnn::Mat search_input;
    if (!preprocess(patch, search_input, error)) return false;

    ncnn::Mat score_map;
    ncnn::Mat size_map;
    ncnn::Mat offset_map;
    const auto start = std::chrono::steady_clock::now();
    ncnn::Extractor extractor = tracking_net_->create_extractor();
    extractor.set_light_mode(true);
    int code = extractor.input("template_features", template_feature_);
    if (code == 0) code = extractor.input("search", search_input);
    if (code == 0) code = extractor.extract("score_map", score_map);
    if (code == 0) code = extractor.extract("size_map", size_map);
    if (code == 0) code = extractor.extract("offset_map", offset_map);
    const auto end = std::chrono::steady_clock::now();
    if (code != 0) {
        error = "tracking inference failed (ncnn code " + std::to_string(code) + ')';
        return false;
    }
    if (!validate_shape(score_map, 16, 16, 1, "score_map", error) ||
        !validate_shape(size_map, 16, 16, 2, "size_map", error) ||
        !validate_shape(offset_map, 16, 16, 2, "offset_map", error)) {
        return false;
    }
    const float inference_ms =
        std::chrono::duration<float, std::milli>(end - start).count();
    output = decode(score_map, size_map, offset_map, resize_factor,
                    bgr_frame.cols, bgr_frame.rows, inference_ms);
    state_ = output.box;
    LOGI("track box=[%.2f %.2f %.2f %.2f] confidence=%.6f inference_ms=%.3f",
         state_.x, state_.y, state_.width, state_.height,
         output.confidence, output.inference_ms);
    return true;
}

void NcnnRuntime::reset_tracker() {
    std::lock_guard<std::mutex> guard(mutex_);
    template_feature_ = ncnn::Mat();
    tracker_initialized_ = false;
    LOGI("tracker state reset");
}

bool NcnnRuntime::is_tracker_initialized() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return tracker_initialized_;
}

}  // namespace trackingapp
