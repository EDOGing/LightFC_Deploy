#include "lightfc/tracker.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace lightfc {
namespace {

constexpr float kPi = 3.14159265358979323846f;

std::string path_utf8(const std::filesystem::path& path) {
#if defined(_WIN32)
    const auto text = path.u8string();
    return std::string(text.begin(), text.end());
#else
    return path.string();
#endif
}

std::string ncnn_error(const char* operation, const char* net_name, const int code) {
    std::ostringstream stream;
    stream << operation << " failed for " << net_name << " (ncnn code " << code << ')';
    return stream.str();
}

}  // namespace

LightFCNcnnTracker::LightFCNcnnTracker() {
    std::array<float, kFeatureSize> one_dimensional{};
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

void LightFCNcnnTracker::configure_net(ncnn::Net& net, const int threads) {
    net.opt.use_vulkan_compute = false;
    net.opt.num_threads = std::max(1, threads);
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_fp16_packed = false;
    net.opt.use_bf16_storage = false;
}

bool LightFCNcnnTracker::load_pair(ncnn::Net& net,
                                   const std::filesystem::path& model_dir,
                                   const char* stem,
                                   std::string& error) {
    const auto param_path = model_dir / (std::string(stem) + ".ncnn.param");
    const auto model_path = model_dir / (std::string(stem) + ".ncnn.bin");
    if (!std::filesystem::is_regular_file(param_path) ||
        !std::filesystem::is_regular_file(model_path)) {
        error = "missing model pair: " + path_utf8(param_path) + " / " + path_utf8(model_path);
        return false;
    }
    int code = net.load_param(path_utf8(param_path).c_str());
    if (code != 0) {
        error = ncnn_error("load_param", stem, code);
        return false;
    }
    code = net.load_model(path_utf8(model_path).c_str());
    if (code != 0) {
        error = ncnn_error("load_model", stem, code);
        return false;
    }
    return true;
}

bool LightFCNcnnTracker::load(const std::filesystem::path& model_dir,
                              const TrackerOptions& options,
                              std::string* error) {
    std::lock_guard<std::mutex> guard(state_mutex_);
    std::string detail;
    const int total_threads = options.cpu_threads > 0
                                  ? options.cpu_threads
                                  : std::max(1u, std::thread::hardware_concurrency());
    configure_net(template_net_, total_threads);
    configure_net(tracking_net_, total_threads);

    loaded_ = load_pair(template_net_, model_dir, "lightfc_template", detail) &&
              load_pair(tracking_net_, model_dir, "lightfc_tracking", detail);
    initialized_ = false;
    has_output_ = false;
    output_sequence_ = 0;
    template_feature_ = ncnn::Mat();
    if (!loaded_ && error != nullptr) {
        *error = detail;
    }
    return loaded_;
}

bool LightFCNcnnTracker::validate_shape(const ncnn::Mat& value,
                                        const int width,
                                        const int height,
                                        const int channels,
                                        const char* name,
                                        std::string& error) {
    const int logical_channels = value.c * value.elempack;
    if (value.empty() || value.w != width || value.h != height ||
        logical_channels != channels) {
        std::ostringstream stream;
        stream << name << " shape mismatch; expected CxHxW=" << channels << 'x'
               << height << 'x' << width << ", got c=" << value.c
               << " elempack=" << value.elempack << " h=" << value.h << " w=" << value.w;
        error = stream.str();
        return false;
    }
    return true;
}

bool LightFCNcnnTracker::initialize(const cv::Mat& first_bgr_frame,
                                    const BBox& initial_box,
                                    std::string* error) {
    std::lock_guard<std::mutex> guard(state_mutex_);
    try {
        if (!loaded_) {
            throw std::runtime_error("models are not loaded");
        }
        const CropResult crop =
            sample_target(first_bgr_frame, initial_box, kTemplateFactor, kTemplateSize);
        const ncnn::Mat input = preprocess_bgr_to_rgb_chw(crop.patch_bgr);
        ncnn::Extractor extractor = template_net_.create_extractor();
        extractor.set_light_mode(true);
        int code = extractor.input("in0", input);
        if (code != 0) {
            throw std::runtime_error(ncnn_error("input", "lightfc_template", code));
        }
        code = extractor.extract("out0", template_feature_);
        if (code != 0) {
            throw std::runtime_error(ncnn_error("extract", "lightfc_template", code));
        }
        std::string shape_error;
        if (!validate_shape(template_feature_, 8, 8, 96, "template_feature", shape_error)) {
            throw std::runtime_error(shape_error);
        }
        state_ = initial_box;  // Python initialize stores the user box without clip_box.
        latest_output_ = TrackingOutput{};
        latest_output_.sequence = ++output_sequence_;
        latest_output_.frame_id = latest_output_.sequence;
        latest_output_.timestamp_unix_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch())
                .count();
        latest_output_.image_width = first_bgr_frame.cols;
        latest_output_.image_height = first_bgr_frame.rows;
        latest_output_.bbox = initial_box;
        latest_output_.confidence = 1.0f;
        has_output_ = true;
        initialized_ = true;
        return true;
    } catch (const std::exception& exception) {
        initialized_ = false;
        has_output_ = false;
        template_feature_ = ncnn::Mat();
        if (error != nullptr) {
            *error = exception.what();
        }
        return false;
    }
}

TrackResult LightFCNcnnTracker::decode_box(const ncnn::Mat& score,
                                           const ncnn::Mat& size,
                                           const ncnn::Mat& offset,
                                           const double resize_factor,
                                           const int image_width,
                                           const int image_height,
                                           const BBox& previous) const {
    const float* score_values = score.channel(0);
    float best_response = -std::numeric_limits<float>::infinity();
    float raw_confidence = -std::numeric_limits<float>::infinity();
    int best_index = 0;
    for (int index = 0; index < kFeatureSize * kFeatureSize; ++index) {
        raw_confidence = std::max(raw_confidence, score_values[index]);
        const float response = score_values[index] * hann_[index];
        // Strict > preserves NumPy argmax's first-index behavior for ties.
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

    const double normalized_cx =
        (best_x + static_cast<double>(dx_values[best_index])) / kFeatureSize;
    const double normalized_cy =
        (best_y + static_cast<double>(dy_values[best_index])) / kFeatureSize;
    const double normalized_width = width_values[best_index];
    const double normalized_height = height_values[best_index];
    const double scale = static_cast<double>(kSearchSize) / resize_factor;
    const double cx = normalized_cx * scale;
    const double cy = normalized_cy * scale;
    const double width = normalized_width * scale;
    const double height = normalized_height * scale;
    const double previous_cx = previous.x + 0.5 * previous.width;
    const double previous_cy = previous.y + 0.5 * previous.height;
    const double half_side = 0.5 * scale;

    const BBox mapped{cx + previous_cx - half_side - 0.5 * width,
                      cy + previous_cy - half_side - 0.5 * height,
                      width,
                      height};
    return TrackResult{clip_box(mapped, image_height, image_width, 2.0),
                       raw_confidence,
                       best_x,
                       best_y};
}

bool LightFCNcnnTracker::track(const cv::Mat& current_bgr_frame,
                               TrackResult& result,
                               std::string* error) {
    std::lock_guard<std::mutex> guard(state_mutex_);
    try {
        if (!loaded_ || !initialized_) {
            throw std::runtime_error("tracker is not initialized");
        }
        const CropResult crop =
            sample_target(current_bgr_frame, state_, kSearchFactor, kSearchSize);
        const ncnn::Mat search_input = preprocess_bgr_to_rgb_chw(crop.patch_bgr);

        ncnn::Mat score_map;
        ncnn::Mat size_map;
        ncnn::Mat offset_map;
        ncnn::Extractor extractor = tracking_net_.create_extractor();
        extractor.set_light_mode(true);
        int code = extractor.input("template_features", template_feature_);
        if (code != 0) {
            throw std::runtime_error(ncnn_error("input template_features", "lightfc_tracking", code));
        }
        code = extractor.input("search", search_input);
        if (code != 0) {
            throw std::runtime_error(ncnn_error("input search", "lightfc_tracking", code));
        }
        code = extractor.extract("score_map", score_map);
        if (code != 0) {
            throw std::runtime_error(ncnn_error("extract score_map", "lightfc_tracking", code));
        }
        code = extractor.extract("size_map", size_map);
        if (code != 0) {
            throw std::runtime_error(ncnn_error("extract size_map", "lightfc_tracking", code));
        }
        code = extractor.extract("offset_map", offset_map);
        if (code != 0) {
            throw std::runtime_error(ncnn_error("extract offset_map", "lightfc_tracking", code));
        }
        std::string shape_error;
        if (!validate_shape(score_map, 16, 16, 1, "score_map", shape_error) ||
            !validate_shape(size_map, 16, 16, 2, "size_map", shape_error) ||
            !validate_shape(offset_map, 16, 16, 2, "offset_map", shape_error)) {
            throw std::runtime_error(shape_error);
        }
        if (score_map.elempack != 1 || size_map.elempack != 1 || offset_map.elempack != 1) {
            throw std::runtime_error("map outputs must use elempack=1 for C++ decoder");
        }

        result = decode_box(score_map, size_map, offset_map, crop.resize_factor,
                            current_bgr_frame.cols, current_bgr_frame.rows, state_);
        state_ = result.box;
        latest_output_ = TrackingOutput{};
        latest_output_.sequence = ++output_sequence_;
        latest_output_.frame_id = latest_output_.sequence;
        latest_output_.timestamp_unix_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch())
                .count();
        latest_output_.image_width = current_bgr_frame.cols;
        latest_output_.image_height = current_bgr_frame.rows;
        latest_output_.bbox = result.box;
        latest_output_.confidence = result.confidence;
        has_output_ = true;
        return true;
    } catch (const std::exception& exception) {
        if (error != nullptr) {
            *error = exception.what();
        }
        return false;
    }
}

bool LightFCNcnnTracker::get_tracking_output(TrackingOutput& output) const {
    std::lock_guard<std::mutex> guard(state_mutex_);
    if (!has_output_) {
        return false;
    }
    output = latest_output_;
    return true;
}

void LightFCNcnnTracker::cancel() {
    std::lock_guard<std::mutex> guard(state_mutex_);
    initialized_ = false;
    has_output_ = false;
    template_feature_ = ncnn::Mat();
}

bool LightFCNcnnTracker::initialized() const {
    std::lock_guard<std::mutex> guard(state_mutex_);
    return initialized_;
}

BBox LightFCNcnnTracker::state() const {
    std::lock_guard<std::mutex> guard(state_mutex_);
    return state_;
}

}  // namespace lightfc
