#pragma once

#include <memory>
#include <mutex>
#include <string>

#include <net.h>
#include <opencv2/core.hpp>

namespace trackingapp {

struct ModelPaths {
    std::string template_param;
    std::string template_bin;
    std::string tracking_param;
    std::string tracking_bin;

    bool operator==(const ModelPaths& other) const;
};

struct Box {
    double x = 0.0;
    double y = 0.0;
    double width = 0.0;
    double height = 0.0;
};

struct TrackOutput {
    Box box;
    float confidence = 0.0f;
    float inference_ms = 0.0f;
};

class NcnnRuntime {
public:
    static NcnnRuntime& instance();

    NcnnRuntime(const NcnnRuntime&) = delete;
    NcnnRuntime& operator=(const NcnnRuntime&) = delete;

    bool load(const ModelPaths& paths, std::string& error, bool& reused);
    void unload();
    bool is_loaded() const;
    std::string status() const;
    bool initialize(const cv::Mat& bgr_frame, const Box& initial_box, std::string& error);
    bool track(const cv::Mat& bgr_frame, TrackOutput& output, std::string& error);
    void reset_tracker();
    bool is_tracker_initialized() const;

private:
    NcnnRuntime();

    static void configure(ncnn::Net& net);
    static bool validate_file(const std::string& path, const char* label, std::string& error);
    static bool load_pair(ncnn::Net& net,
                          const std::string& param_path,
                          const std::string& bin_path,
                          const char* label,
                          std::string& error);
    static bool validate_protocol(const ncnn::Net& template_net,
                                  const ncnn::Net& tracking_net,
                                  std::string& error);
    static bool sample_target(const cv::Mat& frame,
                              const Box& box,
                              double factor,
                              int output_size,
                              cv::Mat& patch,
                              double& resize_factor,
                              std::string& error);
    static bool preprocess(const cv::Mat& patch_bgr, ncnn::Mat& output, std::string& error);
    static bool validate_shape(const ncnn::Mat& value,
                               int width,
                               int height,
                               int channels,
                               const char* name,
                               std::string& error);
    TrackOutput decode(const ncnn::Mat& score,
                       const ncnn::Mat& size,
                       const ncnn::Mat& offset,
                       double resize_factor,
                       int image_width,
                       int image_height,
                       float inference_ms) const;

    mutable std::mutex mutex_;
    std::unique_ptr<ncnn::Net> template_net_;
    std::unique_ptr<ncnn::Net> tracking_net_;
    ModelPaths paths_;
    ncnn::Mat template_feature_;
    Box state_;
    float hann_[16 * 16]{};
    bool tracker_initialized_ = false;
};

}  // namespace trackingapp
