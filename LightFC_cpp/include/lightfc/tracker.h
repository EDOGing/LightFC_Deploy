#pragma once

#include <array>
#include <filesystem>
#include <mutex>
#include <string>

#include <net.h>
#include <opencv2/core.hpp>

#include "lightfc/preprocess.h"
#include "lightfc/tracking_output.h"
#include "lightfc/types.h"

namespace lightfc {

struct TrackerOptions {
    int cpu_threads = 4;
};

class LightFCNcnnTracker {
public:
    LightFCNcnnTracker();
    ~LightFCNcnnTracker() = default;

    LightFCNcnnTracker(const LightFCNcnnTracker&) = delete;
    LightFCNcnnTracker& operator=(const LightFCNcnnTracker&) = delete;

    bool load(const std::filesystem::path& model_dir,
              const TrackerOptions& options,
              std::string* error = nullptr);

    bool initialize(const cv::Mat& first_bgr_frame,
                    const BBox& initial_box,
                    std::string* error = nullptr);

    bool track(const cv::Mat& current_bgr_frame,
               TrackResult& result,
               std::string* error = nullptr);

    // Portable downstream interface: call after initialize()/track() succeeds.
    // Returns false when no valid result is available.
    bool get_tracking_output(TrackingOutput& output) const;

    void cancel();
    bool initialized() const;
    BBox state() const;

    static constexpr int kTemplateSize = 128;
    static constexpr int kSearchSize = 256;
    static constexpr int kFeatureSize = 16;
    static constexpr double kTemplateFactor = 2.0;
    static constexpr double kSearchFactor = 4.0;

private:
    static void configure_net(ncnn::Net& net, int threads);
    static bool load_pair(ncnn::Net& net,
                          const std::filesystem::path& model_dir,
                          const char* stem,
                          std::string& error);
    TrackResult decode_box(const ncnn::Mat& score,
                           const ncnn::Mat& size,
                           const ncnn::Mat& offset,
                           double resize_factor,
                           int image_width,
                           int image_height,
                           const BBox& previous) const;
    static bool validate_shape(const ncnn::Mat& value,
                               int width,
                               int height,
                               int channels,
                               const char* name,
                               std::string& error);

    ncnn::Net template_net_;
    ncnn::Net tracking_net_;
    ncnn::Mat template_feature_;
    std::array<float, kFeatureSize * kFeatureSize> hann_{};
    BBox state_{};
    TrackingOutput latest_output_{};
    std::uint64_t output_sequence_ = 0;
    bool has_output_ = false;
    bool loaded_ = false;
    bool initialized_ = false;
    mutable std::mutex state_mutex_;
};

}  // namespace lightfc
