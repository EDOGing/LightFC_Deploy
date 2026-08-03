#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>

#include "lightfc/tracker.h"
#include "lightfc/tracking_output.h"

namespace lightfc {

struct ServiceStatus {
    bool connected = false;
    bool tracking = false;
    bool has_frame = false;
    std::string url;
    std::string message = "未连接";
    double inference_fps = 0.0;
    double confidence = 0.0;
    BBox box{};
};

class RtspTrackingService {
public:
    explicit RtspTrackingService(LightFCNcnnTracker& tracker);
    ~RtspTrackingService();

    bool connect(const std::string& url, std::string* error = nullptr);
    void disconnect();
    bool request_tracking(const BBox& box, std::string* error = nullptr);
    void cancel_tracking();

    // Used by the Web demo to drain its bounded display queue. Portable users
    // should call LightFCNcnnTracker::get_tracking_output() instead.
    std::vector<TrackingOutput> tracking_outputs_after(std::uint64_t sequence,
                                                       std::size_t limit = 200) const;

    ServiceStatus status() const;
    bool snapshot(std::vector<unsigned char>& jpeg) const;
    bool wait_for_jpeg(std::uint64_t& sequence,
                       std::vector<unsigned char>& jpeg,
                       std::chrono::milliseconds timeout) const;

private:
    void worker_loop(std::string url);
    void publish_frame(const cv::Mat& frame);
    void publish_tracking_output(const BBox& box,
                                 float confidence,
                                 std::uint64_t frame_id,
                                 int image_width,
                                 int image_height);
    void set_message(const std::string& message);

    LightFCNcnnTracker& tracker_;
    mutable std::mutex mutex_;
    mutable std::condition_variable frame_ready_;
    std::thread worker_;
    std::atomic<bool> stop_{false};
    ServiceStatus status_;
    std::vector<unsigned char> latest_jpeg_;
    std::uint64_t frame_sequence_ = 0;
    bool pending_initialization_ = false;
    BBox pending_box_{};
    std::deque<double> inference_ms_;
    std::deque<TrackingOutput> tracking_outputs_;
    std::uint64_t tracking_output_sequence_ = 0;
};

}  // namespace lightfc
