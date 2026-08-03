#include "lightfc/rtsp_service.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

namespace lightfc {
namespace {

using Clock = std::chrono::steady_clock;

bool valid_box(const BBox& box) {
    return std::isfinite(box.x) && std::isfinite(box.y) &&
           std::isfinite(box.width) && std::isfinite(box.height) &&
           box.width > 0.0 && box.height > 0.0;
}

}  // namespace

RtspTrackingService::RtspTrackingService(LightFCNcnnTracker& tracker) : tracker_(tracker) {}

RtspTrackingService::~RtspTrackingService() {
    disconnect();
}

bool RtspTrackingService::connect(const std::string& url, std::string* error) {
    if (url.empty()) {
        if (error != nullptr) {
            *error = "RTSP 地址不能为空";
        }
        return false;
    }
    disconnect();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        status_ = ServiceStatus{};
        status_.url = url;
        status_.message = "正在连接";
        latest_jpeg_.clear();
        frame_sequence_ = 0;
        pending_initialization_ = false;
        inference_ms_.clear();
        tracking_outputs_.clear();
        tracking_output_sequence_ = 0;
    }
    stop_.store(false);
    worker_ = std::thread(&RtspTrackingService::worker_loop, this, url);
    return true;
}

void RtspTrackingService::disconnect() {
    stop_.store(true);
    frame_ready_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
    tracker_.cancel();
    std::lock_guard<std::mutex> lock(mutex_);
    status_.connected = false;
    status_.tracking = false;
    status_.message = "未连接";
    pending_initialization_ = false;
}

bool RtspTrackingService::request_tracking(const BBox& box, std::string* error) {
    if (!valid_box(box)) {
        if (error != nullptr) {
            *error = "目标框必须是有限数值，且 width/height 必须大于 0";
        }
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!status_.connected || !status_.has_frame) {
        if (error != nullptr) {
            *error = "视频尚未连接或还没有可用画面";
        }
        return false;
    }
    pending_box_ = box;
    pending_initialization_ = true;
    status_.tracking = false;
    status_.message = "等待下一帧初始化目标";
    return true;
}

void RtspTrackingService::cancel_tracking() {
    tracker_.cancel();
    std::lock_guard<std::mutex> lock(mutex_);
    pending_initialization_ = false;
    status_.tracking = false;
    status_.confidence = 0.0;
    status_.message = status_.connected ? "已取消跟踪" : "未连接";
}

std::vector<TrackingOutput> RtspTrackingService::tracking_outputs_after(
    const std::uint64_t sequence,
    const std::size_t limit) const {
    std::vector<TrackingOutput> outputs;
    std::lock_guard<std::mutex> lock(mutex_);
    outputs.reserve(std::min(limit, tracking_outputs_.size()));
    for (const TrackingOutput& output : tracking_outputs_) {
        if (output.sequence > sequence) {
            outputs.push_back(output);
            if (outputs.size() >= limit) {
                break;
            }
        }
    }
    return outputs;
}

ServiceStatus RtspTrackingService::status() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return status_;
}

bool RtspTrackingService::snapshot(std::vector<unsigned char>& jpeg) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (latest_jpeg_.empty()) {
        return false;
    }
    jpeg = latest_jpeg_;
    return true;
}

bool RtspTrackingService::wait_for_jpeg(std::uint64_t& sequence,
                                        std::vector<unsigned char>& jpeg,
                                        const std::chrono::milliseconds timeout) const {
    std::unique_lock<std::mutex> lock(mutex_);
    frame_ready_.wait_for(lock, timeout, [&] {
        return stop_.load() || frame_sequence_ != sequence;
    });
    if (latest_jpeg_.empty() || frame_sequence_ == sequence) {
        return false;
    }
    sequence = frame_sequence_;
    jpeg = latest_jpeg_;
    return true;
}

void RtspTrackingService::set_message(const std::string& message) {
    std::lock_guard<std::mutex> lock(mutex_);
    status_.message = message;
}

void RtspTrackingService::publish_frame(const cv::Mat& frame) {
    std::vector<unsigned char> encoded;
    if (!cv::imencode(".jpg", frame, encoded, {cv::IMWRITE_JPEG_QUALITY, 85})) {
        set_message("JPEG 编码失败");
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        latest_jpeg_ = std::move(encoded);
        status_.has_frame = true;
        ++frame_sequence_;
    }
    frame_ready_.notify_all();
}

void RtspTrackingService::publish_tracking_output(const BBox& box,
                                                  const float confidence,
                                                  const std::uint64_t frame_id,
                                                  const int image_width,
                                                  const int image_height) {
    TrackingOutput output;
    output.frame_id = frame_id;
    output.timestamp_unix_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                   std::chrono::system_clock::now().time_since_epoch())
                                   .count();
    output.image_width = image_width;
    output.image_height = image_height;
    output.bbox = box;
    output.confidence = confidence;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        output.sequence = ++tracking_output_sequence_;
        tracking_outputs_.push_back(output);
        constexpr std::size_t kMaximumBufferedOutputs = 512;
        while (tracking_outputs_.size() > kMaximumBufferedOutputs) {
            tracking_outputs_.pop_front();
        }
    }
}

void RtspTrackingService::worker_loop(const std::string url) {
    while (!stop_.load()) {
        cv::VideoCapture capture;
        const std::vector<int> parameters{
            cv::CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
            cv::CAP_PROP_READ_TIMEOUT_MSEC, 3000};
        bool opened = capture.open(url, cv::CAP_FFMPEG, parameters);
        if (opened) {
            // Python calls set() after open(); CAP_PROP_BUFFERSIZE is not a valid
            // FFmpeg open parameter in the OpenCV binary distribution.
            capture.set(cv::CAP_PROP_BUFFERSIZE, 1);
        } else if (url.rfind("rtsp://", 0) != 0 && url.rfind("rtsps://", 0) != 0) {
            // Keep local files/image sequences useful for validation. RTSP remains
            // strict to rtsp_web_demo.py and does not silently switch backend.
            opened = capture.open(url);
        }
        if (!opened) {
            set_message("连接失败，1 秒后重试");
            for (int i = 0; i < 10 && !stop_.load(); ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            status_.connected = true;
            status_.message = "视频已连接";
        }

        cv::Mat frame;
        std::uint64_t source_frame_id = 0;
        while (!stop_.load()) {
            if (!capture.read(frame) || frame.empty()) {
                set_message("视频读取中断，正在重连");
                break;
            }
            ++source_frame_id;

            bool initialize_now = false;
            BBox initial_box;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (pending_initialization_) {
                    initialize_now = true;
                    initial_box = pending_box_;
                    pending_initialization_ = false;
                }
            }

            if (initialize_now) {
                std::string detail;
                const bool initialized = tracker_.initialize(frame, initial_box, &detail);
                if (initialized) {
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        status_.tracking = true;
                        status_.box = initial_box;
                        status_.confidence = 1.0;
                        status_.message = "跟踪中";
                    }
                    TrackingOutput output;
                    if (tracker_.get_tracking_output(output)) {
                        publish_tracking_output(output.bbox, output.confidence,
                                                source_frame_id, frame.cols, frame.rows);
                    }
                } else {
                    std::lock_guard<std::mutex> lock(mutex_);
                    status_.tracking = false;
                    status_.message = "初始化失败: " + detail;
                }
            } else if (tracker_.initialized()) {
                const auto begin = Clock::now();
                TrackResult result;
                std::string detail;
                const bool ok = tracker_.track(frame, result, &detail);
                const double elapsed = std::chrono::duration<double, std::milli>(
                                           Clock::now() - begin)
                                           .count();
                if (ok) {
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        inference_ms_.push_back(elapsed);
                        while (inference_ms_.size() > 20) {
                            inference_ms_.pop_front();
                        }
                        const double total = std::accumulate(inference_ms_.begin(),
                                                             inference_ms_.end(), 0.0);
                        status_.inference_fps = total > 0.0
                                                    ? 1000.0 * inference_ms_.size() / total
                                                    : 0.0;
                        status_.box = result.box;
                        status_.confidence = result.confidence;
                        status_.tracking = true;
                        status_.message = "跟踪中";
                    }
                    TrackingOutput output;
                    if (tracker_.get_tracking_output(output)) {
                        publish_tracking_output(output.bbox, output.confidence,
                                                source_frame_id, frame.cols, frame.rows);
                    }
                } else {
                    std::lock_guard<std::mutex> lock(mutex_);
                    status_.tracking = false;
                    status_.message = "推理失败: " + detail;
                }
            }

            ServiceStatus display;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                display = status_;
            }
            if (display.tracking) {
                const cv::Rect rectangle(
                    cvRound(display.box.x), cvRound(display.box.y),
                    std::max(1, cvRound(display.box.width)),
                    std::max(1, cvRound(display.box.height)));
                cv::rectangle(frame, rectangle, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
                const std::string label = cv::format("LightFC %.3f | %.1f FPS",
                                                     display.confidence,
                                                     display.inference_fps);
                cv::putText(frame, label,
                            cv::Point(rectangle.x, std::max(18, rectangle.y - 6)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 0),
                            2, cv::LINE_AA);
            }
            publish_frame(frame);
        }

        capture.release();
        tracker_.cancel();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            status_.connected = false;
            status_.tracking = false;
        }
        if (!stop_.load()) {
            for (int i = 0; i < 10 && !stop_.load(); ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
    }
}

}  // namespace lightfc
