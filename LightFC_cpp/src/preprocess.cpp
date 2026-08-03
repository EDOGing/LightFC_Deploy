#include "lightfc/preprocess.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace lightfc {

int python_round_to_int(const double value) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument("python_round_to_int received a non-finite value");
    }
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
        throw std::overflow_error("rounded crop coordinate exceeds int range");
    }
    return static_cast<int>(rounded);
}

CropResult sample_target(const cv::Mat& frame_bgr,
                         const BBox& box,
                         const double search_area_factor,
                         const int output_size) {
    if (frame_bgr.empty() || frame_bgr.type() != CV_8UC3) {
        throw std::invalid_argument("sample_target expects a non-empty CV_8UC3 BGR frame");
    }
    if (!std::isfinite(box.x) || !std::isfinite(box.y) ||
        !std::isfinite(box.width) || !std::isfinite(box.height) ||
        box.width <= 0.0 || box.height <= 0.0 ||
        search_area_factor <= 0.0 || output_size <= 0) {
        throw std::invalid_argument("invalid bbox or crop parameters");
    }

    // Keep the same double-precision geometry and ceil/round ordering as Python.
    const double raw_crop_size =
        std::sqrt(box.width * box.height) * search_area_factor;
    const int crop_size = static_cast<int>(std::ceil(raw_crop_size));
    if (crop_size < 1) {
        throw std::invalid_argument("target box is too small");
    }

    const int x1 = python_round_to_int(
        box.x + 0.5 * box.width - 0.5 * static_cast<double>(crop_size));
    const int y1 = python_round_to_int(
        box.y + 0.5 * box.height - 0.5 * static_cast<double>(crop_size));
    const int x2 = x1 + crop_size;
    const int y2 = y1 + crop_size;

    const int left = std::max(0, -x1);
    const int top = std::max(0, -y1);
    // The +1 is intentional and matches the current Python implementation.
    const int right = std::max(x2 - frame_bgr.cols + 1, 0);
    const int bottom = std::max(y2 - frame_bgr.rows + 1, 0);

    const int source_x1 = x1 + left;
    const int source_y1 = y1 + top;
    const int source_x2 = x2 - right;
    const int source_y2 = y2 - bottom;
    if (source_x1 < 0 || source_y1 < 0 || source_x2 > frame_bgr.cols ||
        source_y2 > frame_bgr.rows || source_x2 <= source_x1 || source_y2 <= source_y1) {
        throw std::runtime_error("crop lies completely outside the source frame");
    }

    const cv::Rect roi(source_x1, source_y1, source_x2 - source_x1, source_y2 - source_y1);
    cv::Mat cropped = frame_bgr(roi);
    cv::Mat padded;
    cv::copyMakeBorder(cropped, padded, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    if (padded.cols != crop_size || padded.rows != crop_size) {
        throw std::runtime_error("sample_target padding did not reconstruct crop_size square");
    }

    CropResult result;
    // INTER_LINEAR is OpenCV's default, stated explicitly for deployment clarity.
    cv::resize(padded, result.patch_bgr, cv::Size(output_size, output_size),
               0.0, 0.0, cv::INTER_LINEAR);
    result.resize_factor = static_cast<double>(output_size) / crop_size;
    result.crop_x1 = x1;
    result.crop_y1 = y1;
    result.crop_size = crop_size;
    result.pad_left = left;
    result.pad_top = top;
    result.pad_right = right;
    result.pad_bottom = bottom;
    return result;
}

ncnn::Mat preprocess_bgr_to_rgb_chw(const cv::Mat& patch_bgr) {
    if (patch_bgr.empty() || patch_bgr.type() != CV_8UC3 || !patch_bgr.isContinuous()) {
        if (patch_bgr.empty() || patch_bgr.type() != CV_8UC3) {
            throw std::invalid_argument("preprocess expects CV_8UC3 input");
        }
    }

    static constexpr float mean[3] = {0.485f, 0.456f, 0.406f};
    static constexpr float stddev[3] = {0.229f, 0.224f, 0.225f};

    ncnn::Mat output(patch_bgr.cols, patch_bgr.rows, 3);
    if (output.empty()) {
        throw std::bad_alloc();
    }
    float* out_r = output.channel(0);
    float* out_g = output.channel(1);
    float* out_b = output.channel(2);

    for (int y = 0; y < patch_bgr.rows; ++y) {
        const cv::Vec3b* row = patch_bgr.ptr<cv::Vec3b>(y);
        for (int x = 0; x < patch_bgr.cols; ++x) {
            const int index = y * patch_bgr.cols + x;
            // Match NumPy/PyTorch operation ordering instead of using a fused
            // mean*255 normalization, which has slightly different rounding.
            const float r = static_cast<float>(row[x][2]);
            const float g = static_cast<float>(row[x][1]);
            const float b = static_cast<float>(row[x][0]);
            out_r[index] = (r / 255.0f - mean[0]) / stddev[0];
            out_g[index] = (g / 255.0f - mean[1]) / stddev[1];
            out_b[index] = (b / 255.0f - mean[2]) / stddev[2];
        }
    }
    return output;
}

BBox clip_box(const BBox& box,
              const int image_height,
              const int image_width,
              const double margin) {
    double x1 = box.x;
    double y1 = box.y;
    double x2 = box.x + box.width;
    double y2 = box.y + box.height;
    x1 = std::min(std::max(0.0, x1), image_width - margin);
    x2 = std::min(std::max(margin, x2), static_cast<double>(image_width));
    y1 = std::min(std::max(0.0, y1), image_height - margin);
    y2 = std::min(std::max(margin, y2), static_cast<double>(image_height));
    return BBox{x1, y1, std::max(margin, x2 - x1), std::max(margin, y2 - y1)};
}

}  // namespace lightfc
