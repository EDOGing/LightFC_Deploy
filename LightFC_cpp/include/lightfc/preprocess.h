#pragma once

#include <net.h>
#include <opencv2/core.hpp>

#include "lightfc/types.h"

namespace lightfc {

struct CropResult {
    cv::Mat patch_bgr;
    double resize_factor = 1.0;
    int crop_x1 = 0;
    int crop_y1 = 0;
    int crop_size = 0;
    int pad_left = 0;
    int pad_top = 0;
    int pad_right = 0;
    int pad_bottom = 0;
};

// Python round(): ties are rounded to the nearest even integer. std::round()
// is not equivalent because it rounds half away from zero.
int python_round_to_int(double value);

// Exact port of web_demo.py::sample_target for a BGR OpenCV frame.
CropResult sample_target(const cv::Mat& frame_bgr,
                         const BBox& box,
                         double search_area_factor,
                         int output_size);

// Exact numeric ordering used by web_demo.py:
// BGR -> RGB, uint8 HWC -> float32 CHW, then
// (value / 255.0f - mean[channel]) / std[channel].
ncnn::Mat preprocess_bgr_to_rgb_chw(const cv::Mat& patch_bgr);

BBox clip_box(const BBox& box, int image_height, int image_width, double margin = 2.0);

}  // namespace lightfc
