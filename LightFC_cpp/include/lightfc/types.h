#pragma once

namespace lightfc {

struct BBox {
    double x = 0.0;
    double y = 0.0;
    double width = 0.0;
    double height = 0.0;
};

struct TrackResult {
    BBox box;
    float confidence = 0.0f;
    int peak_x = 0;
    int peak_y = 0;
};

}  // namespace lightfc
