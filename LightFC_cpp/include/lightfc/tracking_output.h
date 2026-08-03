#pragma once

#include <cstdint>

#include "lightfc/types.h"

namespace lightfc {

// Versioned per-frame payload for PTZ control or other downstream consumers.
// New fields can be appended while schema_version tells consumers how to parse it.
struct TrackingOutput {
    std::uint32_t schema_version = 1;
    std::uint64_t sequence = 0;
    std::uint64_t frame_id = 0;
    std::int64_t timestamp_unix_ms = 0;
    int image_width = 0;
    int image_height = 0;
    BBox bbox{};
    float confidence = 0.0f;
};

}  // namespace lightfc
