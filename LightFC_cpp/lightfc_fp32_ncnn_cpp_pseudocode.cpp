// Historical six-graph LightFC FP32 ncnn pipeline pseudocode.
// The production implementation now uses lightfc_template plus the merged
// lightfc_tracking graph. See src/tracker.cpp and tools/merge_tracking_ncnn.py.
// This file communicates architecture and data flow; adapt error handling,
// image utilities, task scheduling, and types to the destination project.

#include <net.h>
#include <opencv2/opencv.hpp>
#include <future>
#include <stdexcept>
#include <string>
#include <vector>

struct BBox {
    float x, y, w, h;  // xywh in original-image pixels
};

struct TrackResult {
    BBox box;
    float confidence;
};

class LightFCNcnnTracker {
public:
    bool load(const std::string& model_dir, int total_cpu_threads = 4) {
        // Six independent graphs = twelve files.
        configure(template_net_, total_cpu_threads);
        configure(search_net_,   total_cpu_threads);
        configure(fusion_net_,   total_cpu_threads);

        // If heads run sequentially, each can use total_cpu_threads.
        // If the three heads run concurrently, set each to 1 (or divide the
        // CPU thread budget) to prevent 3 * total_cpu_threads oversubscription.
        const int head_threads = run_heads_in_parallel_ ? 1 : total_cpu_threads;
        configure(score_net_,  head_threads);
        configure(size_net_,   head_threads);
        configure(offset_net_, head_threads);

        return load_pair(template_net_, model_dir, "lightfc_template") &&
               load_pair(search_net_,   model_dir, "lightfc_search") &&
               load_pair(fusion_net_,   model_dir, "lightfc_fusion") &&
               load_pair(score_net_,    model_dir, "lightfc_score") &&
               load_pair(size_net_,     model_dir, "lightfc_size") &&
               load_pair(offset_net_,   model_dir, "lightfc_offset");
    }

    // Called once when the user draws the initial target rectangle, and again
    // whenever the target is re-selected. It does NOT run every video frame.
    bool initialize(const cv::Mat& first_bgr_frame, const BBox& initial_box) {
        cv::Mat template_patch = sample_target(
            first_bgr_frame, initial_box, template_factor_, template_size_);
        ncnn::Mat template_input = preprocess_rgb_chw(template_patch);

        ncnn::Extractor ex = template_net_.create_extractor();
        if (ex.input("in0", template_input) != 0) return false;
        if (ex.extract("out0", template_feature_) != 0) return false;

        // template_feature_: C=96, H=8, W=8. Cache until re-initialization.
        state_ = initial_box;
        initialized_ = true;
        return true;
    }

    // Called once per incoming frame.
    TrackResult track(const cv::Mat& current_bgr_frame) {
        if (!initialized_) throw std::runtime_error("tracker is not initialized");

        float resize_factor = 1.0f;
        cv::Mat search_patch = sample_target(
            current_bgr_frame, state_, search_factor_, search_size_, &resize_factor);
        ncnn::Mat search_input = preprocess_rgb_chw(search_patch);

        // Stage 1 (serial dependency): search image -> search backbone.
        ncnn::Mat search_feature;  // C=96, H=16, W=16
        {
            ncnn::Extractor ex = search_net_.create_extractor();
            check(ex.input("in0", search_input));
            check(ex.extract("out0", search_feature));
        }

        // Stage 2 (serial dependency): cached template + search -> fusion.
        ncnn::Mat fused_feature;  // C=192, H=16, W=16
        {
            ncnn::Extractor ex = fusion_net_.create_extractor();
            check(ex.input("in0", template_feature_)); // template feature
            check(ex.input("in1", search_feature));   // current search feature
            check(ex.extract("out0", fused_feature));
        }

        // Stage 3: all heads consume the same immutable fused_feature.
        // There are no dependencies among score/size/offset, so they may run
        // sequentially or concurrently. Each task must create its own Extractor.
        ncnn::Mat score_map;   // C=1, H=16, W=16
        ncnn::Mat size_map;    // C=2, H=16, W=16: [width, height]
        ncnn::Mat offset_map;  // C=2, H=16, W=16: [dx, dy]

        if (run_heads_in_parallel_) {
            auto score_job = std::async(std::launch::async, [&] {
                score_map = run_one_input_net(score_net_, fused_feature);
            });
            auto size_job = std::async(std::launch::async, [&] {
                size_map = run_one_input_net(size_net_, fused_feature);
            });
            auto offset_job = std::async(std::launch::async, [&] {
                offset_map = run_one_input_net(offset_net_, fused_feature);
            });
            score_job.get();
            size_job.get();
            offset_job.get();
        } else {
            score_map  = run_one_input_net(score_net_, fused_feature);
            size_map   = run_one_input_net(size_net_, fused_feature);
            offset_map = run_one_input_net(offset_net_, fused_feature);
        }

        // pred_boxes is deliberately absent from all ncnn graphs. Decode it in
        // ordinary C++ from score_map, size_map and offset_map.
        TrackResult result = decode_box(
            score_map, size_map, offset_map, resize_factor,
            current_bgr_frame.cols, current_bgr_frame.rows, state_);
        state_ = result.box;
        return result;
    }

    void cancel_tracking() {
        initialized_ = false;
        template_feature_ = ncnn::Mat();
    }

private:
    ncnn::Net template_net_, search_net_, fusion_net_;
    ncnn::Net score_net_, size_net_, offset_net_;
    ncnn::Mat template_feature_;
    BBox state_{};
    bool initialized_ = false;
    bool run_heads_in_parallel_ = true;

    static constexpr int template_size_ = 128;
    static constexpr int search_size_ = 256;
    static constexpr int feature_size_ = 16;
    static constexpr float template_factor_ = 2.0f; // Read actual value from YAML/config.
    static constexpr float search_factor_ = 4.0f;   // Read actual value from YAML/config.

    static void configure(ncnn::Net& net, int threads) {
        net.opt.use_vulkan_compute = false; // CPU only
        net.opt.num_threads = threads;
        net.opt.use_fp16_storage = false;   // exported FP32 baseline
        net.opt.use_fp16_arithmetic = false;
    }

    static bool load_pair(ncnn::Net& net, const std::string& dir, const std::string& stem) {
        return net.load_param((dir + "/" + stem + ".ncnn.param").c_str()) == 0 &&
               net.load_model((dir + "/" + stem + ".ncnn.bin").c_str()) == 0;
    }

    static ncnn::Mat run_one_input_net(ncnn::Net& net, const ncnn::Mat& input) {
        ncnn::Extractor ex = net.create_extractor();
        check(ex.input("in0", input));
        ncnn::Mat output;
        check(ex.extract("out0", output));
        return output;
    }

    static ncnn::Mat preprocess_rgb_chw(const cv::Mat& bgr_patch) {
        // Required behavior:
        //   1. BGR -> RGB
        //   2. resize has already produced 128x128 or 256x256
        //   3. convert uint8 HWC to float32 CHW
        //   4. value = (value / 255 - mean[channel]) / std[channel]
        //      mean = {0.485, 0.456, 0.406}
        //      std  = {0.229, 0.224, 0.225}
        // Return ncnn::Mat with w=width, h=height, c=3.
        return ncnn::Mat(); // Implement using cv::cvtColor + ncnn::Mat/from_pixels.
    }

    static cv::Mat sample_target(
        const cv::Mat& frame, const BBox& box, float factor, int output_size,
        float* resize_factor = nullptr) {
        // Match Python sample_target exactly:
        // crop_size = ceil(sqrt(box.w * box.h) * factor)
        // crop around the box center, pad outside-image pixels consistently,
        // resize to output_size x output_size.
        // resize_factor = output_size / crop_size.
        return cv::Mat(); // Project-specific implementation.
    }

    static TrackResult decode_box(
        const ncnn::Mat& score, const ncnn::Mat& size, const ncnn::Mat& offset,
        float resize_factor, int image_width, int image_height, const BBox& previous) {
        // Precompute once: hann[y,x] = 2D centered Hann window, 16x16.
        // response[y,x] = score.channel(0)[y,x] * hann[y,x]
        // (best_y,best_x) = argmax(response)
        //
        // float normalized_cx = (best_x + offset.channel(0)[best_y,best_x]) / 16;
        // float normalized_cy = (best_y + offset.channel(1)[best_y,best_x]) / 16;
        // float normalized_w  = size.channel(0)[best_y,best_x];
        // float normalized_h  = size.channel(1)[best_y,best_x];
        //
        // float scale = 256.0f / resize_factor;
        // cx = normalized_cx * scale;
        // cy = normalized_cy * scale;
        // w  = normalized_w  * scale;
        // h  = normalized_h  * scale;
        //
        // previous_cx = previous.x + previous.w / 2;
        // previous_cy = previous.y + previous.h / 2;
        // half_search = 0.5f * scale;
        // result.x = cx + previous_cx - half_search - w / 2;
        // result.y = cy + previous_cy - half_search - h / 2;
        // result.w = w; result.h = h;
        // Clip xywh to image boundaries using the same margin policy as Python.
        // confidence can use raw score at (best_y,best_x).
        return {};
    }

    static void check(int return_code) {
        if (return_code != 0) throw std::runtime_error("ncnn input/extract failed");
    }
};

/*
Data-flow summary
-----------------

Initialization (once):
    template frame -> crop/normalize -> template_net -> cached template_feature

Every frame (serial until fusion):
    current frame -> crop/normalize -> search_net -> search_feature
    cached template_feature + search_feature -> fusion_net -> fused_feature

Parallelizable fan-out:
                               +-> score_net  -> score_map  --+
    fused_feature (read-only) --+-> size_net   -> size_map   --+-> C++ bbox decode
                               +-> offset_net -> offset_map --+

Do not process multiple camera frames concurrently with the same tracker state.
If capture is faster than inference, keep only the newest frame in a size-1 queue.
*/
