#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>

#include <net.h>

namespace {

bool load(ncnn::Net& net, const std::filesystem::path& directory, const char* stem) {
    net.opt.num_threads = 1;
    net.opt.use_vulkan_compute = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_fp16_packed = false;
    net.opt.use_bf16_storage = false;
    const auto param = (directory / (std::string(stem) + ".param")).string();
    const auto bin = (directory / (std::string(stem) + ".bin")).string();
    return net.load_param(param.c_str()) == 0 && net.load_model(bin.c_str()) == 0;
}

ncnn::Mat run_one(ncnn::Net& net, const ncnn::Mat& input) {
    ncnn::Extractor extractor = net.create_extractor();
    extractor.input("in0", input);
    ncnn::Mat output;
    extractor.extract("out0", output);
    return output;
}

float maximum_error(const ncnn::Mat& expected, const ncnn::Mat& actual) {
    if (expected.w != actual.w || expected.h != actual.h || expected.c != actual.c ||
        expected.elempack != 1 || actual.elempack != 1) {
        return INFINITY;
    }
    const std::size_t count = expected.total();
    const float* a = expected;
    const float* b = actual;
    float result = 0.0f;
    for (std::size_t i = 0; i < count; ++i) {
        result = std::max(result, std::abs(a[i] - b[i]));
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: lightfc_compare_models MODEL_DIR\n";
        return 2;
    }
    const std::filesystem::path directory = argv[1];
    ncnn::Net search_net, fusion_net, score_net, size_net, offset_net, tracking_net;
    if (!load(search_net, directory, "lightfc_search.ncnn") ||
        !load(fusion_net, directory, "lightfc_fusion.ncnn") ||
        !load(score_net, directory, "lightfc_score.ncnn") ||
        !load(size_net, directory, "lightfc_size.ncnn") ||
        !load(offset_net, directory, "lightfc_offset.ncnn") ||
        !load(tracking_net, directory, "lightfc_tracking.ncnn")) {
        std::cerr << "model load failed\n";
        return 1;
    }
    std::cout << "models loaded" << std::endl;

    ncnn::Mat search(256, 256, 3);
    ncnn::Mat template_features(8, 8, 96);
    for (std::size_t i = 0; i < search.total(); ++i) {
        static_cast<float*>(search)[i] = std::sin(static_cast<float>(i) * 0.001f);
    }
    for (std::size_t i = 0; i < template_features.total(); ++i) {
        static_cast<float*>(template_features)[i] = std::cos(static_cast<float>(i) * 0.003f);
    }

    const ncnn::Mat search_features = run_one(search_net, search);
    std::cout << "old search done" << std::endl;
    ncnn::Mat fused_features;
    {
        ncnn::Extractor extractor = fusion_net.create_extractor();
        extractor.input("in0", template_features);
        extractor.input("in1", search_features);
        extractor.extract("out0", fused_features);
    }
    std::cout << "old fusion done" << std::endl;
    const ncnn::Mat expected_score = run_one(score_net, fused_features);
    std::cout << "old score done" << std::endl;
    const ncnn::Mat expected_size = run_one(size_net, fused_features);
    std::cout << "old size done" << std::endl;
    const ncnn::Mat expected_offset = run_one(offset_net, fused_features);
    std::cout << "old offset done" << std::endl;

    ncnn::Mat actual_score, actual_size, actual_offset;
    {
        ncnn::Extractor extractor = tracking_net.create_extractor();
        extractor.input("template_features", template_features);
        extractor.input("search", search);
        std::cout << "merged inputs done" << std::endl;
        extractor.extract("score_map", actual_score);
        std::cout << "merged score done" << std::endl;
        extractor.extract("size_map", actual_size);
        std::cout << "merged size done" << std::endl;
        extractor.extract("offset_map", actual_offset);
        std::cout << "merged offset done" << std::endl;
    }
    const float score_error = maximum_error(expected_score, actual_score);
    const float size_error = maximum_error(expected_size, actual_size);
    const float offset_error = maximum_error(expected_offset, actual_offset);
    std::cout << "score max_abs_error=" << score_error << '\n'
              << "size max_abs_error=" << size_error << '\n'
              << "offset max_abs_error=" << offset_error << '\n';
    return score_error <= 1e-5f && size_error <= 1e-5f && offset_error <= 1e-5f ? 0 : 1;
}
