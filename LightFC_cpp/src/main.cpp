#include <windows.h>
#include <shellapi.h>

#include <filesystem>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

#include "lightfc/http_server.h"
#include "lightfc/rtsp_service.h"
#include "lightfc/tracker.h"

namespace {

std::filesystem::path executable_directory() {
    std::wstring path(32768, L'\0');
    const DWORD length = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
    path.resize(length);
    return std::filesystem::path(path).parent_path();
}

void print_usage() {
    std::cout << "LightFC NCNN RTSP Web Demo\n"
              << "  lightfc_web.exe [--model DIR] [--web DIR] [--port N]"
                 " [--threads N] [--no-open] [--url RTSP]"
                 " [--run-seconds N]\n";
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
    SetConsoleOutputCP(CP_UTF8);
    const auto exe_dir = executable_directory();
    std::filesystem::path model_dir = exe_dir / "model";
    std::filesystem::path web_dir = exe_dir / "web";
    std::uint16_t port = 8080;
    int threads = 4;
    bool open_browser = true;
    int run_seconds = 0;
    std::string initial_url;

    for (int i = 1; i < argc; ++i) {
        const std::wstring argument = argv[i];
        auto next = [&]() -> std::wstring {
            if (++i >= argc) throw std::runtime_error("参数缺少值");
            return argv[i];
        };
        try {
            if (argument == L"--model") model_dir = next();
            else if (argument == L"--web") web_dir = next();
            else if (argument == L"--port") port = static_cast<std::uint16_t>(std::stoul(next()));
            else if (argument == L"--threads") threads = std::stoi(next());
            else if (argument == L"--run-seconds") run_seconds = std::stoi(next());
            else if (argument == L"--url") {
                const std::wstring wide = next();
                const int count = WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), -1, nullptr, 0, nullptr, nullptr);
                std::string utf8(static_cast<std::size_t>(count), '\0');
                WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), -1, utf8.data(), count, nullptr, nullptr);
                utf8.pop_back();
                initial_url = std::move(utf8);
            } else if (argument == L"--no-open") open_browser = false;
            else if (argument == L"--help" || argument == L"-h") {
                print_usage();
                return 0;
            } else {
                std::wcerr << L"未知参数: " << argument << L'\n';
                print_usage();
                return 2;
            }
        } catch (const std::exception& exception) {
            std::cerr << "参数错误: " << exception.what() << '\n';
            return 2;
        }
    }

    lightfc::LightFCNcnnTracker tracker;
    std::string error;
    if (!tracker.load(model_dir, lightfc::TrackerOptions{threads}, &error)) {
        std::cerr << "模型加载失败: " << error << '\n';
        return 1;
    }

    lightfc::RtspTrackingService service(tracker);
    lightfc::HttpServer server(service, web_dir);
    if (!server.start("127.0.0.1", port, &error)) {
        std::cerr << "HTTP 服务启动失败: " << error << '\n';
        return 1;
    }
    if (!initial_url.empty()) {
        service.connect(initial_url, &error);
    }

    const std::wstring page = L"http://127.0.0.1:" + std::to_wstring(port) + L"/";
    std::wcout << L"LightFC 已启动: " << page << L"\n按 Enter 退出。\n";
    if (open_browser) {
        ShellExecuteW(nullptr, L"open", page.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
    }
    if (run_seconds > 0) {
        std::this_thread::sleep_for(std::chrono::seconds(run_seconds));
    } else {
        std::string line;
        std::getline(std::cin, line);
    }

    server.stop();
    service.disconnect();
    return 0;
}
