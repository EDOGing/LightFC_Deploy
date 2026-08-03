#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "lightfc/rtsp_service.h"

namespace lightfc {

class HttpServer {
public:
    HttpServer(RtspTrackingService& service, std::filesystem::path web_root);
    ~HttpServer();

    bool start(const std::string& address, std::uint16_t port, std::string* error = nullptr);
    void stop();

private:
    void accept_loop();
    void handle_client(std::uintptr_t socket_value);

    RtspTrackingService& service_;
    std::filesystem::path web_root_;
    std::atomic<bool> stopping_{false};
    std::uintptr_t listen_socket_ = static_cast<std::uintptr_t>(-1);
    std::thread accept_thread_;
    std::mutex clients_mutex_;
    std::vector<std::thread> client_threads_;
};

}  // namespace lightfc
