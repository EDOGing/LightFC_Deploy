#include "lightfc/http_server.h"

#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <optional>
#include <regex>
#include <sstream>
#include <unordered_map>

namespace lightfc {
namespace {

struct Request {
    std::string method;
    std::string path;
    std::string query;
    std::unordered_map<std::string, std::string> headers;
    std::string body;
};

bool send_all(const SOCKET socket, const char* data, std::size_t length) {
    while (length > 0) {
        const int part = send(socket, data, static_cast<int>(std::min<std::size_t>(length, 1 << 20)), 0);
        if (part <= 0) {
            return false;
        }
        data += part;
        length -= static_cast<std::size_t>(part);
    }
    return true;
}

bool send_all(const SOCKET socket, const std::string& text) {
    return send_all(socket, text.data(), text.size());
}

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool read_request(const SOCKET socket, Request& request) {
    std::string data;
    char buffer[4096];
    std::size_t header_end = std::string::npos;
    while ((header_end = data.find("\r\n\r\n")) == std::string::npos) {
        const int received = recv(socket, buffer, sizeof(buffer), 0);
        if (received <= 0 || data.size() + received > 1024 * 1024) {
            return false;
        }
        data.append(buffer, received);
    }
    std::istringstream header_stream(data.substr(0, header_end));
    std::string line;
    if (!std::getline(header_stream, line)) {
        return false;
    }
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
    std::istringstream first_line(line);
    first_line >> request.method >> request.path;
    const auto query = request.path.find('?');
    if (query != std::string::npos) {
        request.query = request.path.substr(query + 1);
        request.path.resize(query);
    }
    while (std::getline(header_stream, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        const auto colon = line.find(':');
        if (colon != std::string::npos) {
            std::string key = lower(line.substr(0, colon));
            std::size_t value_start = colon + 1;
            while (value_start < line.size() && line[value_start] == ' ') {
                ++value_start;
            }
            request.headers[key] = line.substr(value_start);
        }
    }
    std::size_t content_length = 0;
    const auto found = request.headers.find("content-length");
    if (found != request.headers.end()) {
        try {
            content_length = static_cast<std::size_t>(std::stoul(found->second));
        } catch (...) {
            return false;
        }
    }
    request.body = data.substr(header_end + 4);
    while (request.body.size() < content_length) {
        const int received = recv(socket, buffer, sizeof(buffer), 0);
        if (received <= 0) {
            return false;
        }
        request.body.append(buffer, received);
    }
    request.body.resize(content_length);
    return !request.method.empty() && !request.path.empty();
}

std::string json_escape(const std::string& value) {
    std::ostringstream output;
    for (const unsigned char c : value) {
        switch (c) {
        case '\\': output << "\\\\"; break;
        case '"': output << "\\\""; break;
        case '\n': output << "\\n"; break;
        case '\r': output << "\\r"; break;
        case '\t': output << "\\t"; break;
        default:
            if (c < 0x20) {
                output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                       << static_cast<int>(c) << std::dec;
            } else {
                output << c;
            }
        }
    }
    return output.str();
}

void reply(const SOCKET socket,
           const int code,
           const char* reason,
           const char* content_type,
           const void* body,
           const std::size_t size) {
    std::ostringstream headers;
    headers << "HTTP/1.1 " << code << ' ' << reason << "\r\n"
            << "Content-Type: " << content_type << "\r\n"
            << "Content-Length: " << size << "\r\n"
            << "Cache-Control: no-store\r\n"
            << "Access-Control-Allow-Origin: *\r\n"
            << "Connection: close\r\n\r\n";
    if (send_all(socket, headers.str()) && size != 0) {
        send_all(socket, static_cast<const char*>(body), size);
    }
}

void reply_text(const SOCKET socket,
                const int code,
                const char* reason,
                const char* content_type,
                const std::string& body) {
    reply(socket, code, reason, content_type, body.data(), body.size());
}

std::optional<std::string> json_string(const std::string& body, const char* key) {
    const std::regex pattern(std::string("\\\"") + key +
                             "\\\"\\s*:\\s*\\\"((?:\\\\.|[^\\\"])*)\\\"");
    std::smatch match;
    if (!std::regex_search(body, match, pattern)) {
        return std::nullopt;
    }
    std::string value = match[1].str();
    value = std::regex_replace(value, std::regex("\\\\\\\""), "\"");
    value = std::regex_replace(value, std::regex("\\\\\\\\"), "\\");
    return value;
}

std::optional<double> json_number(const std::string& body, const char* key) {
    const std::regex pattern(std::string("\\\"") + key +
                             "\\\"\\s*:\\s*(-?(?:[0-9]+(?:\\.[0-9]*)?|\\.[0-9]+)(?:[eE][+-]?[0-9]+)?)");
    std::smatch match;
    if (!std::regex_search(body, match, pattern)) {
        return std::nullopt;
    }
    try {
        return std::stod(match[1].str());
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<std::uint64_t> query_unsigned(const std::string& query,
                                            const std::string& key) {
    std::size_t begin = 0;
    while (begin <= query.size()) {
        const std::size_t end = query.find('&', begin);
        const std::string item = query.substr(begin, end - begin);
        const std::size_t equals = item.find('=');
        if (equals != std::string::npos && item.substr(0, equals) == key) {
            try {
                std::size_t consumed = 0;
                const auto value = std::stoull(item.substr(equals + 1), &consumed);
                if (consumed == item.size() - equals - 1) {
                    return value;
                }
            } catch (...) {
            }
            return std::nullopt;
        }
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    return std::nullopt;
}

std::string tracking_outputs_json(const std::vector<TrackingOutput>& outputs) {
    std::ostringstream body;
    const std::uint64_t latest_sequence = outputs.empty() ? 0 : outputs.back().sequence;
    body << std::setprecision(8)
         << "{\"ok\":true,\"latest_sequence\":" << latest_sequence
         << ",\"outputs\":[";
    bool first = true;
    for (const TrackingOutput& output : outputs) {
        if (!first) {
            body << ',';
        }
        first = false;
        body << "{\"schema_version\":" << output.schema_version
             << ",\"sequence\":" << output.sequence
             << ",\"frame_id\":" << output.frame_id
             << ",\"timestamp_unix_ms\":" << output.timestamp_unix_ms
             << ",\"image_width\":" << output.image_width
             << ",\"image_height\":" << output.image_height
             << ",\"bbox\":{\"x\":" << output.bbox.x
             << ",\"y\":" << output.bbox.y
             << ",\"width\":" << output.bbox.width
             << ",\"height\":" << output.bbox.height
             << "},\"confidence\":" << output.confidence << '}';
    }
    body << "]}";
    return body.str();
}

std::string status_json(const ServiceStatus& status) {
    std::ostringstream body;
    body << std::boolalpha << std::setprecision(8)
         << "{\"ok\":true,\"connected\":" << status.connected
         << ",\"tracking\":" << status.tracking
         << ",\"has_frame\":" << status.has_frame
         << ",\"url\":\"" << json_escape(status.url)
         << "\",\"message\":\"" << json_escape(status.message)
         << "\",\"backend\":\"ncnn\",\"fps\":" << status.inference_fps
         << ",\"confidence\":" << status.confidence
         << ",\"bbox\":{\"x\":" << status.box.x
         << ",\"y\":" << status.box.y
         << ",\"width\":" << status.box.width
         << ",\"height\":" << status.box.height << "}}";
    return body.str();
}

}  // namespace

HttpServer::HttpServer(RtspTrackingService& service, std::filesystem::path web_root)
    : service_(service), web_root_(std::move(web_root)) {}

HttpServer::~HttpServer() {
    stop();
}

bool HttpServer::start(const std::string& address, const std::uint16_t port, std::string* error) {
    WSADATA data{};
    if (WSAStartup(MAKEWORD(2, 2), &data) != 0) {
        if (error != nullptr) *error = "WSAStartup 失败";
        return false;
    }
    const SOCKET listener = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (listener == INVALID_SOCKET) {
        if (error != nullptr) *error = "创建监听 socket 失败";
        WSACleanup();
        return false;
    }
    BOOL reuse = TRUE;
    setsockopt(listener, SOL_SOCKET, SO_REUSEADDR,
               reinterpret_cast<const char*>(&reuse), sizeof(reuse));
    sockaddr_in endpoint{};
    endpoint.sin_family = AF_INET;
    endpoint.sin_port = htons(port);
    if (inet_pton(AF_INET, address.c_str(), &endpoint.sin_addr) != 1 ||
        bind(listener, reinterpret_cast<const sockaddr*>(&endpoint), sizeof(endpoint)) == SOCKET_ERROR ||
        listen(listener, SOMAXCONN) == SOCKET_ERROR) {
        if (error != nullptr) *error = "绑定 HTTP 地址失败: " + address + ':' + std::to_string(port);
        closesocket(listener);
        WSACleanup();
        return false;
    }
    stopping_.store(false);
    listen_socket_ = static_cast<std::uintptr_t>(listener);
    accept_thread_ = std::thread(&HttpServer::accept_loop, this);
    return true;
}

void HttpServer::stop() {
    if (stopping_.exchange(true)) {
        return;
    }
    const SOCKET listener = static_cast<SOCKET>(listen_socket_);
    if (listener != INVALID_SOCKET) {
        shutdown(listener, SD_BOTH);
        closesocket(listener);
        listen_socket_ = static_cast<std::uintptr_t>(INVALID_SOCKET);
    }
    if (accept_thread_.joinable()) {
        accept_thread_.join();
    }
    std::vector<std::thread> clients;
    {
        std::lock_guard<std::mutex> lock(clients_mutex_);
        clients.swap(client_threads_);
    }
    for (auto& client : clients) {
        if (client.joinable()) client.join();
    }
    WSACleanup();
}

void HttpServer::accept_loop() {
    const SOCKET listener = static_cast<SOCKET>(listen_socket_);
    while (!stopping_.load()) {
        const SOCKET client = accept(listener, nullptr, nullptr);
        if (client == INVALID_SOCKET) {
            break;
        }
        DWORD timeout = 5000;
        setsockopt(client, SOL_SOCKET, SO_RCVTIMEO,
                   reinterpret_cast<const char*>(&timeout), sizeof(timeout));
        std::lock_guard<std::mutex> lock(clients_mutex_);
        client_threads_.emplace_back(&HttpServer::handle_client, this,
                                     static_cast<std::uintptr_t>(client));
    }
}

void HttpServer::handle_client(const std::uintptr_t socket_value) {
    const SOCKET client = static_cast<SOCKET>(socket_value);
    Request request;
    if (!read_request(client, request)) {
        closesocket(client);
        return;
    }

    if (request.method == "GET" && request.path == "/api/stream") {
        const std::string headers =
            "HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary=frame\r\n"
            "Cache-Control: no-store\r\nConnection: close\r\n\r\n";
        if (send_all(client, headers)) {
            std::uint64_t sequence = 0;
            while (!stopping_.load()) {
                std::vector<unsigned char> jpeg;
                if (!service_.wait_for_jpeg(sequence, jpeg, std::chrono::milliseconds(1000))) {
                    continue;
                }
                std::ostringstream part;
                part << "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                     << jpeg.size() << "\r\n\r\n";
                if (!send_all(client, part.str()) ||
                    !send_all(client, reinterpret_cast<const char*>(jpeg.data()), jpeg.size()) ||
                    !send_all(client, "\r\n")) {
                    break;
                }
            }
        }
    } else if (request.method == "GET" && request.path == "/api/snapshot") {
        std::vector<unsigned char> jpeg;
        if (service_.snapshot(jpeg)) {
            reply(client, 200, "OK", "image/jpeg", jpeg.data(), jpeg.size());
        } else {
            reply_text(client, 404, "Not Found", "application/json; charset=utf-8",
                       "{\"ok\":false,\"error\":\"尚无画面\"}");
        }
    } else if (request.method == "GET" && request.path == "/api/status") {
        reply_text(client, 200, "OK", "application/json; charset=utf-8",
                   status_json(service_.status()));
    } else if (request.method == "GET" && request.path == "/api/tracking-output") {
        const std::uint64_t after = query_unsigned(request.query, "after").value_or(0);
        const std::uint64_t requested_limit = query_unsigned(request.query, "limit").value_or(200);
        const std::size_t limit = static_cast<std::size_t>(
            std::clamp<std::uint64_t>(requested_limit, 1, 512));
        reply_text(client, 200, "OK", "application/json; charset=utf-8",
                   tracking_outputs_json(service_.tracking_outputs_after(after, limit)));
    } else if (request.method == "GET" && request.path == "/api/config") {
        reply_text(client, 200, "OK", "application/json; charset=utf-8",
                   "{\"ok\":true,\"backends\":[\"ncnn\"],\"default_backend\":\"ncnn\"}");
    } else if (request.method == "POST" && request.path == "/api/connect") {
        const auto url = json_string(request.body, "url");
        std::string detail;
        if (url && service_.connect(*url, &detail)) {
            reply_text(client, 200, "OK", "application/json; charset=utf-8",
                       "{\"ok\":true}");
        } else {
            reply_text(client, 400, "Bad Request", "application/json; charset=utf-8",
                       "{\"ok\":false,\"error\":\"" + json_escape(detail) + "\"}");
        }
    } else if (request.method == "POST" && request.path == "/api/disconnect") {
        service_.disconnect();
        reply_text(client, 200, "OK", "application/json; charset=utf-8", "{\"ok\":true}");
    } else if (request.method == "POST" && request.path == "/api/cancel") {
        service_.cancel_tracking();
        reply_text(client, 200, "OK", "application/json; charset=utf-8", "{\"ok\":true}");
    } else if (request.method == "POST" && request.path == "/api/track") {
        const auto x = json_number(request.body, "x");
        const auto y = json_number(request.body, "y");
        const auto width = json_number(request.body, "width");
        const auto height = json_number(request.body, "height");
        std::string detail;
        if (x && y && width && height &&
            service_.request_tracking(BBox{*x, *y, *width, *height}, &detail)) {
            reply_text(client, 200, "OK", "application/json; charset=utf-8",
                       "{\"ok\":true}");
        } else {
            if (detail.empty()) detail = "bbox 需要 x/y/width/height";
            reply_text(client, 400, "Bad Request", "application/json; charset=utf-8",
                       "{\"ok\":false,\"error\":\"" + json_escape(detail) + "\"}");
        }
    } else if (request.method == "GET" && (request.path == "/" || request.path == "/index.html")) {
        std::ifstream input(web_root_ / "index.html", std::ios::binary);
        if (!input) {
            reply_text(client, 404, "Not Found", "text/plain; charset=utf-8", "web/index.html 不存在");
        } else {
            const std::string page((std::istreambuf_iterator<char>(input)),
                                   std::istreambuf_iterator<char>());
            reply_text(client, 200, "OK", "text/html; charset=utf-8", page);
        }
    } else {
        reply_text(client, 404, "Not Found", "application/json; charset=utf-8",
                   "{\"ok\":false,\"error\":\"not found\"}");
    }
    shutdown(client, SD_BOTH);
    closesocket(client);
}

}  // namespace lightfc
