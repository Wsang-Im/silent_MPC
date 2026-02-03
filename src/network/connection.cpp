#include "silentmpc/network/connection.hpp"
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstring>
#include <errno.h>
#include <sstream>

namespace silentmpc::network {

// ============================================================================
// TCPConnection Implementation
// ============================================================================

TCPConnection::TCPConnection()
    : socket_fd_(-1), peer_port_(0), connected_(false),
      bytes_sent_(0), bytes_received_(0) {}

TCPConnection::~TCPConnection() {
    if (socket_fd_ >= 0) {
        close();
    }
}

TCPConnection::TCPConnection(TCPConnection&& other) noexcept
    : socket_fd_(other.socket_fd_),
      peer_address_(std::move(other.peer_address_)),
      peer_port_(other.peer_port_),
      connected_(other.connected_),
      bytes_sent_(other.bytes_sent_),
      bytes_received_(other.bytes_received_) {
    other.socket_fd_ = -1;
    other.connected_ = false;
}

TCPConnection& TCPConnection::operator=(TCPConnection&& other) noexcept {
    if (this != &other) {
        if (socket_fd_ >= 0) {
            ::close(socket_fd_);
        }

        socket_fd_ = other.socket_fd_;
        peer_address_ = std::move(other.peer_address_);
        peer_port_ = other.peer_port_;
        connected_ = other.connected_;
        bytes_sent_ = other.bytes_sent_;
        bytes_received_ = other.bytes_received_;

        other.socket_fd_ = -1;
        other.connected_ = false;
    }
    return *this;
}

void TCPConnection::connect(const std::string& host, uint16_t port) {
    if (connected_) {
        throw ConnectionError("Already connected");
    }

    // Create socket
    socket_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_fd_ < 0) {
        throw ConnectionError("Failed to create socket: " + std::string(strerror(errno)));
    }

    // Disable Nagle's algorithm for low latency
    int flag = 1;
    if (setsockopt(socket_fd_, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag)) < 0) {
        ::close(socket_fd_);
        socket_fd_ = -1;
        throw ConnectionError("Failed to set TCP_NODELAY: " + std::string(strerror(errno)));
    }

    // Setup address
    struct sockaddr_in server_addr;
    std::memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);

    // Convert hostname to IP
    if (inet_pton(AF_INET, host.c_str(), &server_addr.sin_addr) <= 0) {
        ::close(socket_fd_);
        socket_fd_ = -1;
        throw ConnectionError("Invalid address: " + host);
    }

    // Connect
    if (::connect(socket_fd_, reinterpret_cast<struct sockaddr*>(&server_addr),
                  sizeof(server_addr)) < 0) {
        int err = errno;
        ::close(socket_fd_);
        socket_fd_ = -1;
        throw ConnectionError("Failed to connect to " + host + ":" + std::to_string(port) +
                             " - " + strerror(err));
    }

    peer_address_ = host;
    peer_port_ = port;
    connected_ = true;
}

void TCPConnection::listen_and_accept(uint16_t port, int backlog) {
    if (connected_) {
        throw ConnectionError("Already connected");
    }

    // Create listening socket
    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        throw ConnectionError("Failed to create socket: " + std::string(strerror(errno)));
    }

    // Allow reuse of address
    int flag = 1;
    if (setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &flag, sizeof(flag)) < 0) {
        ::close(listen_fd);
        throw ConnectionError("Failed to set SO_REUSEADDR: " + std::string(strerror(errno)));
    }

    // Bind
    struct sockaddr_in server_addr;
    std::memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port);

    if (bind(listen_fd, reinterpret_cast<struct sockaddr*>(&server_addr),
             sizeof(server_addr)) < 0) {
        int err = errno;
        ::close(listen_fd);
        throw ConnectionError("Failed to bind to port " + std::to_string(port) +
                             " - " + strerror(err));
    }

    // Listen
    if (listen(listen_fd, backlog) < 0) {
        int err = errno;
        ::close(listen_fd);
        throw ConnectionError("Failed to listen: " + std::string(strerror(err)));
    }

    // Accept one connection
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);

    socket_fd_ = accept(listen_fd, reinterpret_cast<struct sockaddr*>(&client_addr),
                        &client_len);

    // Close listening socket
    ::close(listen_fd);

    if (socket_fd_ < 0) {
        throw ConnectionError("Failed to accept connection: " + std::string(strerror(errno)));
    }

    // Disable Nagle's algorithm
    flag = 1;
    if (setsockopt(socket_fd_, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag)) < 0) {
        ::close(socket_fd_);
        socket_fd_ = -1;
        throw ConnectionError("Failed to set TCP_NODELAY: " + std::string(strerror(errno)));
    }

    // Get peer info
    char peer_ip[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &client_addr.sin_addr, peer_ip, sizeof(peer_ip));
    peer_address_ = peer_ip;
    peer_port_ = ntohs(client_addr.sin_port);
    connected_ = true;
}

void TCPConnection::send(std::span<const uint8_t> data) {
    if (!connected_) {
        throw SendError("Not connected");
    }

    send_all(data.data(), data.size());
    bytes_sent_ += data.size();
}

void TCPConnection::send_with_length(std::span<const uint8_t> data) {
    if (data.size() > 0xFFFFFFFF) {
        throw SendError("Data too large (max 4GB)");
    }

    // Send length (4 bytes, big-endian)
    uint8_t length_buf[4];
    write_uint32_be(length_buf, static_cast<uint32_t>(data.size()));
    send(length_buf);

    // Send data
    send(data);
}

std::vector<uint8_t> TCPConnection::receive(size_t size) {
    if (!connected_) {
        throw ReceiveError("Not connected");
    }

    std::vector<uint8_t> buffer(size);
    receive_all(buffer.data(), size);
    bytes_received_ += size;
    return buffer;
}

std::vector<uint8_t> TCPConnection::receive_with_length() {
    // Receive length (4 bytes, big-endian)
    auto length_buf = receive(4);
    uint32_t length = read_uint32_be(length_buf.data());

    if (length == 0) {
        return {};
    }

    if (length > 100'000'000) {  // Sanity check: 100 MB max
        throw ReceiveError("Received length too large: " + std::to_string(length));
    }

    // Receive data
    return receive(length);
}

void TCPConnection::close() {
    if (socket_fd_ >= 0) {
        // Shutdown both directions
        shutdown(socket_fd_, SHUT_RDWR);
        ::close(socket_fd_);
        socket_fd_ = -1;
        connected_ = false;
    }
}

void TCPConnection::send_all(const uint8_t* data, size_t size) {
    size_t total_sent = 0;

    while (total_sent < size) {
        ssize_t sent = ::send(socket_fd_, data + total_sent, size - total_sent, 0);

        if (sent < 0) {
            if (errno == EINTR) {
                continue;  // Interrupted, retry
            }
            throw SendError("Send failed: " + std::string(strerror(errno)));
        }

        if (sent == 0) {
            throw SendError("Connection closed by peer");
        }

        total_sent += sent;
    }
}

void TCPConnection::receive_all(uint8_t* buffer, size_t size) {
    size_t total_received = 0;

    while (total_received < size) {
        ssize_t received = recv(socket_fd_, buffer + total_received,
                               size - total_received, 0);

        if (received < 0) {
            if (errno == EINTR) {
                continue;  // Interrupted, retry
            }
            throw ReceiveError("Receive failed: " + std::string(strerror(errno)));
        }

        if (received == 0) {
            throw ReceiveError("Connection closed by peer (expected " +
                              std::to_string(size) + " bytes, got " +
                              std::to_string(total_received) + ")");
        }

        total_received += received;
    }
}

// ============================================================================
// Local Connection Pair (for testing)
// ============================================================================

LocalConnectionPair create_local_connection_pair() {
    // Use Unix domain sockets or loopback for local testing
    // Simplified: use TCP on loopback

    const uint16_t port = 50000;  // Use fixed port for testing

    LocalConnectionPair pair;
    pair.party0 = std::make_unique<TCPConnection>();
    pair.party1 = std::make_unique<TCPConnection>();

    // Start server in background thread would be needed
    // For now, throw - user should manually setup

    throw NetworkError("create_local_connection_pair not implemented - use manual setup");
}

} // namespace silentmpc::network
