#pragma once

#include <string>
#include <vector>
#include <span>
#include <cstdint>
#include <stdexcept>
#include <memory>

namespace silentmpc::network {

// ============================================================================
// Network Exceptions
// ============================================================================

class NetworkError : public std::runtime_error {
public:
    explicit NetworkError(const std::string& msg) : std::runtime_error(msg) {}
};

class ConnectionError : public NetworkError {
public:
    explicit ConnectionError(const std::string& msg) : NetworkError(msg) {}
};

class SendError : public NetworkError {
public:
    explicit SendError(const std::string& msg) : NetworkError(msg) {}
};

class ReceiveError : public NetworkError {
public:
    explicit ReceiveError(const std::string& msg) : NetworkError(msg) {}
};

// ============================================================================
// TCP Connection (Blocking I/O)
// ============================================================================

/// TCP socket wrapper for 2-party communication
class TCPConnection {
private:
    int socket_fd_;
    std::string peer_address_;
    uint16_t peer_port_;
    bool connected_;

    // Statistics
    size_t bytes_sent_;
    size_t bytes_received_;

public:
    TCPConnection();
    ~TCPConnection();

    // No copy
    TCPConnection(const TCPConnection&) = delete;
    TCPConnection& operator=(const TCPConnection&) = delete;

    // Move ok
    TCPConnection(TCPConnection&& other) noexcept;
    TCPConnection& operator=(TCPConnection&& other) noexcept;

    /// Connect to remote peer (client mode)
    void connect(const std::string& host, uint16_t port);

    /// Listen and accept one connection (server mode)
    void listen_and_accept(uint16_t port, int backlog = 1);

    /// Send data (blocking)
    /// Throws SendError if fails
    void send(std::span<const uint8_t> data);

    /// Send data with length prefix (4-byte big-endian)
    void send_with_length(std::span<const uint8_t> data);

    /// Receive exact amount of data (blocking)
    /// Throws ReceiveError if connection closes or fails
    std::vector<uint8_t> receive(size_t size);

    /// Receive data with length prefix (4-byte big-endian)
    std::vector<uint8_t> receive_with_length();

    /// Close connection
    void close();

    // Status
    bool is_connected() const { return connected_; }
    const std::string& peer_address() const { return peer_address_; }
    uint16_t peer_port() const { return peer_port_; }

    // Statistics
    size_t bytes_sent() const { return bytes_sent_; }
    size_t bytes_received() const { return bytes_received_; }
    void reset_statistics() { bytes_sent_ = 0; bytes_received_ = 0; }

private:
    /// Helper: send all bytes
    void send_all(const uint8_t* data, size_t size);

    /// Helper: receive all bytes
    void receive_all(uint8_t* buffer, size_t size);
};

// ============================================================================
// Connection Factory
// ============================================================================

/// Create connection pair for local testing
struct LocalConnectionPair {
    std::unique_ptr<TCPConnection> party0;
    std::unique_ptr<TCPConnection> party1;
};

/// Create two connected sockets using Unix domain sockets for testing
LocalConnectionPair create_local_connection_pair();

// ============================================================================
// Network Utilities
// ============================================================================

/// Serialize uint32_t to big-endian
inline void write_uint32_be(uint8_t* buffer, uint32_t value) {
    buffer[0] = static_cast<uint8_t>((value >> 24) & 0xFF);
    buffer[1] = static_cast<uint8_t>((value >> 16) & 0xFF);
    buffer[2] = static_cast<uint8_t>((value >> 8) & 0xFF);
    buffer[3] = static_cast<uint8_t>(value & 0xFF);
}

/// Deserialize uint32_t from big-endian
inline uint32_t read_uint32_be(const uint8_t* buffer) {
    return (static_cast<uint32_t>(buffer[0]) << 24) |
           (static_cast<uint32_t>(buffer[1]) << 16) |
           (static_cast<uint32_t>(buffer[2]) << 8) |
           static_cast<uint32_t>(buffer[3]);
}

} // namespace silentmpc::network
