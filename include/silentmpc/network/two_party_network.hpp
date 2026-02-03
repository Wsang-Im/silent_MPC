#pragma once

#include "silentmpc/network/connection.hpp"
#include "silentmpc/network/message.hpp"
#include "silentmpc/core/types.hpp"
#include <memory>
#include <chrono>

namespace silentmpc::network {

// ============================================================================
// Two-Party Network Manager
// ============================================================================

/// Manages communication between two parties (Party 0 and Party 1)
class TwoPartyNetwork {
private:
    PartyId my_party_;
    SessionId session_id_;
    std::unique_ptr<TCPConnection> connection_;

    // Statistics
    size_t messages_sent_;
    size_t messages_received_;
    std::chrono::microseconds total_send_time_;
    std::chrono::microseconds total_receive_time_;

public:
    /// Constructor for client (connects to server)
    /// my_party: 0 or 1
    /// peer_host: IP address of peer
    /// peer_port: Port number
    TwoPartyNetwork(
        PartyId my_party,
        const SessionId& session_id,
        const std::string& peer_host,
        uint16_t peer_port
    );

    /// Constructor for server (waits for connection)
    /// my_party: 0 or 1
    /// listen_port: Port to listen on
    TwoPartyNetwork(
        PartyId my_party,
        const SessionId& session_id,
        uint16_t listen_port
    );

    ~TwoPartyNetwork();

    // No copy
    TwoPartyNetwork(const TwoPartyNetwork&) = delete;
    TwoPartyNetwork& operator=(const TwoPartyNetwork&) = delete;

    /// Perform handshake with peer
    /// Verifies session ID matches
    void handshake();

    /// Send generic message
    void send_message(const Message& msg);

    /// Receive generic message
    Message receive_message();

    /// Send raw bytes
    void send_bytes(std::span<const uint8_t> data);

    /// Receive raw bytes
    std::vector<uint8_t> receive_bytes(size_t size);

    /// Close connection
    void close();

    // Accessors
    PartyId my_party() const { return my_party_; }
    PartyId peer_party() const { return my_party_ == 0 ? 1 : 0; }
    const SessionId& session_id() const { return session_id_; }
    bool is_connected() const { return connection_ && connection_->is_connected(); }

    // Statistics
    size_t messages_sent() const { return messages_sent_; }
    size_t messages_received() const { return messages_received_; }
    size_t bytes_sent() const { return connection_ ? connection_->bytes_sent() : 0; }
    size_t bytes_received() const { return connection_ ? connection_->bytes_received() : 0; }

    double avg_send_time_ms() const {
        return messages_sent_ > 0 ?
               (total_send_time_.count() / 1000.0 / messages_sent_) : 0.0;
    }

    double avg_receive_time_ms() const {
        return messages_received_ > 0 ?
               (total_receive_time_.count() / 1000.0 / messages_received_) : 0.0;
    }

    /// Print statistics
    void print_statistics() const;

    /// Reset statistics
    void reset_statistics();

private:
    void setup_client(const std::string& host, uint16_t port);
    void setup_server(uint16_t port);
};

// ============================================================================
// Network Factory
// ============================================================================

/// Create network pair for testing on localhost
struct NetworkPair {
    std::unique_ptr<TwoPartyNetwork> party0;
    std::unique_ptr<TwoPartyNetwork> party1;
};

/// Setup two parties on localhost for testing
/// Returns pair of connected networks
/// Note: This spawns a thread for server setup
NetworkPair create_localhost_network_pair(
    const SessionId& session_id,
    uint16_t port = 12345
);

} // namespace silentmpc::network
