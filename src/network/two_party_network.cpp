#include "silentmpc/network/two_party_network.hpp"
#include <iostream>
#include <iomanip>
#include <thread>

namespace silentmpc::network {

// ============================================================================
// TwoPartyNetwork Implementation
// ============================================================================

TwoPartyNetwork::TwoPartyNetwork(
    PartyId my_party,
    const SessionId& session_id,
    const std::string& peer_host,
    uint16_t peer_port
) : my_party_(my_party),
    session_id_(session_id),
    messages_sent_(0),
    messages_received_(0),
    total_send_time_(0),
    total_receive_time_(0) {

    setup_client(peer_host, peer_port);
}

TwoPartyNetwork::TwoPartyNetwork(
    PartyId my_party,
    const SessionId& session_id,
    uint16_t listen_port
) : my_party_(my_party),
    session_id_(session_id),
    messages_sent_(0),
    messages_received_(0),
    total_send_time_(0),
    total_receive_time_(0) {

    setup_server(listen_port);
}

TwoPartyNetwork::~TwoPartyNetwork() {
    if (connection_) {
        connection_->close();
    }
}

void TwoPartyNetwork::setup_client(const std::string& host, uint16_t port) {
    connection_ = std::make_unique<TCPConnection>();

    std::cout << "[Party " << static_cast<int>(my_party_) << "] "
              << "Connecting to " << host << ":" << port << "..." << std::endl;

    connection_->connect(host, port);

    std::cout << "[Party " << static_cast<int>(my_party_) << "] "
              << "Connected successfully" << std::endl;
}

void TwoPartyNetwork::setup_server(uint16_t port) {
    connection_ = std::make_unique<TCPConnection>();

    std::cout << "[Party " << static_cast<int>(my_party_) << "] "
              << "Listening on port " << port << "..." << std::endl;

    connection_->listen_and_accept(port);

    std::cout << "[Party " << static_cast<int>(my_party_) << "] "
              << "Connection accepted from "
              << connection_->peer_address() << ":"
              << connection_->peer_port() << std::endl;
}

void TwoPartyNetwork::handshake() {
    if (my_party_ == 0) {
        // Party 0: send first
        HandshakeMessage my_handshake;
        my_handshake.protocol_version = 1;
        my_handshake.party_id = my_party_;
        my_handshake.session_id = session_id_;

        send_message(my_handshake.to_message());

        // Then receive
        auto peer_msg = receive_message();
        auto peer_handshake = HandshakeMessage::from_message(peer_msg);

        if (peer_handshake.party_id != 1) {
            throw NetworkError("Peer party ID mismatch (expected 1, got " +
                              std::to_string(peer_handshake.party_id) + ")");
        }

        if (peer_handshake.session_id != session_id_) {
            throw NetworkError("Session ID mismatch during handshake");
        }

        std::cout << "[Party 0] Handshake complete with Party 1" << std::endl;

    } else {
        // Party 1: receive first
        auto peer_msg = receive_message();
        auto peer_handshake = HandshakeMessage::from_message(peer_msg);

        if (peer_handshake.party_id != 0) {
            throw NetworkError("Peer party ID mismatch (expected 0, got " +
                              std::to_string(peer_handshake.party_id) + ")");
        }

        if (peer_handshake.session_id != session_id_) {
            throw NetworkError("Session ID mismatch during handshake");
        }

        // Then send
        HandshakeMessage my_handshake;
        my_handshake.protocol_version = 1;
        my_handshake.party_id = my_party_;
        my_handshake.session_id = session_id_;

        send_message(my_handshake.to_message());

        std::cout << "[Party 1] Handshake complete with Party 0" << std::endl;
    }
}

void TwoPartyNetwork::send_message(const Message& msg) {
    auto start = std::chrono::high_resolution_clock::now();

    msg.send(*connection_);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    messages_sent_++;
    total_send_time_ += duration;
}

Message TwoPartyNetwork::receive_message() {
    auto start = std::chrono::high_resolution_clock::now();

    auto msg = Message::receive(*connection_);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    messages_received_++;
    total_receive_time_ += duration;

    return msg;
}

void TwoPartyNetwork::send_bytes(std::span<const uint8_t> data) {
    auto start = std::chrono::high_resolution_clock::now();

    connection_->send_with_length(data);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    messages_sent_++;
    total_send_time_ += duration;
}

std::vector<uint8_t> TwoPartyNetwork::receive_bytes(size_t size) {
    auto start = std::chrono::high_resolution_clock::now();

    auto data = connection_->receive(size);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    messages_received_++;
    total_receive_time_ += duration;

    return data;
}

void TwoPartyNetwork::close() {
    if (connection_) {
        connection_->close();
    }
}

void TwoPartyNetwork::print_statistics() const {
    std::cout << "\n╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         Network Statistics (Party " << static_cast<int>(my_party_) << ")                    ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n";

    std::cout << "Messages sent:     " << messages_sent_ << "\n";
    std::cout << "Messages received: " << messages_received_ << "\n";
    std::cout << "Bytes sent:        " << bytes_sent() << " (" << (bytes_sent() / 1024.0) << " KB)\n";
    std::cout << "Bytes received:    " << bytes_received() << " (" << (bytes_received() / 1024.0) << " KB)\n";

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Avg send time:     " << avg_send_time_ms() << " ms\n";
    std::cout << "Avg receive time:  " << avg_receive_time_ms() << " ms\n";
    std::cout << "════════════════════════════════════════════════════════════\n";
}

void TwoPartyNetwork::reset_statistics() {
    messages_sent_ = 0;
    messages_received_ = 0;
    total_send_time_ = std::chrono::microseconds(0);
    total_receive_time_ = std::chrono::microseconds(0);

    if (connection_) {
        connection_->reset_statistics();
    }
}

// ============================================================================
// Network Factory Implementation
// ============================================================================

NetworkPair create_localhost_network_pair(
    const SessionId& session_id,
    uint16_t port
) {
    NetworkPair pair;

    // Create party 1 (server) in a thread
    std::thread server_thread([&pair, session_id, port]() {
        pair.party1 = std::make_unique<TwoPartyNetwork>(1, session_id, port);
    });

    // Give server time to start listening
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Create party 0 (client)
    pair.party0 = std::make_unique<TwoPartyNetwork>(0, session_id, "127.0.0.1", port);

    // Wait for server setup to complete
    server_thread.join();

    // Perform handshakes
    std::thread handshake_thread([&pair]() {
        pair.party1->handshake();
    });

    pair.party0->handshake();
    handshake_thread.join();

    return pair;
}

} // namespace silentmpc::network
