/**
 * Table 4: Tile-size Sensitivity Study - Real Network Measurement
 *
 * Measures setup cost for different tile sizes: B = 2^12, 2^14, 2^16
 * - OT and DPF backends
 * - w = 32 (fixed)
 * - Real TCP + tc netem
 */

#include "silentmpc/selective_delivery/point_share.hpp"
#include "silentmpc/algebra/gf2.hpp"
#include "silentmpc/core/types.hpp"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>

using namespace std;
using namespace std::chrono;
using namespace silentmpc;

// ============================================================================
// Network Helpers
// ============================================================================

void send_all(int sock, const void* data, size_t size) {
    size_t sent = 0;
    const char* ptr = static_cast<const char*>(data);
    while (sent < size) {
        ssize_t n = send(sock, ptr + sent, size - sent, 0);
        if (n <= 0) throw runtime_error("send failed");
        sent += n;
    }
}

void recv_all(int sock, void* data, size_t size) {
    size_t received = 0;
    char* ptr = static_cast<char*>(data);
    while (received < size) {
        ssize_t n = recv(sock, ptr + received, size - received, 0);
        if (n <= 0) throw runtime_error("recv failed");
        received += n;
    }
}

// ============================================================================
// Setup Data Generation
// ============================================================================

struct SetupStates {
    vector<uint8_t> sender_state;
    vector<uint8_t> receiver_state;
    size_t communication_bytes;
};

SetupStates generate_setup_states(selective_delivery::BackendPolicy policy, size_t tile_size) {
    SetupStates result;

    auto backend = selective_delivery::BackendFactory<algebra::GF2>::create(policy);

    // Parameters: w=32, tile_size=variable
    vector<size_t> indices;
    size_t w = 32;

    for (size_t i = 0; i < w; ++i) {
        indices.push_back(i * (tile_size / w));
    }

    SessionId sid{};
    for (auto& b : sid) b = rand() % 256;

    // Generate setup states
    auto setup_result = backend->setup(indices, tile_size, sid, 0, 0);

    result.sender_state = setup_result.sender_state;
    result.receiver_state = setup_result.receiver_state;
    result.communication_bytes = setup_result.communication_bytes;

    return result;
}

// ============================================================================
// Server (receives setup states)
// ============================================================================

void run_server() {
    cout << "Starting server on port 9999...\n";

    int server_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (server_sock < 0) throw runtime_error("socket creation failed");

    int opt = 1;
    setsockopt(server_sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(9999);

    if (bind(server_sock, (sockaddr*)&addr, sizeof(addr)) < 0) {
        close(server_sock);
        throw runtime_error("bind failed");
    }

    if (listen(server_sock, 1) < 0) {
        close(server_sock);
        throw runtime_error("listen failed");
    }

    cout << "Server listening...\n";

    sockaddr_in client_addr{};
    socklen_t client_len = sizeof(client_addr);
    int client_sock = accept(server_sock, (sockaddr*)&client_addr, &client_len);

    if (client_sock < 0) {
        close(server_sock);
        throw runtime_error("accept failed");
    }

    int flag = 1;
    setsockopt(client_sock, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

    cout << "Client connected!\n\n";

    // Receive tile_size first
    uint32_t tile_size_bits;
    recv_all(client_sock, &tile_size_bits, sizeof(tile_size_bits));
    size_t tile_size = 1ULL << tile_size_bits;

    cout << "Tile size: 2^" << tile_size_bits << " = " << tile_size << "\n\n";

    // Test OT backend
    {
        uint32_t sender_size;
        recv_all(client_sock, &sender_size, sizeof(sender_size));

        vector<uint8_t> sender_state(sender_size);
        auto start = high_resolution_clock::now();
        recv_all(client_sock, sender_state.data(), sender_size);
        auto mid = high_resolution_clock::now();

        uint32_t receiver_size;
        recv_all(client_sock, &receiver_size, sizeof(receiver_size));

        vector<uint8_t> receiver_state(receiver_size);
        recv_all(client_sock, receiver_state.data(), receiver_size);
        auto end = high_resolution_clock::now();

        double elapsed1 = duration<double, milli>(mid - start).count();
        double elapsed2 = duration<double, milli>(end - mid).count();
        double total = duration<double, milli>(end - start).count();

        uint8_t ack = 1;
        send_all(client_sock, &ack, 1);

        cout << "OT Backend:\n";
        cout << "  Sender:   " << sender_size << " bytes - " << fixed << setprecision(2) << elapsed1 << " ms\n";
        cout << "  Receiver: " << receiver_size << " bytes - " << fixed << setprecision(2) << elapsed2 << " ms\n";
        cout << "  Total:    " << (sender_size + receiver_size) << " bytes ("
             << fixed << setprecision(2) << ((sender_size + receiver_size) / 1024.0) << " KiB) - "
             << fixed << setprecision(2) << total << " ms\n\n";
    }

    // Test DPF backend
    {
        uint32_t sender_size;
        recv_all(client_sock, &sender_size, sizeof(sender_size));

        vector<uint8_t> sender_state(sender_size);
        auto start = high_resolution_clock::now();
        recv_all(client_sock, sender_state.data(), sender_size);
        auto mid = high_resolution_clock::now();

        uint32_t receiver_size;
        recv_all(client_sock, &receiver_size, sizeof(receiver_size));

        vector<uint8_t> receiver_state(receiver_size);
        recv_all(client_sock, receiver_state.data(), receiver_size);
        auto end = high_resolution_clock::now();

        double elapsed1 = duration<double, milli>(mid - start).count();
        double elapsed2 = duration<double, milli>(end - mid).count();
        double total = duration<double, milli>(end - start).count();

        uint8_t ack = 1;
        send_all(client_sock, &ack, 1);

        cout << "DPF Backend:\n";
        cout << "  Sender:   " << sender_size << " bytes - " << fixed << setprecision(2) << elapsed1 << " ms\n";
        cout << "  Receiver: " << receiver_size << " bytes - " << fixed << setprecision(2) << elapsed2 << " ms\n";
        cout << "  Total:    " << (sender_size + receiver_size) << " bytes ("
             << fixed << setprecision(2) << ((sender_size + receiver_size) / 1024.0) << " KiB) - "
             << fixed << setprecision(2) << total << " ms\n\n";
    }

    close(client_sock);
    close(server_sock);
}

// ============================================================================
// Client (sends setup states)
// ============================================================================

void run_client(size_t tile_size_bits) {
    size_t tile_size = 1ULL << tile_size_bits;

    cout << "Tile size: 2^" << tile_size_bits << " = " << tile_size << "\n";
    cout << "Generating setup states...\n";

    // Pre-generate states
    auto ot_states = generate_setup_states(selective_delivery::BackendPolicy::OT, tile_size);
    cout << "  OT Backend:\n";
    cout << "    Total: " << (ot_states.sender_state.size() + ot_states.receiver_state.size())
         << " bytes (" << fixed << setprecision(2)
         << ((ot_states.sender_state.size() + ot_states.receiver_state.size()) / 1024.0) << " KiB)\n";
    cout << "    Comm:  " << ot_states.communication_bytes << " bytes ("
         << fixed << setprecision(2) << (ot_states.communication_bytes / 1024.0) << " KiB)\n";

    auto dpf_states = generate_setup_states(selective_delivery::BackendPolicy::DPF, tile_size);
    cout << "  DPF Backend:\n";
    cout << "    Total: " << (dpf_states.sender_state.size() + dpf_states.receiver_state.size())
         << " bytes (" << fixed << setprecision(2)
         << ((dpf_states.sender_state.size() + dpf_states.receiver_state.size()) / 1024.0) << " KiB)\n";
    cout << "    Comm:  " << dpf_states.communication_bytes << " bytes ("
         << fixed << setprecision(2) << (dpf_states.communication_bytes / 1024.0) << " KiB)\n\n";

    cout << "Connecting to server...\n";

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) throw runtime_error("socket creation failed");

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(9999);
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);

    if (connect(sock, (sockaddr*)&addr, sizeof(addr)) < 0) {
        close(sock);
        throw runtime_error("connect failed");
    }

    int flag = 1;
    setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

    cout << "Connected!\n\n";

    // Send tile_size
    uint32_t tile_size_bits_u32 = tile_size_bits;
    send_all(sock, &tile_size_bits_u32, sizeof(tile_size_bits_u32));

    cout << "╔═══════════════════════════════════════════════════════════╗\n";
    cout << "║  Table 4: Tile-size Study (B = 2^" << tile_size_bits << ")                   ║\n";
    cout << "╚═══════════════════════════════════════════════════════════╝\n\n";

    cout << setw(12) << "Backend"
         << setw(18) << "Comm (KiB)"
         << setw(18) << "Transfer (ms)"
         << "\n";
    cout << string(48, '-') << "\n";

    // Test 1: OT Backend
    {
        uint32_t sender_size = ot_states.sender_state.size();
        uint32_t receiver_size = ot_states.receiver_state.size();

        auto start = high_resolution_clock::now();

        send_all(sock, &sender_size, sizeof(sender_size));
        send_all(sock, ot_states.sender_state.data(), sender_size);

        send_all(sock, &receiver_size, sizeof(receiver_size));
        send_all(sock, ot_states.receiver_state.data(), receiver_size);

        uint8_t ack;
        recv_all(sock, &ack, 1);
        auto end = high_resolution_clock::now();

        double elapsed = duration<double, milli>(end - start).count();

        cout << setw(12) << "OT"
             << setw(18) << fixed << setprecision(2) << (ot_states.communication_bytes / 1024.0)
             << setw(18) << fixed << setprecision(2) << elapsed
             << "\n";
    }

    // Test 2: DPF Backend
    {
        uint32_t sender_size = dpf_states.sender_state.size();
        uint32_t receiver_size = dpf_states.receiver_state.size();

        auto start = high_resolution_clock::now();

        send_all(sock, &sender_size, sizeof(sender_size));
        send_all(sock, dpf_states.sender_state.data(), sender_size);

        send_all(sock, &receiver_size, sizeof(receiver_size));
        send_all(sock, dpf_states.receiver_state.data(), receiver_size);

        uint8_t ack;
        recv_all(sock, &ack, 1);
        auto end = high_resolution_clock::now();

        double elapsed = duration<double, milli>(end - start).count();

        cout << setw(12) << "DPF"
             << setw(18) << fixed << setprecision(2) << (dpf_states.communication_bytes / 1024.0)
             << setw(18) << fixed << setprecision(2) << elapsed
             << "\n";
    }

    cout << string(48, '-') << "\n\n";

    close(sock);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    try {
        if (argc < 2) {
            cerr << "Usage:\n";
            cerr << "  " << argv[0] << " server\n";
            cerr << "  " << argv[0] << " client <tile_size_bits>\n";
            cerr << "\nExample:\n";
            cerr << "  " << argv[0] << " server &\n";
            cerr << "  " << argv[0] << " client 12  # for B = 2^12\n";
            return 1;
        }

        string mode = argv[1];

        if (mode == "server") {
            run_server();
        } else if (mode == "client") {
            if (argc < 3) {
                cerr << "Error: client mode requires tile_size_bits argument\n";
                cerr << "Example: " << argv[0] << " client 12\n";
                return 1;
            }
            size_t tile_size_bits = stoull(argv[2]);
            run_client(tile_size_bits);
        } else {
            cerr << "Invalid mode. Use 'server' or 'client'\n";
            return 1;
        }

        return 0;
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
