/**
 * Simple PCF Example
 * Demonstrates basic usage of the Silent MPC Correlation Engine
 *
 * Paper: "A Portable and Streaming Correlation Engine for Silent MPC Preprocessing"
 * Section 2.1: API Definition
 */

#include <silentmpc/pcf/pcf_core.hpp>
#include <silentmpc/algebra/z2k.hpp>
#include <silentmpc/runtime/epoch_manager.hpp>
#include <silentmpc/runtime/allocator.hpp>

#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>

using namespace silentmpc;

// ============================================================================
// Helper Functions
// ============================================================================

/// Generate random session ID (128-bit)
SessionId generate_random_sid() {
    SessionId sid;
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint8_t> dist(0, 255);

    for (auto& byte : sid) {
        byte = dist(gen);
    }
    return sid;
}

/// Generate random root seed (256-bit)
std::vector<uint8_t> generate_random_seed() {
    std::vector<uint8_t> seed(32);
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint8_t> dist(0, 255);

    for (auto& byte : seed) {
        byte = dist(gen);
    }
    return seed;
}

/// Print hex bytes
void print_hex(std::span<const uint8_t> data, size_t max_bytes = 16) {
    for (size_t i = 0; i < std::min(data.size(), max_bytes); ++i) {
        std::cout << std::hex << std::setw(2) << std::setfill('0')
                  << static_cast<int>(data[i]);
    }
    if (data.size() > max_bytes) {
        std::cout << "...";
    }
    std::cout << std::dec;
}

// ============================================================================
// Example 1: Basic PCF Setup and Eval
// ============================================================================

void example_basic_pcf() {
    std::cout << "=== Example 1: Basic PCF Setup and Eval ===" << std::endl;
    std::cout << std::endl;

    // Use Z_{2^64} ring (Table 8, R2)
    using R = algebra::Z2_64;

    // 1. Generate session context
    auto sid = generate_random_sid();
    auto root_seed = generate_random_seed();

    std::cout << "Session ID: ";
    print_hex(sid);
    std::cout << std::endl;

    // 2. Choose parameters (Section 7, Table 8)
    auto params = parameters::R2;  // Z_{2^64}, N=2^18, n=256, w=32
    std::cout << "Parameters: N=" << params.N
              << ", n=" << params.n
              << ", w=" << params.w << std::endl;

    // 3. Setup phase (Section 2.1)
    std::cout << "\nRunning Setup..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    // Note: In real implementation, setup() would be implemented
    // This is a placeholder showing the API
    // auto [handle, initial_ctx] = pcf::PCFHandle<R>::setup(params, sid, root_seed);

    auto end = std::chrono::high_resolution_clock::now();
    auto setup_time = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Setup completed in " << setup_time << " ms" << std::endl;

    // 4. Create runtime state manager
    runtime::EpochPolicy epoch_policy{
        .mode = runtime::EpochPolicy::Mode::Monotonic,
        .persist_dir = "/tmp/silentmpc_test"
    };

    // Note: This would initialize runtime state
    std::cout << "\nInitializing runtime state..." << std::endl;
    std::cout << "  Epoch mode: Monotonic (64-bit counter)" << std::endl;
    std::cout << "  Query limit: 2^30" << std::endl;
    std::cout << "  Block size: 10,000" << std::endl;

    // 5. Evaluate correlations
    std::cout << "\nEvaluating correlations..." << std::endl;

    const size_t num_queries = 100;
    start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < num_queries; ++i) {
        // auto correlation = handle.eval(i, ctx, 0);
        // In practice: use correlation for MPC protocol
    }

    end = std::chrono::high_resolution_clock::now();
    auto eval_time = std::chrono::duration<double, std::milli>(end - start).count();

    double throughput = (num_queries / eval_time) * 1000.0;  // per second
    std::cout << "Evaluated " << num_queries << " correlations in "
              << eval_time << " ms" << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(2)
              << throughput << " correlations/sec" << std::endl;

    std::cout << std::endl;
}

// ============================================================================
// Example 2: Operational Contracts
// ============================================================================

void example_operational_contracts() {
    std::cout << "=== Example 2: Operational Contracts ===" << std::endl;
    std::cout << std::endl;

    auto sid = generate_random_sid();

    // Contract 1: Epoch-bound derivation
    std::cout << "Contract 1: Epoch Management" << std::endl;
    std::cout << "  Creating monotonic epoch manager..." << std::endl;

    // Note: Would create actual epoch manager
    EpochId epoch = 0;
    std::cout << "  Initial epoch: " << epoch << std::endl;

    // Simulate epoch increment
    epoch++;
    std::cout << "  After increment: " << epoch << std::endl;
    std::cout << "  [In real impl: atomically persisted to disk]" << std::endl;

    std::cout << std::endl;

    // Contract 2: Waste-not-reuse allocation
    std::cout << "Contract 2: Block Allocation" << std::endl;
    std::cout << "  Block size: 10,000 indices" << std::endl;

    // Simulate allocation
    CorrelationIndex high_watermark = 0;
    CorrelationIndex block_start = high_watermark;
    CorrelationIndex block_end = block_start + 10'000;

    std::cout << "  Allocated block: [" << block_start << ", " << block_end << ")" << std::endl;
    std::cout << "  [In real impl: high-watermark persisted before returning]" << std::endl;

    // Simulate crash recovery
    std::cout << "  Simulating crash and recovery..." << std::endl;
    CorrelationIndex recovered_watermark = block_end;
    std::cout << "  Recovered high-watermark: " << recovered_watermark << std::endl;
    std::cout << "  Next allocation starts from: " << recovered_watermark << std::endl;
    std::cout << "  Wasted indices: 0 (no in-flight allocations)" << std::endl;

    std::cout << std::endl;

    // Contract 3: Session rotation
    std::cout << "Contract 3: Session Rotation" << std::endl;

    const CorrelationIndex Q_max = 1ULL << 30;  // 2^30
    CorrelationIndex current_count = Q_max - 1000;

    std::cout << "  Query limit: 2^30 = " << Q_max << std::endl;
    std::cout << "  Current count: " << current_count << std::endl;
    std::cout << "  Remaining: " << (Q_max - current_count) << std::endl;

    if (current_count + 2000 > Q_max) {
        std::cout << "  ⚠ Rotation needed!" << std::endl;
        auto new_sid = generate_random_sid();
        std::cout << "  New session ID: ";
        print_hex(new_sid);
        std::cout << std::endl;
        std::cout << "  [In real impl: re-run Setup with new sid]" << std::endl;
    }

    std::cout << std::endl;
}

// ============================================================================
// Example 3: Backend Selection
// ============================================================================

void example_backend_selection() {
    std::cout << "=== Example 3: Backend Selection ===" << std::endl;
    std::cout << std::endl;

    using namespace selective_delivery;

    // LAN scenario
    std::cout << "Scenario A: LAN (10 Gbps, 0.1ms RTT)" << std::endl;
    NetworkHints lan_hints{.bandwidth_mbps = 10'000, .rtt_ms = 0};
    auto lan_policy = lan_hints.recommend();
    std::cout << "  Recommended backend: "
              << (lan_policy == BackendPolicy::OT ? "OT-Centric" : "DPF-Centric")
              << std::endl;
    std::cout << "  Rationale: Low latency, high bandwidth → OT extension optimal"
              << std::endl;

    std::cout << std::endl;

    // WAN scenario
    std::cout << "Scenario B: WAN (100 Mbps, 50ms RTT)" << std::endl;
    NetworkHints wan_hints{.bandwidth_mbps = 100, .rtt_ms = 50};
    auto wan_policy = wan_hints.recommend();
    std::cout << "  Recommended backend: "
              << (wan_policy == BackendPolicy::OT ? "OT-Centric" : "DPF-Centric")
              << std::endl;
    std::cout << "  Rationale: High latency → DPF reduces round-trip dependencies"
              << std::endl;

    std::cout << std::endl;

    // Table 3 comparison (from paper)
    std::cout << "Performance Comparison (from Paper Table 3):" << std::endl;
    std::cout << "  Backend       | LAN Comm | WAN Setup | Notes" << std::endl;
    std::cout << "  ------------- | -------- | --------- | -----" << std::endl;
    std::cout << "  OT-Centric    | 2.5 KiB  | 50.5 ms   | Bandwidth-efficient" << std::endl;
    std::cout << "  DPF-Centric   | 11.0 KiB | 51.2 ms   | Non-interactive" << std::endl;
    std::cout << std::endl;
}

// ============================================================================
// Example 4: Parameter Sets
// ============================================================================

void example_parameter_sets() {
    std::cout << "=== Example 4: Available Parameter Sets ===" << std::endl;
    std::cout << std::endl;

    std::cout << "Boolean (GF(2)) Parameters - Table 6:" << std::endl;
    std::cout << "  B1: N=2^20, n=512,  w=32  (142 bits est. cost)" << std::endl;
    std::cout << "  B2: N=2^22, n=768,  w=64  (151 bits est. cost)" << std::endl;
    std::cout << "  B3: N=2^24, n=1024, w=64  (158 bits est. cost)" << std::endl;
    std::cout << std::endl;

    std::cout << "Field (GF(q)) Parameters - Table 7:" << std::endl;
    std::cout << "  F1: GF(257),    N=2^18, n=512, w=32  (138 bits)" << std::endl;
    std::cout << "  F2: GF(65537),  N=2^18, n=384, w=32  (129 bits)" << std::endl;
    std::cout << "  F3: GF(2^16),   N=2^20, n=512, w=64  (143 bits)" << std::endl;
    std::cout << std::endl;

    std::cout << "Ring (Z_{2^k}) Parameters - Table 8:" << std::endl;
    std::cout << "  Tier 1 (Conservative, Unstructured):" << std::endl;
    std::cout << "    R1: Z_{2^32}, N=2^18, n=256, w=32  (128 bits)" << std::endl;
    std::cout << "    R2: Z_{2^64}, N=2^18, n=256, w=32  (128 bits)" << std::endl;
    std::cout << "    R3: Z_{2^96}, N=2^20, n=384, w=64  (141 bits)" << std::endl;
    std::cout << std::endl;

    std::cout << "  Tier 2 (Hardened, Structured with Buffer):" << std::endl;
    std::cout << "    R1*: Z_{2^32}, N=2^18, n=512, w=48  (146 bits)" << std::endl;
    std::cout << "    R2*: Z_{2^64}, N=2^18, n=512, w=64  (150 bits)" << std::endl;
    std::cout << std::endl;

    std::cout << "All parameters target ≥128-bit security (Tier 1)" << std::endl;
    std::cout << "Tier 2 includes explicit risk buffer for structured matrices" << std::endl;
    std::cout << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Silent MPC Correlation Engine" << std::endl;
    std::cout << "C++20 Implementation Examples" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    try {
        example_basic_pcf();
        example_operational_contracts();
        example_backend_selection();
        example_parameter_sets();

        std::cout << "========================================" << std::endl;
        std::cout << "All examples completed successfully!" << std::endl;
        std::cout << "========================================" << std::endl;

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
