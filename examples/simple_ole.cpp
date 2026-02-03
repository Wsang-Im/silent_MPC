/**
 * Simple OLE (Oblivious Linear Evaluation) Example
 * Demonstrates basic usage of Silent MPC library
 */

#include "silentmpc/pcf/pcf_core.hpp"
#include "silentmpc/algebra/z2k.hpp"
#include "silentmpc/algebra/gf2.hpp"
#include "silentmpc/core/types.hpp"
#include <iostream>
#include <iomanip>
#include <random>

using namespace silentmpc;
using namespace silentmpc::algebra;
using namespace silentmpc::pcf;

void print_header(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(60, '=') << "\n";
}

void example_gf2() {
    print_header("Example 1: GF(2) Operations");

    // Create GF2 elements
    GF2 a = GF2::from_uint64(1);
    GF2 b = GF2::from_uint64(0);

    std::cout << "a = " << a.to_uint64() << "\n";
    std::cout << "b = " << b.to_uint64() << "\n";
    std::cout << "a + b = " << (a + b).to_uint64() << " (XOR)\n";
    std::cout << "a * b = " << (a * b).to_uint64() << " (AND)\n";

    // GF2 Vector operations
    GF2Vector vec(8);
    std::cout << "\nGF2Vector of size 8 created\n";
    std::cout << "Hamming weight: " << vec.hamming_weight() << "\n";
}

void example_z2k() {
    print_header("Example 2: Z_{2^32} Ring Operations");

    // Create Z2k elements
    Z2k<32> x = Z2k<32>::from_uint64(42);
    Z2k<32> y = Z2k<32>::from_uint64(17);

    std::cout << "x = " << x.to_uint64() << "\n";
    std::cout << "y = " << y.to_uint64() << "\n";
    std::cout << "x + y = " << (x + y).to_uint64() << "\n";
    std::cout << "x * y = " << (x * y).to_uint64() << "\n";
    std::cout << "x - y = " << (x - y).to_uint64() << "\n";
    std::cout << "-x = " << (-x).to_uint64() << "\n";
}

void example_pcf_parameters() {
    print_header("Example 3: PCF Parameter Sets (Table 6-8 from paper)");

    std::cout << "\nBoolean (GF(2)) Parameters:\n";
    std::cout << "  B1: N=" << parameters::B1.N << ", n=" << parameters::B1.n
              << ", w=" << parameters::B1.w << "\n";
    std::cout << "  B2: N=" << parameters::B2.N << ", n=" << parameters::B2.n
              << ", w=" << parameters::B2.w << "\n";
    std::cout << "  B3: N=" << parameters::B3.N << ", n=" << parameters::B3.n
              << ", w=" << parameters::B3.w << "\n";

    std::cout << "\nRing (Z_{2^k}) Parameters:\n";
    std::cout << "  R1: N=" << parameters::R1.N << ", n=" << parameters::R1.n
              << ", w=" << parameters::R1.w << "\n";
    std::cout << "  R2: N=" << parameters::R2.N << ", n=" << parameters::R2.n
              << ", w=" << parameters::R2.w << "\n";

    std::cout << "\nHardened Variants (Tier 2):\n";
    std::cout << "  R1*: N=" << parameters::R1_star.N << ", n=" << parameters::R1_star.n
              << ", w=" << parameters::R1_star.w << " (Structured matrix)\n";
}

void example_collision_probability() {
    print_header("Example 4: Collision Analysis (Section 3.2)");

    // Expected collision count: E[|C|] = w²/N
    struct TestCase {
        size_t N, w;
        const char* name;
    };

    TestCase cases[] = {
        {1ULL << 20, 32, "B1 (N=2^20, w=32)"},
        {1ULL << 22, 64, "B2 (N=2^22, w=64)"},
        {1ULL << 18, 32, "R1 (N=2^18, w=32)"},
    };

    for (const auto& tc : cases) {
        double expected = static_cast<double>(tc.w * tc.w) / tc.N;
        std::cout << tc.name << ":\n";
        std::cout << "  Expected collisions: " << std::fixed << std::setprecision(4)
                  << expected << "\n";
    }
}

void example_session_ids() {
    print_header("Example 5: Session ID Generation (Contract 0)");

    // Generate random session ID
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint8_t> dist(0, 255);

    SessionId sid;
    for (size_t i = 0; i < sid.size(); ++i) {
        sid[i] = dist(gen);
    }

    std::cout << "Session ID (128-bit): ";
    for (size_t i = 0; i < 8; ++i) {
        std::cout << std::hex << std::setw(2) << std::setfill('0')
                  << static_cast<int>(sid[i]);
    }
    std::cout << "...\n";

    std::cout << "\nEpoch examples:\n";
    EpochId epoch1 = 0;
    EpochId epoch2 = 1;
    std::cout << "  Initial epoch: " << std::dec << epoch1 << "\n";
    std::cout << "  Next epoch: " << epoch2 << "\n";
}

void example_derivation_context() {
    print_header("Example 6: Derivation Context (Section 5.3)");

    SessionId sid = {};
    DerivationContext ctx;
    ctx.sid = sid;
    ctx.epoch = 0;
    ctx.index = 42;
    ctx.party = 0;
    ctx.term = TermId::Term1_DenseDense;
    ctx.tile_id = 0;

    std::cout << "Derivation Context for PRF/KDF:\n";
    std::cout << "  Session ID: [16 bytes]\n";
    std::cout << "  Epoch: " << ctx.epoch << "\n";
    std::cout << "  Index: " << ctx.index << "\n";
    std::cout << "  Party: " << static_cast<int>(ctx.party) << "\n";
    std::cout << "  Term: " << static_cast<int>(ctx.term)
              << " (Term1_DenseDense)\n";
    std::cout << "  Tile ID: " << ctx.tile_id << "\n";

    std::cout << "\nThis provides domain separation for c=4 terms:\n";
    std::cout << "  Term 1: Dense-Dense (As0 ⊙ As1)\n";
    std::cout << "  Term 2: Sparse-Dense (e0 ⊙ As1)\n";
    std::cout << "  Term 3: Sparse-Dense (e1 ⊙ As0)\n";
    std::cout << "  Term 4: Sparse-Sparse (e0 ⊙ e1)\n";
}

int main() {
    try {
        std::cout << "\n";
        std::cout << "╔═══════════════════════════════════════════════════════════╗\n";
        std::cout << "║         Silent MPC Library - Usage Examples            ║\n";
        std::cout << "║  Portable and Streaming Pseudorandom Correlation        ║\n";
        std::cout << "║  Generator from Batch-OLE (ePrint 2025/119)            ║\n";
        std::cout << "╚═══════════════════════════════════════════════════════════╝\n";

        example_gf2();
        example_z2k();
        example_pcf_parameters();
        example_collision_probability();
        example_session_ids();
        example_derivation_context();

        print_header("Summary");
        std::cout << "✓ All examples completed successfully!\n";
        std::cout << "\nNext steps:\n";
        std::cout << "  1. Run tests: make test\n";
        std::cout << "  2. Run benchmarks: ./benchmarks/bench_algebra\n";
        std::cout << "  3. Check full PCF evaluation in test_pcf\n";
        std::cout << "\nLibrary features demonstrated:\n";
        std::cout << "  ✓ Ring operations (GF2, Z2k)\n";
        std::cout << "  ✓ PCF parameter sets from paper\n";
        std::cout << "  ✓ Collision analysis\n";
        std::cout << "  ✓ Session management\n";
        std::cout << "  ✓ Domain separation\n";
        std::cout << "\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
