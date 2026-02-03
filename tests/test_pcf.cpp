#include <gtest/gtest.h>
#include "silentmpc/pcf/pcf_core.hpp"
#include "silentmpc/algebra/gf2.hpp"
#include "silentmpc/algebra/z2k.hpp"

using namespace silentmpc;
using namespace silentmpc::pcf;
using namespace silentmpc::algebra;

// ============================================================================
// PCFGenerator Tests
// ============================================================================

TEST(PCFGeneratorTest, Initialization) {
    PCFGenerator<Z2k<64>> gen;

    SessionId sid;
    sid.fill(0x42);
    EpochId epoch = 1;
    CorrelationIndex idx = 0;
    PCFParameters params = PCFParameters::R2;  // Z_{2^64} params

    gen.initialize(sid, epoch, idx, params);

    EXPECT_EQ(gen.parameters().N, params.N);
    EXPECT_EQ(gen.parameters().tau, params.tau);
}

TEST(PCFGeneratorTest, DenseVectorGeneration) {
    PCFGenerator<Z2k<64>> gen;

    SessionId sid;
    sid.fill(0x11);
    gen.initialize(sid, 0, 0, PCFParameters::R2);

    // Generate dense vectors for both parties
    auto dense0 = gen.generate_dense_vector(0, 0);
    auto dense1 = gen.generate_dense_vector(1, 0);

    EXPECT_EQ(dense0.size(), PCFParameters::R2.N);
    EXPECT_EQ(dense1.size(), PCFParameters::R2.N);

    // Vectors should be different (with high probability)
    bool all_same = true;
    for (size_t i = 0; i < std::min(dense0.size(), dense1.size()); ++i) {
        if (dense0[i] != dense1[i]) {
            all_same = false;
            break;
        }
    }
    EXPECT_FALSE(all_same);
}

TEST(PCFGeneratorTest, SparseIndicesRegularity) {
    PCFGenerator<GF2> gen;

    SessionId sid;
    sid.fill(0x22);
    gen.initialize(sid, 0, 0, PCFParameters::B1);

    auto indices = gen.generate_sparse_indices(0, 0);

    EXPECT_EQ(indices.size(), PCFParameters::B1.tau);

    // Check all indices are within bounds
    for (size_t idx : indices) {
        EXPECT_LT(idx, PCFParameters::B1.N);
    }

    // Check regularity: indices should be roughly evenly distributed
    // Partition into buckets and verify each has at least one
    size_t bucket_count = 8;
    size_t bucket_size = PCFParameters::B1.N / bucket_count;
    std::vector<size_t> buckets(bucket_count, 0);

    for (size_t idx : indices) {
        size_t bucket = std::min(idx / bucket_size, bucket_count - 1);
        buckets[bucket]++;
    }

    // At least half the buckets should have elements (regularity check)
    size_t non_empty = 0;
    for (size_t count : buckets) {
        if (count > 0) non_empty++;
    }
    EXPECT_GE(non_empty, bucket_count / 2);
}

TEST(PCFGeneratorTest, SparseValuesGeneration) {
    PCFGenerator<Z2k<64>> gen;

    SessionId sid;
    sid.fill(0x33);
    gen.initialize(sid, 0, 0, PCFParameters::R2);

    auto values = gen.generate_sparse_values(0, 0);

    EXPECT_EQ(values.size(), PCFParameters::R2.tau);

    // Check values are non-trivial (not all zero)
    bool has_nonzero = false;
    for (const auto& val : values) {
        if (val != Z2k<64>::zero()) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

TEST(PCFGeneratorTest, DeterministicGeneration) {
    // Same seed should produce same output
    SessionId sid;
    sid.fill(0xAB);

    PCFGenerator<Z2k<64>> gen1, gen2;
    gen1.initialize(sid, 5, 100, PCFParameters::R2);
    gen2.initialize(sid, 5, 100, PCFParameters::R2);

    auto dense1 = gen1.generate_dense_vector(0, 0);
    auto dense2 = gen2.generate_dense_vector(0, 0);

    EXPECT_EQ(dense1, dense2);

    auto sparse1 = gen1.generate_sparse_indices(0, 0);
    auto sparse2 = gen2.generate_sparse_indices(0, 0);

    EXPECT_EQ(sparse1, sparse2);
}

TEST(PCFGeneratorTest, SecurityLevelComputation) {
    auto security = PCFGenerator<GF2>::compute_security_level(
        PCFParameters::B1,
        128  // 128-bit statistical security
    );

    EXPECT_EQ(security.computational_security, PCFParameters::B1.seed_size * 8);
    EXPECT_GT(security.noise_rate, 0.0);
    EXPECT_LT(security.noise_rate, 1.0);

    // Noise rate should be tau / N
    double expected_rate = static_cast<double>(PCFParameters::B1.tau) /
                          static_cast<double>(PCFParameters::B1.N);
    EXPECT_NEAR(security.noise_rate, expected_rate, 1e-6);
}

// ============================================================================
// FourTermComputer Tests
// ============================================================================

TEST(FourTermComputerTest, BasicComputation) {
    using R = Z2k<64>;

    std::vector<R> dense0 = {
        R::from_uint64(1), R::from_uint64(2),
        R::from_uint64(3), R::from_uint64(4)
    };

    std::vector<R> dense1 = {
        R::from_uint64(5), R::from_uint64(6),
        R::from_uint64(7), R::from_uint64(8)
    };

    std::vector<size_t> sparse0_idx = {0, 2};
    std::vector<R> sparse0_val = {R::from_uint64(10), R::from_uint64(20)};

    std::vector<size_t> sparse1_idx = {1, 2};
    std::vector<R> sparse1_val = {R::from_uint64(30), R::from_uint64(40)};

    auto result = FourTermComputer<R>::compute(
        dense0, dense1, sparse0_idx, sparse0_val, sparse1_idx, sparse1_val
    );

    // Term 1: element-wise product of dense vectors
    ASSERT_EQ(result.term1_dense_dense.size(), 4UL);
    EXPECT_EQ(result.term1_dense_dense[0], R::from_uint64(1 * 5));
    EXPECT_EQ(result.term1_dense_dense[1], R::from_uint64(2 * 6));
    EXPECT_EQ(result.term1_dense_dense[2], R::from_uint64(3 * 7));
    EXPECT_EQ(result.term1_dense_dense[3], R::from_uint64(4 * 8));

    // Term 2: sparse0[i] * dense1[sparse0_idx[i]]
    ASSERT_EQ(result.term2_sparse_dense.size(), 2UL);
    EXPECT_EQ(result.term2_sparse_dense[0], R::from_uint64(10 * 5));  // idx 0
    EXPECT_EQ(result.term2_sparse_dense[1], R::from_uint64(20 * 7));  // idx 2

    // Term 3: sparse1[i] * dense0[sparse1_idx[i]]
    ASSERT_EQ(result.term3_sparse_dense.size(), 2UL);
    EXPECT_EQ(result.term3_sparse_dense[0], R::from_uint64(30 * 2));  // idx 1
    EXPECT_EQ(result.term3_sparse_dense[1], R::from_uint64(40 * 3));  // idx 2

    // Term 4: collision at index 2
    ASSERT_GT(result.term4_sparse_sparse.size(), 0UL);
    // sparse0[1] * sparse1[1] at index 2
    EXPECT_EQ(result.term4_sparse_sparse[1], R::from_uint64(20 * 40));
}

TEST(FourTermComputerTest, NoCollisions) {
    using R = Z2k<32>;

    std::vector<R> dense0(8, R::one());
    std::vector<R> dense1(8, R::one());

    std::vector<size_t> sparse0_idx = {0, 2, 4};
    std::vector<R> sparse0_val(3, R::from_uint64(10));

    std::vector<size_t> sparse1_idx = {1, 3, 5};
    std::vector<R> sparse1_val(3, R::from_uint64(20));

    auto result = FourTermComputer<R>::compute(
        dense0, dense1, sparse0_idx, sparse0_val, sparse1_idx, sparse1_val
    );

    // Term 4 should have all zeros (no collisions)
    for (const auto& val : result.term4_sparse_sparse) {
        EXPECT_EQ(val, R::zero());
    }
}

TEST(FourTermComputerTest, GF2Computation) {
    std::vector<GF2> dense0 = {GF2(true), GF2(false), GF2(true), GF2(true)};
    std::vector<GF2> dense1 = {GF2(true), GF2(true), GF2(false), GF2(true)};

    std::vector<size_t> sparse0_idx = {0, 2};
    std::vector<GF2> sparse0_val = {GF2(true), GF2(true)};

    std::vector<size_t> sparse1_idx = {1};
    std::vector<GF2> sparse1_val = {GF2(true)};

    auto result = FourTermComputer<GF2>::compute(
        dense0, dense1, sparse0_idx, sparse0_val, sparse1_idx, sparse1_val
    );

    // Term 1: AND of dense vectors
    ASSERT_EQ(result.term1_dense_dense.size(), 4UL);
    EXPECT_EQ(result.term1_dense_dense[0].value(), true);   // 1 AND 1
    EXPECT_EQ(result.term1_dense_dense[1].value(), false);  // 0 AND 1
    EXPECT_EQ(result.term1_dense_dense[2].value(), false);  // 1 AND 0
    EXPECT_EQ(result.term1_dense_dense[3].value(), true);   // 1 AND 1
}

TEST(FourTermComputerTest, TiledComputation) {
    PCFGenerator<Z2k<64>> gen;

    SessionId sid;
    sid.fill(0x99);
    gen.initialize(sid, 0, 0, PCFParameters::R2);

    // Use small tile size for testing
    size_t tile_size = 4096;

    auto result = FourTermComputer<Z2k<64>>::compute_tiled(gen, tile_size);

    // Should have results for all N elements
    EXPECT_EQ(result.term1_dense_dense.size(), PCFParameters::R2.N);

    // Sparse terms should have tau elements per party
    EXPECT_EQ(result.term2_sparse_dense.size(), PCFParameters::R2.tau);
    EXPECT_EQ(result.term3_sparse_dense.size(), PCFParameters::R2.tau);
}

// ============================================================================
// OLE Reconstruction Tests
// ============================================================================

TEST(OLETest, CorrectReconstruction) {
    // Test that four-term decomposition correctly computes OLE
    // c = ab where parties hold shares (a0, a1) and (b0, b1)

    using R = Z2k<64>;

    // Secret values
    R a = R::from_uint64(123);
    R b = R::from_uint64(456);
    R expected = a * b;  // 123 * 456

    // Simulate shares (simplified - not true secret sharing)
    R a0 = R::from_uint64(50);
    R a1 = a - a0;  // a = a0 + a1

    R b0 = R::from_uint64(200);
    R b1 = b - b0;  // b = b0 + b1

    // Compute expansion: (a0 + a1)(b0 + b1) = a0*b0 + a0*b1 + a1*b0 + a1*b1
    R term1 = a0 * b0;
    R term2 = a0 * b1;
    R term3 = a1 * b0;
    R term4 = a1 * b1;

    R reconstructed = term1 + term2 + term3 + term4;

    EXPECT_EQ(reconstructed, expected);
}

// ============================================================================
// Parameter Set Tests
// ============================================================================

TEST(ParametersTest, B1Parameters) {
    // From Table 6: GF(2) parameters
    const auto& params = PCFParameters::B1;

    EXPECT_EQ(params.N, 1ULL << 20);  // 2^20
    EXPECT_EQ(params.tau, 512UL);
    EXPECT_EQ(params.seed_size, 32UL);

    // Noise rate should be reasonable
    double rate = static_cast<double>(params.tau) / static_cast<double>(params.N);
    EXPECT_LT(rate, 0.001);  // Less than 0.1%
}

TEST(ParametersTest, R2Parameters) {
    // From Table 8: Z_{2^64} parameters
    const auto& params = PCFParameters::R2;

    EXPECT_EQ(params.N, 1ULL << 18);  // 2^18
    EXPECT_EQ(params.tau, 256UL);
    EXPECT_EQ(params.seed_size, 32UL);
}

TEST(ParametersTest, AllParametersValid) {
    std::vector<PCFParameters> all_params = {
        PCFParameters::B1, PCFParameters::B2, PCFParameters::B3,
        PCFParameters::B4, PCFParameters::R1, PCFParameters::R2,
        PCFParameters::R3, PCFParameters::P1
    };

    for (const auto& params : all_params) {
        EXPECT_GT(params.N, 0UL);
        EXPECT_GT(params.tau, 0UL);
        EXPECT_LE(params.tau, params.N);
        EXPECT_GT(params.seed_size, 0UL);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
