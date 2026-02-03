#include <gtest/gtest.h>
#include "silentmpc/algebra/gf2.hpp"
#include "silentmpc/algebra/z2k.hpp"
#include "silentmpc/algebra/gfq.hpp"

using namespace silentmpc::algebra;

// ============================================================================
// GF(2) Tests
// ============================================================================

TEST(GF2Test, BasicOperations) {
    GF2 zero = GF2::zero();
    GF2 one = GF2::one();

    // Addition (XOR)
    EXPECT_EQ((zero + zero).value(), false);
    EXPECT_EQ((zero + one).value(), true);
    EXPECT_EQ((one + zero).value(), true);
    EXPECT_EQ((one + one).value(), false);

    // Multiplication (AND)
    EXPECT_EQ((zero * zero).value(), false);
    EXPECT_EQ((zero * one).value(), false);
    EXPECT_EQ((one * zero).value(), false);
    EXPECT_EQ((one * one).value(), true);
}

TEST(GF2Test, RingProperties) {
    GF2 a(true);
    GF2 b(false);
    GF2 c(true);

    // Commutativity
    EXPECT_EQ(a + b, b + a);
    EXPECT_EQ(a * b, b * a);

    // Associativity
    EXPECT_EQ((a + b) + c, a + (b + c));
    EXPECT_EQ((a * b) * c, a * (b * c));

    // Distributivity
    EXPECT_EQ(a * (b + c), (a * b) + (a * c));

    // Identity elements
    EXPECT_EQ(a + GF2::zero(), a);
    EXPECT_EQ(a * GF2::one(), a);
}

TEST(GF2VectorTest, VectorOperations) {
    GF2Vector v1(8);
    GF2Vector v2(8);

    v1.set(0, true);
    v1.set(3, true);
    v1.set(7, true);

    v2.set(0, true);
    v2.set(2, true);
    v2.set(3, true);

    // Addition (XOR)
    GF2Vector v3 = v1 + v2;
    EXPECT_FALSE(v3.get(0));  // 1 XOR 1 = 0
    EXPECT_TRUE(v3.get(2));   // 0 XOR 1 = 1
    EXPECT_FALSE(v3.get(3));  // 1 XOR 1 = 0
    EXPECT_TRUE(v3.get(7));   // 1 XOR 0 = 1

    // Hadamard product
    GF2Vector v4 = v1.hadamard(v2);
    EXPECT_TRUE(v4.get(0));   // 1 AND 1 = 1
    EXPECT_FALSE(v4.get(2));  // 0 AND 1 = 0
    EXPECT_TRUE(v4.get(3));   // 1 AND 1 = 1

    // Inner product
    bool ip = v1.inner_product(v2);
    // v1: 10010001, v2: 10100001
    // AND: 10000001 -> popcount = 2 -> 2 mod 2 = 0
    EXPECT_FALSE(ip);
}

TEST(GF2AVX512Test, MatrixVectorMul) {
    // Test matrix-vector multiplication over GF(2)
    size_t N = 16;
    size_t n = 8;

    std::vector<uint64_t> A((N * n + 63) / 64, 0);  // N × n matrix
    std::vector<uint64_t> s((n + 63) / 64, 0);      // n-bit vector
    std::vector<uint64_t> y((N + 63) / 64, 0);      // N-bit output

    // Set up simple test case: identity-like pattern
    // A[i][i] = 1 for i < min(N, n)
    for (size_t i = 0; i < std::min(N, n); ++i) {
        size_t bit_index = i * n + i;
        size_t word = bit_index / 64;
        size_t bit = bit_index % 64;
        if (word < A.size()) {
            A[word] |= (1ULL << bit);
        }
    }

    // Set s to alternating pattern: 10101010
    s[0] = 0xAAAAAAAAAAAAAAAAULL;

    // Compute y = A * s
    GF2_AVX512::matrix_vector_mul(A, s, y, N, n);

    // Check result: should match pattern in s for diagonal elements
    for (size_t i = 0; i < std::min(N, n); ++i) {
        bool s_bit = (i % 2 == 1);  // Alternating pattern
        bool y_bit = (y[i / 64] >> (i % 64)) & 1;
        EXPECT_EQ(y_bit, s_bit);
    }
}

// ============================================================================
// Z2k Tests
// ============================================================================

TEST(Z2kTest, Z264BasicOperations) {
    using Z64 = Z2k<64>;

    Z64 a = Z64::from_uint64(100);
    Z64 b = Z64::from_uint64(50);

    // Addition
    Z64 sum = a + b;
    EXPECT_EQ(sum.to_uint64(), 150ULL);

    // Multiplication
    Z64 prod = a * b;
    EXPECT_EQ(prod.to_uint64(), 5000ULL);

    // Subtraction (via negation)
    Z64 neg_b = Z64::zero() - b;
    Z64 diff = a + neg_b;
    EXPECT_EQ(diff.to_uint64(), 50ULL);
}

TEST(Z2kTest, Z232BasicOperations) {
    using Z32 = Z2k<32>;

    Z32 a = Z32::from_uint64(0xFFFFFFFF);  // Max 32-bit
    Z32 b = Z32::from_uint64(1);

    // Overflow wraps
    Z32 sum = a + b;
    EXPECT_EQ(sum.to_uint64(), 0ULL);

    // Identity
    EXPECT_EQ(a + Z32::zero(), a);
    EXPECT_EQ(a * Z32::one(), a);
}

TEST(Z2kTest, LargeZ2k) {
    using Z128 = Z2k<128>;

    Z128 a = Z128::from_uint64(0xFFFFFFFFFFFFFFFFULL);
    Z128 b = Z128::from_uint64(1);

    // Addition with carry
    Z128 sum = a + b;
    // Result should be 2^64 (represented in 128 bits)

    Z128 expected = Z128::zero();
    // Since we can't directly construct 2^64, test properties
    EXPECT_NE(sum, a);
    EXPECT_NE(sum, b);
}

TEST(Z2kTest, ZeroDivisors) {
    using Z32 = Z2k<32>;

    // In Z_{2^32}, 2^31 is a zero divisor
    // 2 * 2^31 = 2^32 = 0 (mod 2^32)
    Z32 two = Z32::from_uint64(2);
    Z32 half = Z32::from_uint64(1ULL << 31);

    Z32 prod = two * half;
    EXPECT_EQ(prod, Z32::zero());
}

// ============================================================================
// GF(q) Tests
// ============================================================================

TEST(GFqTest, GF127Operations) {
    using GF127 = GFq<127>;  // Mersenne prime 2^7 - 1

    GF127 a = GF127::from_uint64(100);
    GF127 b = GF127::from_uint64(50);

    // Addition
    GF127 sum = a + b;
    EXPECT_EQ(sum.to_uint64(), 23ULL);  // (100 + 50) mod 127 = 23

    // Multiplication
    GF127 prod = a * b;
    EXPECT_EQ(prod.to_uint64(), 5000ULL % 127);

    // Subtraction
    GF127 diff = a - b;
    EXPECT_EQ(diff.to_uint64(), 50ULL);
}

TEST(GFqTest, MersennePrimeOptimization) {
    // Test with 2^61 - 1 (Mersenne prime)
    using GFMersenne = GFq<2305843009213693951ULL>;

    GFMersenne a = GFMersenne::from_uint64(1000000000000ULL);
    GFMersenne b = GFMersenne::from_uint64(2000000000000ULL);

    GFMersenne sum = a + b;
    EXPECT_EQ(sum.to_uint64(), 3000000000000ULL);

    // Test near modulus
    GFMersenne large = GFMersenne::from_uint64(2305843009213693950ULL);
    GFMersenne two = GFMersenne::from_uint64(2);
    GFMersenne wrap = large + two;
    EXPECT_EQ(wrap.to_uint64(), 1ULL);
}

TEST(GFqTest, FieldProperties) {
    using GF127 = GFq<127>;

    GF127 a = GF127::from_uint64(37);
    GF127 b = GF127::from_uint64(89);
    GF127 c = GF127::from_uint64(13);

    // Commutativity
    EXPECT_EQ(a + b, b + a);
    EXPECT_EQ(a * b, b * a);

    // Associativity
    EXPECT_EQ((a + b) + c, a + (b + c));
    EXPECT_EQ((a * b) * c, a * (b * c));

    // Distributivity
    EXPECT_EQ(a * (b + c), (a * b) + (a * c));

    // Identities
    EXPECT_EQ(a + GF127::zero(), a);
    EXPECT_EQ(a * GF127::one(), a);
}

// ============================================================================
// Ring Concept Tests
// ============================================================================

template<Ring R>
void test_ring_axioms() {
    R a = R::from_uint64(17);
    R b = R::from_uint64(23);
    R c = R::from_uint64(31);

    // Associativity of addition
    EXPECT_EQ((a + b) + c, a + (b + c));

    // Associativity of multiplication
    EXPECT_EQ((a * b) * c, a * (b * c));

    // Commutativity of addition
    EXPECT_EQ(a + b, b + a);

    // Additive identity
    EXPECT_EQ(a + R::zero(), a);

    // Multiplicative identity
    EXPECT_EQ(a * R::one(), a);

    // Distributivity
    EXPECT_EQ(a * (b + c), (a * b) + (a * c));
}

TEST(RingConceptTest, GF2SatisfiesRing) {
    test_ring_axioms<GF2>();
}

TEST(RingConceptTest, Z264SatisfiesRing) {
    test_ring_axioms<Z2k<64>>();
}

TEST(RingConceptTest, GFqSatisfiesRing) {
    test_ring_axioms<GFq<127>>();
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
