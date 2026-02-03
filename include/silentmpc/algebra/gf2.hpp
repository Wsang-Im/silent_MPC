#pragma once

#include <cstdint>
#include <vector>
#include <span>
#include <bit>
#include <immintrin.h>  // AVX-512

namespace silentmpc::algebra {

// ============================================================================
// GF(2) Implementation with AVX-512 (Section 6.2)
// ============================================================================

/// GF(2) element (Boolean field)
class GF2 {
private:
    bool value_;

public:
    // Constructors
    constexpr GF2() : value_(false) {}
    constexpr explicit GF2(bool v) : value_(v) {}

    // Ring operations
    constexpr GF2 operator+(const GF2& other) const {
        return GF2(value_ ^ other.value_);
    }

    constexpr GF2& operator+=(const GF2& other) {
        value_ ^= other.value_;
        return *this;
    }

    constexpr GF2 operator*(const GF2& other) const {
        return GF2(value_ & other.value_);
    }

    constexpr GF2& operator*=(const GF2& other) {
        value_ &= other.value_;
        return *this;
    }

    // Ring identity elements
    static constexpr GF2 zero() { return GF2(false); }
    static constexpr GF2 one() { return GF2(true); }

    // Serialization
    std::vector<uint8_t> serialize() const {
        return {static_cast<uint8_t>(value_)};
    }

    static GF2 deserialize(std::span<const uint8_t> data) {
        return GF2(data[0] != 0);
    }

    // Utility
    constexpr bool value() const { return value_; }
    constexpr bool as_bool() const { return value_; }
    constexpr operator bool() const { return value_; }

    // Conversion methods for Ring compatibility
    static constexpr GF2 from_uint64(uint64_t v) {
        return GF2(v & 1);  // Only use LSB
    }

    constexpr uint64_t to_uint64() const {
        return value_ ? 1ULL : 0ULL;
    }

    // Comparison operators
    constexpr bool operator==(const GF2& other) const = default;
};

// ============================================================================
// Bit-sliced SIMD Operations (Section 6.2)
// ============================================================================

/// Vectorized GF(2) operations using AVX-512
class GF2_AVX512 {
public:
    /// XOR two 512-bit blocks (addition in GF(2))
    static inline __m512i add(__m512i a, __m512i b) {
        return _mm512_xor_si512(a, b);
    }

    /// AND two 512-bit blocks (multiplication in GF(2))
    static inline __m512i mul(__m512i a, __m512i b) {
        return _mm512_and_si512(a, b);
    }

    /// Ternary logic for complex GF(2) expressions
    /// Computes: (a & b) ^ c using VPTERNLOG
    static inline __m512i ternary_and_xor(__m512i a, __m512i b, __m512i c) {
        // Immediate value 0x96 represents: (A & B) ^ C
        return _mm512_ternarylogic_epi64(a, b, c, 0x96);
    }

    /// Dense matrix-vector multiplication: y = A * s over GF(2)
    /// A: N × n bit matrix (row-major)
    /// s: n-bit vector
    /// y: N-bit vector (output)
    static void matrix_vector_mul(
        std::span<const uint64_t> A,  // N × n packed bits
        std::span<const uint64_t> s,  // n packed bits
        std::span<uint64_t> y,        // N packed bits
        size_t N, size_t n
    );

    /// Optimized inner product for sparse vectors
    static bool sparse_inner_product(
        std::span<const size_t> indices,
        std::span<const uint64_t> dense_vec,
        size_t dense_len
    );
};

// ============================================================================
// Batched GF(2) Vector Operations
// ============================================================================

/// Batch operations on GF(2) vectors
class GF2Vector {
private:
    std::vector<uint64_t> data_;  // Packed bits (64 per uint64_t)
    size_t bit_length_;

public:
    explicit GF2Vector(size_t n)
        : data_((n + 63) / 64, 0), bit_length_(n) {}

    // Element access
    bool get(size_t i) const {
        return (data_[i / 64] >> (i % 64)) & 1;
    }

    void set(size_t i, bool value) {
        if (value) {
            data_[i / 64] |= (1ULL << (i % 64));
        } else {
            data_[i / 64] &= ~(1ULL << (i % 64));
        }
    }

    // Vector addition (XOR)
    GF2Vector& operator+=(const GF2Vector& other);

    // Hadamard product (element-wise AND)
    GF2Vector hadamard(const GF2Vector& other) const;

    // Inner product
    bool inner_product(const GF2Vector& other) const;

    // Utility
    size_t size() const { return bit_length_; }
    std::span<const uint64_t> packed_data() const { return data_; }
    std::span<uint64_t> packed_data() { return data_; }

    // Hamming weight (popcount)
    size_t hamming_weight() const {
        size_t weight = 0;
        for (auto word : data_) {
            weight += std::popcount(word);
        }
        return weight;
    }
};

// ============================================================================
// AVX2 Optimized GF(2) Operations (256-bit SIMD)
// ============================================================================

/// Vectorized GF(2) operations using AVX2 (256-bit)
/// Used when AVX-512 is not available but AVX2 is supported
class GF2_AVX2 {
public:
    /// XOR two 256-bit blocks (addition in GF(2))
    static inline __m256i add(__m256i a, __m256i b) {
        return _mm256_xor_si256(a, b);
    }

    /// AND two 256-bit blocks (multiplication in GF(2))
    static inline __m256i mul(__m256i a, __m256i b) {
        return _mm256_and_si256(a, b);
    }

    /// Optimized matrix-vector multiplication for GF(2) using AVX2
    /// A: N × n bit matrix (packed as uint64_t, row-major)
    /// s: n-bit vector (packed as uint64_t)
    /// y: N-bit output vector (packed as uint64_t)
    static void matrix_vector_mul_packed(
        const uint64_t* A,     // Matrix: N rows × (n/64) uint64_t per row
        const uint64_t* s,     // Vector: (n/64) uint64_t
        uint64_t* y,           // Output: (N/64) uint64_t
        size_t N,              // Number of rows (must be multiple of 256)
        size_t n               // Number of columns (must be multiple of 64)
    ) {
        const size_t n_words = n / 64;  // Number of 64-bit words in s
        const size_t N_words = N / 64;  // Number of 64-bit words in output

        // Process 256 rows at a time (256 bits = 4 × 64-bit words)
        for (size_t row_block = 0; row_block < N; row_block += 256) {
            // Initialize accumulator for 4 output words
            __m256i acc0 = _mm256_setzero_si256();
            __m256i acc1 = _mm256_setzero_si256();
            __m256i acc2 = _mm256_setzero_si256();
            __m256i acc3 = _mm256_setzero_si256();

            // For each column block
            for (size_t col = 0; col < n_words; ++col) {
                // Load s[col] and broadcast to all 4 lanes
                __m256i s_broadcast = _mm256_set1_epi64x(s[col]);

                // Process 4 rows (256 bits) at a time
                for (size_t i = 0; i < 4; ++i) {
                    size_t row = row_block + i * 64;
                    if (row >= N) break;

                    const uint64_t* A_row = A + row * n_words;
                    __m256i a_val = _mm256_loadu_si256((__m256i*)(A_row + col * 4));

                    // Compute AND and accumulate with XOR
                    __m256i product = _mm256_and_si256(a_val, s_broadcast);

                    if (i == 0) acc0 = _mm256_xor_si256(acc0, product);
                    else if (i == 1) acc1 = _mm256_xor_si256(acc1, product);
                    else if (i == 2) acc2 = _mm256_xor_si256(acc2, product);
                    else acc3 = _mm256_xor_si256(acc3, product);
                }
            }

            // Horizontal reduction: XOR all bits in each 256-bit accumulator
            // to get a single bit per row
            // Store results (simplified - needs proper bit packing)
            size_t out_idx = row_block / 64;
            if (out_idx < N_words) {
                // This is a simplified version - full implementation would need
                // proper horizontal reduction and bit packing
                _mm256_storeu_si256((__m256i*)&y[out_idx], acc0);
            }
        }
    }

    /// Simpler row-by-row matrix-vector multiplication
    /// More straightforward but still uses AVX2 for inner products
    static void matrix_vector_mul_rowwise(
        const uint64_t* A,     // Matrix: N × n_words uint64_t
        const uint64_t* s,     // Vector: n_words uint64_t
        uint64_t* y,           // Output: N_words uint64_t (packed bits)
        size_t N,              // Number of rows
        size_t n_words         // Number of 64-bit words per row
    ) {
        // Clear output
        size_t N_words = (N + 63) / 64;
        for (size_t i = 0; i < N_words; ++i) {
            y[i] = 0;
        }

        // For each row, compute inner product with s
        for (size_t row = 0; row < N; ++row) {
            const uint64_t* A_row = A + row * n_words;

            // Compute bitwise inner product: XOR of (A_row[i] & s[i]) for all i
            __m256i acc = _mm256_setzero_si256();

            // Process 4 words at a time with AVX2
            size_t col;
            for (col = 0; col + 4 <= n_words; col += 4) {
                __m256i a_vec = _mm256_loadu_si256((__m256i*)&A_row[col]);
                __m256i s_vec = _mm256_loadu_si256((__m256i*)&s[col]);
                __m256i product = _mm256_and_si256(a_vec, s_vec);
                acc = _mm256_xor_si256(acc, product);
            }

            // Horizontal XOR reduction: XOR all 4 uint64_t lanes
            uint64_t acc_array[4];
            _mm256_storeu_si256((__m256i*)acc_array, acc);
            uint64_t result = acc_array[0] ^ acc_array[1] ^ acc_array[2] ^ acc_array[3];

            // Handle remaining words (scalar)
            for (; col < n_words; ++col) {
                result ^= (A_row[col] & s[col]);
            }

            // XOR all bits in result to get final bit
            result ^= result >> 32;
            result ^= result >> 16;
            result ^= result >> 8;
            result ^= result >> 4;
            result ^= result >> 2;
            result ^= result >> 1;
            bool bit = result & 1;

            // Store bit in output
            if (bit) {
                y[row / 64] |= (1ULL << (row % 64));
            }
        }
    }
};

} // namespace silentmpc::algebra
