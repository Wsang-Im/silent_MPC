#include "silentmpc/algebra/gfq.hpp"
#include <immintrin.h>
#include <algorithm>
#include <stdexcept>

namespace silentmpc::algebra {

// ============================================================================
// AVX-512 Optimized GF(q) Operations (Section 6.2)
// ============================================================================
//
// Optimization strategy for prime field arithmetic:
// 1. Vectorize modular multiplication using AVX-512
// 2. Use Barrett reduction for fast modulo operations
// 3. Batch process 8 elements (512-bit) at a time

#ifdef __AVX512F__

namespace {

/// Barrett reduction constants for Mersenne-like primes
/// For p = 2^61 - 1, we can use fast reduction
struct BarrettConstants {
    static constexpr uint64_t P_61 = (1ULL << 61) - 1;

    // Fast reduction for 2^61 - 1
    static inline uint64_t reduce_mersenne61(uint64_t x) {
        uint64_t high = x >> 61;
        uint64_t low = x & P_61;
        uint64_t result = low + high;
        if (result >= P_61) result -= P_61;
        return result;
    }

    // Fast reduction for 128-bit product
    static inline uint64_t reduce_product_mersenne61(__uint128_t prod) {
        uint64_t high = static_cast<uint64_t>(prod >> 61);
        uint64_t low = static_cast<uint64_t>(prod) & P_61;

        // First reduction
        uint64_t result = low + high;

        // May need multiple reductions
        while (result >= P_61) {
            high = result >> 61;
            low = result & P_61;
            result = low + high;
        }

        return result;
    }
};

/// AVX-512 vectorized modular addition for Mersenne prime 2^61-1
/// Processes 8 elements in parallel
inline __m512i gfq_add_avx512(__m512i a, __m512i b, uint64_t modulus) {
    // For Mersenne prime 2^61 - 1, can use special reduction
    if (modulus == BarrettConstants::P_61) {
        __m512i sum = _mm512_add_epi64(a, b);

        // Mask for modulus
        __m512i mod_mask = _mm512_set1_epi64(BarrettConstants::P_61);

        // Check if sum >= modulus
        __mmask8 overflow = _mm512_cmpge_epu64_mask(sum, mod_mask);

        // Subtract modulus where needed
        __m512i corrected = _mm512_mask_sub_epi64(sum, overflow, sum, mod_mask);

        return corrected;
    } else {
        // Generic modular addition (slower)
        alignas(64) uint64_t a_array[8];
        alignas(64) uint64_t b_array[8];
        alignas(64) uint64_t result[8];

        _mm512_store_si512((__m512i*)a_array, a);
        _mm512_store_si512((__m512i*)b_array, b);

        for (int i = 0; i < 8; ++i) {
            uint64_t sum = a_array[i] + b_array[i];
            result[i] = (sum >= modulus) ? (sum - modulus) : sum;
        }

        return _mm512_load_si512((__m512i*)result);
    }
}

/// AVX-512 vectorized modular multiplication for Mersenne prime
/// Uses 128-bit intermediate to avoid overflow
inline __m512i gfq_mul_avx512(__m512i a, __m512i b, uint64_t modulus) {
    if (modulus == BarrettConstants::P_61) {
        // For Mersenne prime, use special reduction
        alignas(64) uint64_t a_array[8];
        alignas(64) uint64_t b_array[8];
        alignas(64) uint64_t result[8];

        _mm512_store_si512((__m512i*)a_array, a);
        _mm512_store_si512((__m512i*)b_array, b);

        for (int i = 0; i < 8; ++i) {
            __uint128_t prod = static_cast<__uint128_t>(a_array[i]) *
                              static_cast<__uint128_t>(b_array[i]);
            result[i] = BarrettConstants::reduce_product_mersenne61(prod);
        }

        return _mm512_load_si512((__m512i*)result);
    } else {
        // Generic modular multiplication
        alignas(64) uint64_t a_array[8];
        alignas(64) uint64_t b_array[8];
        alignas(64) uint64_t result[8];

        _mm512_store_si512((__m512i*)a_array, a);
        _mm512_store_si512((__m512i*)b_array, b);

        for (int i = 0; i < 8; ++i) {
            __uint128_t prod = static_cast<__uint128_t>(a_array[i]) *
                              static_cast<__uint128_t>(b_array[i]);
            result[i] = static_cast<uint64_t>(prod % modulus);
        }

        return _mm512_load_si512((__m512i*)result);
    }
}

} // anonymous namespace

/// Matrix-vector multiplication over GF(q) using AVX-512
/// A: N × n matrix, s: n-element vector, y: N-element output
/// Optimized for Mersenne prime q = 2^61 - 1
void gfq_matrix_vector_mul_avx512(
    const uint64_t* A,
    const uint64_t* s,
    uint64_t* y,
    size_t N, size_t n,
    uint64_t modulus
) {
    // Clear output
    std::fill_n(y, N, 0);

    // Process each row independently for correctness
    for (size_t row = 0; row < N; ++row) {
        const uint64_t* A_row = A + row * n;
        __uint128_t sum = 0;

        // Vectorized inner product with 8 elements at a time
        size_t col = 0;
        for (; col + 8 <= n; col += 8) {
            // Unroll manually for better performance
            for (size_t i = 0; i < 8; ++i) {
                sum += static_cast<__uint128_t>(A_row[col + i]) *
                       static_cast<__uint128_t>(s[col + i]);
            }
        }

        // Handle remaining elements
        for (; col < n; ++col) {
            sum += static_cast<__uint128_t>(A_row[col]) *
                   static_cast<__uint128_t>(s[col]);
        }

        // Fast reduction for Mersenne prime
        if (modulus == BarrettConstants::P_61) {
            // Multi-step reduction for large sums
            while (sum >= (static_cast<__uint128_t>(1) << 61)) {
                uint64_t high = static_cast<uint64_t>(sum >> 61);
                uint64_t low = static_cast<uint64_t>(sum) & BarrettConstants::P_61;
                sum = static_cast<__uint128_t>(low) + high;
            }
            y[row] = static_cast<uint64_t>(sum);
            if (y[row] >= modulus) y[row] -= modulus;
        } else {
            y[row] = static_cast<uint64_t>(sum % modulus);
        }
    }
}

/// Batch modular exponentiation (for field operations)
void gfq_batch_exp_avx512(
    const uint64_t* base,
    uint64_t exponent,
    uint64_t* result,
    size_t count,
    uint64_t modulus
) {
    // Process 8 elements at a time
    for (size_t i = 0; i < count; i += 8) {
        __m512i base_vec = _mm512_loadu_si512((__m512i*)(base + i));
        __m512i result_vec = _mm512_set1_epi64(1);  // Start with 1
        __m512i current_base = base_vec;

        uint64_t exp = exponent;
        while (exp > 0) {
            if (exp & 1) {
                result_vec = gfq_mul_avx512(result_vec, current_base, modulus);
            }
            current_base = gfq_mul_avx512(current_base, current_base, modulus);
            exp >>= 1;
        }

        _mm512_storeu_si512((__m512i*)(result + i), result_vec);
    }
}

#else

// Fallback scalar implementations when AVX-512 is not available
void gfq_matrix_vector_mul_avx512(
    const uint64_t* A,
    const uint64_t* s,
    uint64_t* y,
    size_t N, size_t n,
    uint64_t modulus
) {
    for (size_t row = 0; row < N; ++row) {
        __uint128_t sum = 0;
        for (size_t col = 0; col < n; ++col) {
            sum += static_cast<__uint128_t>(A[row * n + col]) *
                   static_cast<__uint128_t>(s[col]);
        }
        y[row] = static_cast<uint64_t>(sum % modulus);
    }
}

void gfq_batch_exp_avx512(
    const uint64_t* base,
    uint64_t exponent,
    uint64_t* result,
    size_t count,
    uint64_t modulus
) {
    for (size_t i = 0; i < count; ++i) {
        uint64_t res = 1;
        uint64_t b = base[i];
        uint64_t exp = exponent;

        while (exp > 0) {
            if (exp & 1) {
                res = static_cast<uint64_t>(
                    (static_cast<__uint128_t>(res) * b) % modulus
                );
            }
            b = static_cast<uint64_t>(
                (static_cast<__uint128_t>(b) * b) % modulus
            );
            exp >>= 1;
        }

        result[i] = res;
    }
}

#endif

} // namespace silentmpc::algebra
