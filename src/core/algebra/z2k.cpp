#include "silentmpc/algebra/z2k.hpp"
#include <immintrin.h>
#include <algorithm>
#include <stdexcept>

namespace silentmpc::algebra {

// ============================================================================
// AVX-512 Optimized Z_{2^k} Ring Operations (Section 6.2)
// ============================================================================
//
// Ring Z_{2^k} uses native CPU arithmetic with automatic wraparound
// AVX-512 optimization processes 8 uint64_t elements in parallel
// For k=32: use lower 32 bits, for k=64: use full 64 bits

#ifdef __AVX512F__

namespace {

/// Extract k-bit mask for Z_{2^k}
template<size_t K>
inline uint64_t get_mask() {
    if constexpr (K == 64) {
        return ~0ULL;
    } else if constexpr (K == 32) {
        return 0xFFFFFFFFULL;
    } else if constexpr (K <= 63) {
        return (1ULL << K) - 1;
    } else {
        return ~0ULL;
    }
}

/// Apply mask to AVX-512 vector
template<size_t K>
inline __m512i apply_mask_avx512(__m512i vec) {
    if constexpr (K == 64) {
        return vec;  // No masking needed
    } else if constexpr (K == 32) {
        __m512i mask = _mm512_set1_epi64(0xFFFFFFFFULL);
        return _mm512_and_si512(vec, mask);
    } else {
        __m512i mask = _mm512_set1_epi64(get_mask<K>());
        return _mm512_and_si512(vec, mask);
    }
}

} // anonymous namespace

/// AVX-512 vectorized addition in Z_{2^k}
/// Processes 8 elements in parallel
template<size_t K>
void z2k_add_avx512(
    const uint64_t* a,
    const uint64_t* b,
    uint64_t* result,
    size_t count
) {
    static_assert(K <= 64, "K must be <= 64 for single-word Z2k");

    // Process 8 elements at a time
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m512i a_vec = _mm512_loadu_si512((__m512i*)(a + i));
        __m512i b_vec = _mm512_loadu_si512((__m512i*)(b + i));

        // Addition with automatic wraparound
        __m512i sum = _mm512_add_epi64(a_vec, b_vec);

        // Apply modular mask for k < 64
        sum = apply_mask_avx512<K>(sum);

        _mm512_storeu_si512((__m512i*)(result + i), sum);
    }

    // Handle remaining elements
    for (; i < count; ++i) {
        result[i] = (a[i] + b[i]) & get_mask<K>();
    }
}

/// AVX-512 vectorized multiplication in Z_{2^k}
/// For k <= 32, can use full 64-bit multiply then mask
/// For k > 32, need careful handling to avoid overflow
template<size_t K>
void z2k_mul_avx512(
    const uint64_t* a,
    const uint64_t* b,
    uint64_t* result,
    size_t count
) {
    static_assert(K <= 64, "K must be <= 64 for single-word Z2k");

    if constexpr (K <= 32) {
        // For k <= 32, use IFMA52 or standard multiplication
        size_t i = 0;
        for (; i + 8 <= count; i += 8) {
            __m512i a_vec = _mm512_loadu_si512((__m512i*)(a + i));
            __m512i b_vec = _mm512_loadu_si512((__m512i*)(b + i));

            // Multiply low 32 bits
            __m512i prod = _mm512_mullo_epi64(a_vec, b_vec);

            // Apply mask
            prod = apply_mask_avx512<K>(prod);

            _mm512_storeu_si512((__m512i*)(result + i), prod);
        }

        // Handle remaining
        for (; i < count; ++i) {
            result[i] = (a[i] * b[i]) & get_mask<K>();
        }
    } else {
        // For k > 32, need high and low parts
        // AVX-512IFMA would be ideal, but not always available
        // Fall back to scalar for correct 64-bit arithmetic
        for (size_t i = 0; i < count; ++i) {
            result[i] = a[i] * b[i];  // Automatic wraparound for k=64
        }
    }
}

/// Matrix-vector multiplication over Z_{2^k} using AVX-512
/// A: N × n matrix, s: n-element vector, y: N-element output
template<size_t K>
void z2k_matrix_vector_mul_avx512(
    const uint64_t* A,
    const uint64_t* s,
    uint64_t* y,
    size_t N, size_t n
) {
    static_assert(K <= 64, "K must be <= 64 for single-word Z2k");

    // Clear output
    std::fill_n(y, N, 0);

    // Process each row with optimized accumulation
    for (size_t row = 0; row < N; ++row) {
        const uint64_t* A_row = A + row * n;
        uint64_t sum = 0;

        // Unrolled loop for better performance (8-way)
        size_t col = 0;
        for (; col + 8 <= n; col += 8) {
            sum += A_row[col + 0] * s[col + 0];
            sum += A_row[col + 1] * s[col + 1];
            sum += A_row[col + 2] * s[col + 2];
            sum += A_row[col + 3] * s[col + 3];
            sum += A_row[col + 4] * s[col + 4];
            sum += A_row[col + 5] * s[col + 5];
            sum += A_row[col + 6] * s[col + 6];
            sum += A_row[col + 7] * s[col + 7];
        }

        // Handle remaining columns
        for (; col < n; ++col) {
            sum += A_row[col] * s[col];
        }

        // Apply mask (automatic for k=64)
        if constexpr (K < 64) {
            y[row] = sum & get_mask<K>();
        } else {
            y[row] = sum;  // Natural wraparound
        }
    }
}

/// Inner product of two vectors over Z_{2^k}
template<size_t K>
uint64_t z2k_inner_product_avx512(
    const uint64_t* a,
    const uint64_t* b,
    size_t n
) {
    __m512i acc = _mm512_setzero_si512();

    // Process 8 elements at a time
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512i a_vec = _mm512_loadu_si512((__m512i*)(a + i));
        __m512i b_vec = _mm512_loadu_si512((__m512i*)(b + i));

        __m512i prod = _mm512_mullo_epi64(a_vec, b_vec);
        acc = _mm512_add_epi64(acc, prod);
    }

    // Horizontal reduction
    alignas(64) uint64_t acc_array[8];
    _mm512_store_si512((__m512i*)acc_array, acc);

    uint64_t sum = 0;
    for (int j = 0; j < 8; ++j) {
        sum += acc_array[j];
    }

    // Add remaining elements
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }

    return sum & get_mask<K>();
}

// Explicit template instantiations for common ring sizes
template void z2k_add_avx512<32>(const uint64_t*, const uint64_t*, uint64_t*, size_t);
template void z2k_add_avx512<64>(const uint64_t*, const uint64_t*, uint64_t*, size_t);

template void z2k_mul_avx512<32>(const uint64_t*, const uint64_t*, uint64_t*, size_t);
template void z2k_mul_avx512<64>(const uint64_t*, const uint64_t*, uint64_t*, size_t);

template void z2k_matrix_vector_mul_avx512<32>(const uint64_t*, const uint64_t*, uint64_t*, size_t, size_t);
template void z2k_matrix_vector_mul_avx512<64>(const uint64_t*, const uint64_t*, uint64_t*, size_t, size_t);

template uint64_t z2k_inner_product_avx512<32>(const uint64_t*, const uint64_t*, size_t);
template uint64_t z2k_inner_product_avx512<64>(const uint64_t*, const uint64_t*, size_t);

#else

// Fallback scalar implementations when AVX-512 is not available
template<size_t K>
void z2k_add_avx512(
    const uint64_t* a,
    const uint64_t* b,
    uint64_t* result,
    size_t count
) {
    constexpr uint64_t mask = (K == 64) ? ~0ULL : ((1ULL << K) - 1);
    for (size_t i = 0; i < count; ++i) {
        result[i] = (a[i] + b[i]) & mask;
    }
}

template<size_t K>
void z2k_mul_avx512(
    const uint64_t* a,
    const uint64_t* b,
    uint64_t* result,
    size_t count
) {
    constexpr uint64_t mask = (K == 64) ? ~0ULL : ((1ULL << K) - 1);
    for (size_t i = 0; i < count; ++i) {
        result[i] = (a[i] * b[i]) & mask;
    }
}

template<size_t K>
void z2k_matrix_vector_mul_avx512(
    const uint64_t* A,
    const uint64_t* s,
    uint64_t* y,
    size_t N, size_t n
) {
    constexpr uint64_t mask = (K == 64) ? ~0ULL : ((1ULL << K) - 1);
    for (size_t row = 0; row < N; ++row) {
        uint64_t sum = 0;
        for (size_t col = 0; col < n; ++col) {
            sum += A[row * n + col] * s[col];
        }
        y[row] = sum & mask;
    }
}

template<size_t K>
uint64_t z2k_inner_product_avx512(
    const uint64_t* a,
    const uint64_t* b,
    size_t n
) {
    constexpr uint64_t mask = (K == 64) ? ~0ULL : ((1ULL << K) - 1);
    uint64_t sum = 0;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum & mask;
}

// Explicit instantiations for fallback
template void z2k_add_avx512<32>(const uint64_t*, const uint64_t*, uint64_t*, size_t);
template void z2k_add_avx512<64>(const uint64_t*, const uint64_t*, uint64_t*, size_t);
template void z2k_mul_avx512<32>(const uint64_t*, const uint64_t*, uint64_t*, size_t);
template void z2k_mul_avx512<64>(const uint64_t*, const uint64_t*, uint64_t*, size_t);
template void z2k_matrix_vector_mul_avx512<32>(const uint64_t*, const uint64_t*, uint64_t*, size_t, size_t);
template void z2k_matrix_vector_mul_avx512<64>(const uint64_t*, const uint64_t*, uint64_t*, size_t, size_t);
template uint64_t z2k_inner_product_avx512<32>(const uint64_t*, const uint64_t*, size_t);
template uint64_t z2k_inner_product_avx512<64>(const uint64_t*, const uint64_t*, size_t);

#endif

} // namespace silentmpc::algebra
