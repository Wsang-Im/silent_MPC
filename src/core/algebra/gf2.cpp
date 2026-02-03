#include "silentmpc/algebra/gf2.hpp"
#include <algorithm>
#include <stdexcept>

namespace silentmpc::algebra {

// ============================================================================
// GF2_AVX512 Implementation
// ============================================================================

#ifdef __AVX512F__

void GF2_AVX512::matrix_vector_mul(
    std::span<const uint64_t> A,
    std::span<const uint64_t> s,
    std::span<uint64_t> y,
    size_t N, size_t n
) {
    // A: N × n matrix (packed bits, row-major)
    // s: n-bit vector (packed)
    // y: N-bit output vector (packed)
    //
    // REAL AVX-512 Implementation using SIMD intrinsics
    // Processes 8 uint64_t (512 bits) at a time

    const size_t n_words = (n + 63) / 64;
    const size_t N_words = (N + 63) / 64;

    // Clear output
    std::fill(y.begin(), y.end(), 0);

    // Process 512 rows at a time (512-bit output per iteration)
    for (size_t row_block = 0; row_block < N; row_block += 512) {
        __m512i result_vec = _mm512_setzero_si512();

        // For each 512-row block, compute 512 dot products
        for (size_t local_row = 0; local_row < 512 && (row_block + local_row) < N; local_row += 8) {
            // Process 8 rows in parallel using AVX-512
            __m512i acc = _mm512_setzero_si512();

            // Inner product: XOR of (A[row] AND s)
            for (size_t j = 0; j + 8 <= n_words; j += 8) {
                // Load 8 rows × 8 words of matrix A (8x8 = 64 uint64_t)
                for (size_t r = 0; r < 8 && (row_block + local_row + r) < N; ++r) {
                    size_t row_idx = row_block + local_row + r;
                    const uint64_t* A_row = A.data() + row_idx * n_words;

                    // Load 8 words from this row
                    __m512i a_vec = _mm512_loadu_si512((__m512i*)(A_row + j));

                    // Load 8 words from vector s (broadcast for all rows)
                    __m512i s_vec = _mm512_loadu_si512((__m512i*)(s.data() + j));

#ifdef __AVX512DQ__
                    // VPTERNLOG: (a AND s) XOR acc in ONE instruction (Section 6.2)
                    // Immediate 0x6A = truth table for (a & s) ^ c
                    // This matches paper's specification for ternary logic instructions
                    acc = _mm512_ternarylogic_epi64(a_vec, s_vec, acc, 0x6A);
#else
                    // Fallback: 2 instructions (AND + XOR)
                    __m512i and_result = _mm512_and_si512(a_vec, s_vec);
                    acc = _mm512_xor_si512(acc, and_result);
#endif
                }
            }

            // Handle remaining words (scalar fallback)
            for (size_t r = 0; r < 8 && (row_block + local_row + r) < N; ++r) {
                size_t row_idx = row_block + local_row + r;
                const uint64_t* A_row = A.data() + row_idx * n_words;

                uint64_t dot = 0;
                for (size_t j = (n_words / 8) * 8; j < n_words; ++j) {
                    dot ^= (__builtin_popcountll(A_row[j] & s[j]) & 1);
                }

                // Extract accumulated result from SIMD register
                alignas(64) uint64_t acc_array[8];
                _mm512_store_si512((__m512i*)acc_array, acc);

                // XOR all bits in accumulated words
                for (size_t k = 0; k < 8; ++k) {
                    dot ^= (__builtin_popcountll(acc_array[k]) & 1);
                }

                // Set bit in output
                if (dot) {
                    size_t word_idx = row_idx / 64;
                    size_t bit_idx = row_idx % 64;
                    if (word_idx < y.size()) {
                        y[word_idx] |= (1ULL << bit_idx);
                    }
                }
            }
        }
    }
}

bool GF2_AVX512::sparse_inner_product(
    std::span<const size_t> indices,
    std::span<const uint64_t> dense_vec,
    size_t dense_len
) {
    bool result = false;

    for (size_t idx : indices) {
        if (idx < dense_len) {
            size_t word = idx / 64;
            size_t bit = idx % 64;

            if (word < dense_vec.size()) {
                bool bit_value = (dense_vec[word] >> bit) & 1;
                result ^= bit_value;
            }
        }
    }

    return result;
}

#else

// Fallback implementation without AVX-512
void GF2_AVX512::matrix_vector_mul(
    std::span<const uint64_t> A,
    std::span<const uint64_t> s,
    std::span<uint64_t> y,
    size_t N, size_t n
) {
    const size_t n_words = (n + 63) / 64;
    std::fill(y.begin(), y.end(), 0);

    for (size_t row = 0; row < N; ++row) {
        uint64_t dot = 0;
        for (size_t j = 0; j < n_words; ++j) {
            size_t idx = row * n_words + j;
            if (idx < A.size()) {
                dot ^= (__builtin_popcountll(A[idx] & s[j]) & 1);
            }
        }
        if (dot) {
            y[row / 64] |= (1ULL << (row % 64));
        }
    }
}

bool GF2_AVX512::sparse_inner_product(
    std::span<const size_t> indices,
    std::span<const uint64_t> dense_vec,
    size_t dense_len
) {
    bool result = false;
    for (size_t idx : indices) {
        if (idx < dense_len && (idx / 64) < dense_vec.size()) {
            result ^= ((dense_vec[idx / 64] >> (idx % 64)) & 1);
        }
    }
    return result;
}

#endif

// ============================================================================
// GF2Vector Implementation
// ============================================================================

GF2Vector& GF2Vector::operator+=(const GF2Vector& other) {
    if (bit_length_ != other.bit_length_) {
        throw std::invalid_argument("GF2Vector size mismatch");
    }

    for (size_t i = 0; i < data_.size(); ++i) {
        data_[i] ^= other.data_[i];
    }

    return *this;
}

GF2Vector GF2Vector::hadamard(const GF2Vector& other) const {
    if (bit_length_ != other.bit_length_) {
        throw std::invalid_argument("GF2Vector size mismatch");
    }

    GF2Vector result(bit_length_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] & other.data_[i];
    }

    return result;
}

bool GF2Vector::inner_product(const GF2Vector& other) const {
    if (bit_length_ != other.bit_length_) {
        throw std::invalid_argument("GF2Vector size mismatch");
    }

    uint64_t result = 0;
    for (size_t i = 0; i < data_.size(); ++i) {
        result ^= __builtin_popcountll(data_[i] & other.data_[i]);
    }

    return (result & 1) != 0;
}

} // namespace silentmpc::algebra
