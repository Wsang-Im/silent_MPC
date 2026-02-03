#pragma once

#include <cstdint>
#include <vector>
#include <span>
#include <concepts>
#include <type_traits>

namespace silentmpc::algebra {

// ============================================================================
// Z_{2^k} Implementation (Section 6.2 - Rings)
// ============================================================================

/// Ring Z_{2^k} for k ≤ 64 using native CPU arithmetic
/// Handles zero divisors correctly (Section 5.3, 9.2)
template<size_t K>
    requires (K <= 64)
class Z2k {
private:
    using StorageType = std::conditional_t<
        K <= 8, uint8_t,
        std::conditional_t<K <= 16, uint16_t,
        std::conditional_t<K <= 32, uint32_t, uint64_t>>>;

    StorageType value_;

    // Mask for modular reduction
    static constexpr StorageType MASK = (K == 64) ? ~StorageType(0) :
                                         (K == 32 && sizeof(StorageType) == 4) ? ~StorageType(0) :
                                         ((StorageType(1) << K) - 1);

public:
    // Constructors
    constexpr Z2k() : value_(0) {}

    constexpr explicit Z2k(StorageType v) : value_(v & MASK) {}

    template<std::integral T>
    constexpr explicit Z2k(T v) : value_(static_cast<StorageType>(v) & MASK) {}

    // Ring operations (no overflow - wrap around naturally)
    constexpr Z2k operator+(const Z2k& other) const {
        return Z2k(value_ + other.value_);
    }

    constexpr Z2k& operator+=(const Z2k& other) {
        value_ = (value_ + other.value_) & MASK;
        return *this;
    }

    // Subtraction
    constexpr Z2k operator-(const Z2k& other) const {
        return Z2k(value_ - other.value_);
    }

    constexpr Z2k& operator-=(const Z2k& other) {
        value_ = (value_ - other.value_) & MASK;
        return *this;
    }

    constexpr Z2k operator*(const Z2k& other) const {
        if constexpr (K <= 32) {
            return Z2k(value_ * other.value_);
        } else {
            // For K > 32, use 64-bit multiplication with explicit modular reduction
            uint64_t result = static_cast<uint64_t>(value_) *
                              static_cast<uint64_t>(other.value_);
            return Z2k(result);
        }
    }

    constexpr Z2k& operator*=(const Z2k& other) {
        *this = *this * other;
        return *this;
    }

    // Negation (additive inverse)
    constexpr Z2k operator-() const {
        return Z2k((~value_ + 1) & MASK);
    }

    // Comparison operators (for sorting/containers)
    constexpr bool operator<(const Z2k& other) const {
        return value_ < other.value_;
    }

    constexpr bool operator<=(const Z2k& other) const {
        return value_ <= other.value_;
    }

    constexpr bool operator>(const Z2k& other) const {
        return value_ > other.value_;
    }

    constexpr bool operator>=(const Z2k& other) const {
        return value_ >= other.value_;
    }

    // Ring identity elements
    static constexpr Z2k zero() { return Z2k(0); }
    static constexpr Z2k one() { return Z2k(1); }

    // Conversion from uint64_t (for compatibility with DPF code)
    static constexpr Z2k from_uint64(uint64_t v) {
        return Z2k(static_cast<StorageType>(v));
    }

    // Conversion to uint64_t
    constexpr uint64_t to_uint64() const {
        return static_cast<uint64_t>(value_);
    }

    // Serialization (fixed endianness for reproducibility)
    std::vector<uint8_t> serialize() const {
        std::vector<uint8_t> result;
        size_t bytes = (K + 7) / 8;
        result.reserve(bytes);

        StorageType v = value_;
        for (size_t i = 0; i < bytes; ++i) {
            result.push_back(static_cast<uint8_t>(v & 0xFF));
            v >>= 8;
        }
        return result;
    }

    static Z2k deserialize(std::span<const uint8_t> data) {
        StorageType v = 0;
        size_t bytes = std::min(data.size(), (K + 7) / 8);

        for (size_t i = 0; i < bytes; ++i) {
            v |= static_cast<StorageType>(data[i]) << (i * 8);
        }
        return Z2k(v);
    }

    // Utility
    constexpr StorageType value() const { return value_; }

    // Comparison
    constexpr bool operator==(const Z2k& other) const = default;
};

// ============================================================================
// Common type aliases
// ============================================================================

using Z2_32 = Z2k<32>;  ///< Z_{2^32} (Table 8, R1)
using Z2_64 = Z2k<64>;  ///< Z_{2^64} (Table 8, R2)

// ============================================================================
// Multi-limb Z_{2^k} for k > 64 (Section 6.2)
// ============================================================================

/// Z_{2^k} for arbitrary k using multi-limb arithmetic
template<size_t K>
    requires (K > 64)
class Z2k_MultiLimb {
private:
    static constexpr size_t NUM_LIMBS = (K + 63) / 64;
    std::array<uint64_t, NUM_LIMBS> limbs_;

public:
    constexpr Z2k_MultiLimb() : limbs_{} {}

    explicit Z2k_MultiLimb(uint64_t v) : limbs_{} {
        limbs_[0] = v;
    }

    // Addition with explicit carry handling
    Z2k_MultiLimb operator+(const Z2k_MultiLimb& other) const {
        Z2k_MultiLimb result;
        uint64_t carry = 0;

        for (size_t i = 0; i < NUM_LIMBS; ++i) {
            __uint128_t sum = static_cast<__uint128_t>(limbs_[i]) +
                              static_cast<__uint128_t>(other.limbs_[i]) +
                              carry;
            result.limbs_[i] = static_cast<uint64_t>(sum);
            carry = static_cast<uint64_t>(sum >> 64);
        }

        // Mask high bits if K is not a multiple of 64
        if constexpr (K % 64 != 0) {
            constexpr uint64_t high_mask = (1ULL << (K % 64)) - 1;
            result.limbs_[NUM_LIMBS - 1] &= high_mask;
        }

        return result;
    }

    Z2k_MultiLimb& operator+=(const Z2k_MultiLimb& other) {
        *this = *this + other;
        return *this;
    }

    // Multiplication (schoolbook algorithm with carries)
    Z2k_MultiLimb operator*(const Z2k_MultiLimb& other) const {
        Z2k_MultiLimb result;

        for (size_t i = 0; i < NUM_LIMBS; ++i) {
            uint64_t carry = 0;
            for (size_t j = 0; j < NUM_LIMBS - i; ++j) {
                __uint128_t prod = static_cast<__uint128_t>(limbs_[i]) *
                                   static_cast<__uint128_t>(other.limbs_[j]) +
                                   static_cast<__uint128_t>(result.limbs_[i + j]) +
                                   carry;
                result.limbs_[i + j] = static_cast<uint64_t>(prod);
                carry = static_cast<uint64_t>(prod >> 64);
            }
        }

        // Modular reduction
        if constexpr (K % 64 != 0) {
            constexpr uint64_t high_mask = (1ULL << (K % 64)) - 1;
            result.limbs_[NUM_LIMBS - 1] &= high_mask;
        }

        return result;
    }

    Z2k_MultiLimb& operator*=(const Z2k_MultiLimb& other) {
        *this = *this * other;
        return *this;
    }

    // Ring identity
    static Z2k_MultiLimb zero() { return Z2k_MultiLimb(); }

    static Z2k_MultiLimb one() {
        Z2k_MultiLimb result;
        result.limbs_[0] = 1;
        return result;
    }

    // Serialization (little-endian)
    std::vector<uint8_t> serialize() const {
        std::vector<uint8_t> result;
        size_t bytes = (K + 7) / 8;
        result.reserve(bytes);

        for (size_t limb_idx = 0; limb_idx < NUM_LIMBS; ++limb_idx) {
            uint64_t limb = limbs_[limb_idx];
            size_t limb_bytes = (limb_idx == NUM_LIMBS - 1 && K % 64 != 0) ?
                                ((K % 64) + 7) / 8 : 8;

            for (size_t i = 0; i < limb_bytes; ++i) {
                result.push_back(static_cast<uint8_t>(limb & 0xFF));
                limb >>= 8;
            }
        }

        return result;
    }

    static Z2k_MultiLimb deserialize(std::span<const uint8_t> data) {
        Z2k_MultiLimb result;
        size_t byte_idx = 0;

        for (size_t limb_idx = 0; limb_idx < NUM_LIMBS && byte_idx < data.size(); ++limb_idx) {
            uint64_t limb = 0;
            for (size_t i = 0; i < 8 && byte_idx < data.size(); ++i, ++byte_idx) {
                limb |= static_cast<uint64_t>(data[byte_idx]) << (i * 8);
            }
            result.limbs_[limb_idx] = limb;
        }

        return result;
    }

    // Comparison
    bool operator==(const Z2k_MultiLimb& other) const = default;
};

// Alias for Z_{2^96} (Table 8, R3)
using Z2_96 = Z2k_MultiLimb<96>;

// ============================================================================
// Vectorized operations for Z_{2^k}
// ============================================================================

/// Batch operations on Z_{2^k} vectors
template<size_t K>
class Z2kVector {
private:
    std::vector<Z2k<K>> elements_;

public:
    explicit Z2kVector(size_t n) : elements_(n) {}

    // Element access
    Z2k<K>& operator[](size_t i) { return elements_[i]; }
    const Z2k<K>& operator[](size_t i) const { return elements_[i]; }

    // Vector operations
    Z2kVector& operator+=(const Z2kVector& other) {
        for (size_t i = 0; i < elements_.size(); ++i) {
            elements_[i] += other.elements_[i];
        }
        return *this;
    }

    // Hadamard (element-wise) product
    Z2kVector hadamard(const Z2kVector& other) const {
        Z2kVector result(elements_.size());
        for (size_t i = 0; i < elements_.size(); ++i) {
            result.elements_[i] = elements_[i] * other.elements_[i];
        }
        return result;
    }

    // Inner product
    Z2k<K> inner_product(const Z2kVector& other) const {
        Z2k<K> result = Z2k<K>::zero();
        for (size_t i = 0; i < elements_.size(); ++i) {
            result += elements_[i] * other.elements_[i];
        }
        return result;
    }

    size_t size() const { return elements_.size(); }
    std::span<const Z2k<K>> data() const { return elements_; }
    std::span<Z2k<K>> data() { return elements_; }
};

// ============================================================================
// AVX-512 Optimized Functions (implemented in z2k.cpp)
// ============================================================================

/// AVX-512 vectorized addition in Z_{2^k}
template<size_t K>
void z2k_add_avx512(
    const uint64_t* a,
    const uint64_t* b,
    uint64_t* result,
    size_t count
);

/// AVX-512 vectorized multiplication in Z_{2^k}
template<size_t K>
void z2k_mul_avx512(
    const uint64_t* a,
    const uint64_t* b,
    uint64_t* result,
    size_t count
);

/// Matrix-vector multiplication over Z_{2^k} using AVX-512
template<size_t K>
void z2k_matrix_vector_mul_avx512(
    const uint64_t* A,
    const uint64_t* s,
    uint64_t* y,
    size_t N, size_t n
);

/// Inner product over Z_{2^k} using AVX-512
template<size_t K>
uint64_t z2k_inner_product_avx512(
    const uint64_t* a,
    const uint64_t* b,
    size_t n
);

// Explicit extern template declarations for common sizes
extern template void z2k_add_avx512<32>(const uint64_t*, const uint64_t*, uint64_t*, size_t);
extern template void z2k_add_avx512<64>(const uint64_t*, const uint64_t*, uint64_t*, size_t);
extern template void z2k_mul_avx512<32>(const uint64_t*, const uint64_t*, uint64_t*, size_t);
extern template void z2k_mul_avx512<64>(const uint64_t*, const uint64_t*, uint64_t*, size_t);
extern template void z2k_matrix_vector_mul_avx512<32>(const uint64_t*, const uint64_t*, uint64_t*, size_t, size_t);
extern template void z2k_matrix_vector_mul_avx512<64>(const uint64_t*, const uint64_t*, uint64_t*, size_t, size_t);
extern template uint64_t z2k_inner_product_avx512<32>(const uint64_t*, const uint64_t*, size_t);
extern template uint64_t z2k_inner_product_avx512<64>(const uint64_t*, const uint64_t*, size_t);

} // namespace silentmpc::algebra
