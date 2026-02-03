#pragma once

#include <cstdint>
#include <vector>
#include <span>
#include <concepts>
#include <stdexcept>

namespace silentmpc::algebra {

// ============================================================================
// GF(q) Prime Field Implementation (Section 6.2)
// ============================================================================

/// Prime field GF(q) with constant-time modular arithmetic
template<uint64_t Q>
    requires (Q > 2 && Q < (1ULL << 62))  // Mersenne primes fit
class GFq {
private:
    uint64_t value_;

    // Montgomery reduction parameters (optional optimization)
    static constexpr bool USE_MONTGOMERY = (Q < (1ULL << 32));

public:
    // Constructors
    constexpr GFq() : value_(0) {}

    constexpr explicit GFq(uint64_t v) : value_(v % Q) {}

    // Modular reduction (constant-time)
    static constexpr uint64_t reduce(uint64_t v) {
        // For Mersenne primes (2^k - 1), can optimize
        // General case: use Barrett or Montgomery reduction
        return v % Q;
    }

    // Ring operations
    constexpr GFq operator+(const GFq& other) const {
        uint64_t sum = value_ + other.value_;
        // Constant-time reduction
        if (sum >= Q) sum -= Q;
        return GFq(sum);
    }

    constexpr GFq& operator+=(const GFq& other) {
        value_ += other.value_;
        if (value_ >= Q) value_ -= Q;
        return *this;
    }

    constexpr GFq operator*(const GFq& other) const {
        // Use 128-bit intermediate to avoid overflow
        __uint128_t prod = static_cast<__uint128_t>(value_) *
                          static_cast<__uint128_t>(other.value_);
        return GFq(static_cast<uint64_t>(prod % Q));
    }

    constexpr GFq& operator*=(const GFq& other) {
        *this = *this * other;
        return *this;
    }

    constexpr GFq operator-(const GFq& other) const {
        if (value_ >= other.value_) {
            return GFq(value_ - other.value_);
        } else {
            return GFq(Q - (other.value_ - value_));
        }
    }

    constexpr GFq operator-() const {
        return (value_ == 0) ? GFq(0) : GFq(Q - value_);
    }

    // Ring identity
    static constexpr GFq zero() { return GFq(0); }
    static constexpr GFq one() { return GFq(1); }

    // Modular inverse (extended Euclidean algorithm)
    GFq inverse() const {
        if (value_ == 0) {
            throw std::domain_error("Division by zero in GF(q)");
        }

        int64_t t = 0, newt = 1;
        uint64_t r = Q, newr = value_;

        while (newr != 0) {
            uint64_t quotient = r / newr;

            int64_t tempt = t - static_cast<int64_t>(quotient) * newt;
            t = newt;
            newt = tempt;

            uint64_t tempr = r - quotient * newr;
            r = newr;
            newr = tempr;
        }

        if (r > 1) {
            throw std::domain_error("Element is not invertible");
        }

        if (t < 0) {
            t += static_cast<int64_t>(Q);
        }

        return GFq(static_cast<uint64_t>(t));
    }

    GFq operator/(const GFq& other) const {
        return *this * other.inverse();
    }

    // Serialization
    std::vector<uint8_t> serialize() const {
        std::vector<uint8_t> result(8);
        uint64_t v = value_;
        for (size_t i = 0; i < 8; ++i) {
            result[i] = static_cast<uint8_t>(v & 0xFF);
            v >>= 8;
        }
        return result;
    }

    static GFq deserialize(std::span<const uint8_t> data) {
        uint64_t v = 0;
        size_t bytes = std::min(data.size(), size_t(8));
        for (size_t i = 0; i < bytes; ++i) {
            v |= static_cast<uint64_t>(data[i]) << (i * 8);
        }
        return GFq(v);
    }

    // Utility
    constexpr uint64_t value() const { return value_; }
    constexpr bool operator==(const GFq& other) const = default;

    // Exponentiation (for testing)
    GFq pow(uint64_t exp) const {
        GFq result = one();
        GFq base = *this;

        while (exp > 0) {
            if (exp & 1) result *= base;
            base *= base;
            exp >>= 1;
        }
        return result;
    }
};

// ============================================================================
// Mersenne Prime Optimizations (Section 6.2)
// ============================================================================

/// Optimized GF(2^61 - 1) using Mersenne prime properties
class GF_Mersenne61 {
private:
    uint64_t value_;
    static constexpr uint64_t P = (1ULL << 61) - 1;

public:
    constexpr GF_Mersenne61() : value_(0) {}
    constexpr explicit GF_Mersenne61(uint64_t v) : value_(reduce(v)) {}

    // Fast Mersenne reduction: v mod (2^61 - 1)
    static constexpr uint64_t reduce(uint64_t v) {
        // Split into high and low parts
        uint64_t high = v >> 61;
        uint64_t low = v & P;
        uint64_t result = low + high;

        // May need one more reduction
        if (result >= P) result -= P;
        return result;
    }

    constexpr GF_Mersenne61 operator+(const GF_Mersenne61& other) const {
        uint64_t sum = value_ + other.value_;
        if (sum >= P) sum -= P;
        return GF_Mersenne61(sum);
    }

    constexpr GF_Mersenne61& operator+=(const GF_Mersenne61& other) {
        value_ += other.value_;
        if (value_ >= P) value_ -= P;
        return *this;
    }

    constexpr GF_Mersenne61 operator*(const GF_Mersenne61& other) const {
        __uint128_t prod = static_cast<__uint128_t>(value_) *
                          static_cast<__uint128_t>(other.value_);

        // Reduce 128-bit product
        uint64_t high = static_cast<uint64_t>(prod >> 61);
        uint64_t low = static_cast<uint64_t>(prod) & P;

        return GF_Mersenne61(low + high);
    }

    constexpr GF_Mersenne61& operator*=(const GF_Mersenne61& other) {
        *this = *this * other;
        return *this;
    }

    static constexpr GF_Mersenne61 zero() { return GF_Mersenne61(0); }
    static constexpr GF_Mersenne61 one() { return GF_Mersenne61(1); }

    std::vector<uint8_t> serialize() const {
        std::vector<uint8_t> result(8);
        uint64_t v = value_;
        for (size_t i = 0; i < 8; ++i) {
            result[i] = static_cast<uint8_t>(v & 0xFF);
            v >>= 8;
        }
        return result;
    }

    static GF_Mersenne61 deserialize(std::span<const uint8_t> data) {
        uint64_t v = 0;
        for (size_t i = 0; i < std::min(data.size(), size_t(8)); ++i) {
            v |= static_cast<uint64_t>(data[i]) << (i * 8);
        }
        return GF_Mersenne61(v);
    }

    constexpr uint64_t value() const { return value_; }
    constexpr bool operator==(const GF_Mersenne61& other) const = default;
};

// ============================================================================
// Common Type Aliases (Table 7)
// ============================================================================

using GF257 = GFq<257>;          ///< Table 7, F1
using GF65537 = GFq<65537>;      ///< Table 7, F2
using GF_2_61_1 = GF_Mersenne61; ///< Optimized Mersenne prime

// Extension field GF(2^16) - represented as polynomials
class GF2_16 {
private:
    uint16_t value_;  // Polynomial coefficients

    // Irreducible polynomial: x^16 + x^5 + x^3 + x^2 + 1
    static constexpr uint32_t IRREDUCIBLE = 0x1002D;

public:
    constexpr GF2_16() : value_(0) {}
    constexpr explicit GF2_16(uint16_t v) : value_(v) {}

    constexpr GF2_16 operator+(const GF2_16& other) const {
        return GF2_16(value_ ^ other.value_);  // XOR for polynomial addition
    }

    constexpr GF2_16& operator+=(const GF2_16& other) {
        value_ ^= other.value_;
        return *this;
    }

    constexpr GF2_16 operator*(const GF2_16& other) const {
        uint32_t prod = 0;
        uint32_t a = value_;
        uint32_t b = other.value_;

        // Polynomial multiplication
        for (int i = 0; i < 16; ++i) {
            if (b & 1) prod ^= a;
            b >>= 1;
            a <<= 1;
        }

        // Reduce modulo irreducible polynomial
        for (int i = 31; i >= 16; --i) {
            if (prod & (1U << i)) {
                prod ^= (IRREDUCIBLE << (i - 16));
            }
        }

        return GF2_16(static_cast<uint16_t>(prod));
    }

    constexpr GF2_16& operator*=(const GF2_16& other) {
        *this = *this * other;
        return *this;
    }

    static constexpr GF2_16 zero() { return GF2_16(0); }
    static constexpr GF2_16 one() { return GF2_16(1); }

    std::vector<uint8_t> serialize() const {
        return {static_cast<uint8_t>(value_ & 0xFF),
                static_cast<uint8_t>(value_ >> 8)};
    }

    static GF2_16 deserialize(std::span<const uint8_t> data) {
        uint16_t v = data[0];
        if (data.size() > 1) v |= static_cast<uint16_t>(data[1]) << 8;
        return GF2_16(v);
    }

    constexpr uint16_t value() const { return value_; }
    constexpr bool operator==(const GF2_16& other) const = default;
};

// ============================================================================
// NTT Support for Convolution (Section 6.2)
// ============================================================================

/// Number Theoretic Transform for fast polynomial multiplication
template<uint64_t Q>
class NTTHelper {
private:
    std::vector<GFq<Q>> roots_of_unity_;
    std::vector<GFq<Q>> inverse_roots_;
    size_t max_size_;

public:
    explicit NTTHelper(size_t max_size);

    /// Forward NTT
    void forward_ntt(std::span<GFq<Q>> data) const;

    /// Inverse NTT
    void inverse_ntt(std::span<GFq<Q>> data) const;

    /// Convolution via NTT
    std::vector<GFq<Q>> convolve(
        std::span<const GFq<Q>> a,
        std::span<const GFq<Q>> b
    ) const;

private:
    void compute_roots_of_unity();
    GFq<Q> find_primitive_root() const;
};

// ============================================================================
// AVX-512 Optimized Functions (implemented in gfq.cpp)
// ============================================================================

/// Matrix-vector multiplication over GF(q) using AVX-512
/// A: N × n matrix (row-major), s: n-element vector, y: N-element output
void gfq_matrix_vector_mul_avx512(
    const uint64_t* A,
    const uint64_t* s,
    uint64_t* y,
    size_t N, size_t n,
    uint64_t modulus
);

/// Batch modular exponentiation using AVX-512
void gfq_batch_exp_avx512(
    const uint64_t* base,
    uint64_t exponent,
    uint64_t* result,
    size_t count,
    uint64_t modulus
);

} // namespace silentmpc::algebra
