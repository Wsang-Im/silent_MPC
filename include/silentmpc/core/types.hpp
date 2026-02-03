#pragma once

#include <cstdint>
#include <array>
#include <span>
#include <vector>
#include <concepts>
#include <stdexcept>

namespace silentmpc {

// ============================================================================
// Basic Types (Section 2.1)
// ============================================================================

/// Session identifier (128-bit entropy, Contract 0)
using SessionId = std::array<uint8_t, 16>;

/// Epoch ID (64-bit monotonic counter, Contract 1)
using EpochId = uint64_t;

/// Index type for correlation queries
using CorrelationIndex = uint64_t;

/// Party identifier
using PartyId = uint8_t;

// ============================================================================
// Ring Interface (Section 3.1, Layer 1)
// ============================================================================

/// Commutative ring concept - supports GF(2), GF(q), Z_{2^k}
template<typename R>
concept Ring = requires(R a, R b) {
    // Addition
    { a + b } -> std::same_as<R>;
    { a += b } -> std::same_as<R&>;

    // Multiplication
    { a * b } -> std::same_as<R>;
    { a *= b } -> std::same_as<R&>;

    // Zero and One
    { R::zero() } -> std::same_as<R>;
    { R::one() } -> std::same_as<R>;

    // Serialization
    { a.serialize() } -> std::convertible_to<std::vector<uint8_t>>;
    { R::deserialize(std::span<const uint8_t>{}) } -> std::same_as<R>;
};

// ============================================================================
// Parameter Configuration (Appendix D)
// ============================================================================

/// LPN/PCF parameters
struct PCFParameters {
    size_t N;           ///< Batch size (expansion length)
    size_t n;           ///< Primal dimension
    size_t w;           ///< Regular noise weight
    size_t tile_size;   ///< Tile size B (Section 3.3), default 2^14

    /// Security level target
    enum class SecurityLevel : uint8_t {
        Conservative = 128,  ///< Tier 1: unstructured
        Hardened = 150       ///< Tier 2: structured with buffer
    } security_level;

    /// Matrix structure
    enum class MatrixStructure : uint8_t {
        Unstructured,   ///< Tier 1: A ← U(R^{N×n})
        Structured      ///< Tier 2: A from structured family
    } matrix_structure;

    PCFParameters() : N(0), n(0), w(0), tile_size(1ULL << 14),
        security_level(SecurityLevel::Conservative),
        matrix_structure(MatrixStructure::Unstructured) {}

    constexpr PCFParameters(
        size_t N_, size_t n_, size_t w_,
        size_t tile_size_ = 1ULL << 14,
        SecurityLevel sec = SecurityLevel::Conservative,
        MatrixStructure mat = MatrixStructure::Unstructured
    ) : N(N_), n(n_), w(w_), tile_size(tile_size_),
        security_level(sec), matrix_structure(mat) {}
};

// Parameter sets from Tables 6-8
namespace parameters {

// Boolean (GF(2)) - Table 6
constexpr PCFParameters B1{1ULL << 20, 512, 32};
constexpr PCFParameters B2{1ULL << 22, 768, 64};
constexpr PCFParameters B3{1ULL << 24, 1024, 64};

// Field (GF(q)) - Table 7
constexpr PCFParameters F1{1ULL << 18, 512, 32};  // GF(257)
constexpr PCFParameters F2{1ULL << 18, 384, 32};  // GF(65537)
constexpr PCFParameters F3{1ULL << 20, 512, 64};  // GF(2^16)

// Ring (Z_{2^k}) - Table 8
constexpr PCFParameters R1{1ULL << 18, 256, 32};  // Z_{2^32}
constexpr PCFParameters R2{1ULL << 18, 256, 32};  // Z_{2^64}
constexpr PCFParameters R3{1ULL << 20, 384, 64};  // Z_{2^96}

// Hardened variants (Tier 2)
constexpr PCFParameters R1_star{
    1ULL << 18, 512, 48,
    1ULL << 14,
    PCFParameters::SecurityLevel::Hardened,
    PCFParameters::MatrixStructure::Structured
};
constexpr PCFParameters R2_star{
    1ULL << 18, 512, 64,
    1ULL << 14,
    PCFParameters::SecurityLevel::Hardened,
    PCFParameters::MatrixStructure::Structured
};

} // namespace parameters

// ============================================================================
// Runtime State (Section 2.3)
// ============================================================================

// ============================================================================
// Derivation Context (Section 5.3, C.1)
// ============================================================================

/// Domain separation tags for PRF/KDF
enum class TermId : uint8_t {
    Setup = 0,
    Term1_DenseDense = 1,
    Term2_SparseDense = 2,
    Term3_SparseDense = 3,
    Term4_SparseSparse = 4,
    DenseVector = 5,
    SparseIndices = 6,
    SparseValues = 7,
    ShortSecret = 8,
    Noise = 9
};

/// Full derivation context for PRF/KDF (why c=4 in Eq. 3)
struct DerivationContext {
    SessionId sid;
    EpochId epoch;
    CorrelationIndex index;
    PartyId party;
    TermId term_id;
    size_t tile_id;  ///< For tiled evaluation (Section 3.3)

    DerivationContext() = default;

    DerivationContext(SessionId s, EpochId e, CorrelationIndex i, PartyId p, TermId t, size_t tid)
        : sid(s), epoch(e), index(i), party(p), term_id(t), tile_id(tid) {}

    /// Serialize to bytes for KDF input
    std::vector<uint8_t> serialize() const;
};

/// Immutable cryptographic handle (Section 2.1)
struct CryptoHandle {
    std::vector<uint8_t> root_seed;  ///< K_root for PRF/KDF
    PCFParameters params;             ///< Fixed parameters
    SessionId sid;                    ///< Public session identifier

    CryptoHandle() = default;

    CryptoHandle(std::vector<uint8_t> seed, PCFParameters p, SessionId s)
        : root_seed(std::move(seed)), params(p), sid(s) {}
};

/// Mutable runtime context (Section 2.1)
struct RuntimeContext {
    SessionId sid;                    ///< Session identifier
    EpochId epoch;                    ///< Current epoch ε
    CorrelationIndex high_watermark;  ///< Allocation high-watermark
    DerivationContext derivation_ctx;  ///< Context for derivations

    explicit RuntimeContext(EpochId e = 0)
        : sid{0}, epoch(e), high_watermark(0), derivation_ctx{} {}
};

// ============================================================================
// Correlation Types
// ============================================================================

/// OLE correlation output (Section 3.2)
template<Ring R>
struct OLECorrelation {
    R value;                   ///< Correlation value
    CorrelationIndex index;    ///< Index for this correlation

    OLECorrelation() : value(R::zero()), index(0) {}
    explicit OLECorrelation(R v) : value(v), index(0) {}
    OLECorrelation(R v, CorrelationIndex idx) : value(v), index(idx) {}
};

/// Beaver triple (derived from OLE)
template<Ring R>
struct BeaverTriple {
    R a, b, c;  ///< Satisfies c = a * b (in secret-shared form)

    BeaverTriple() = default;
    BeaverTriple(R a_, R b_, R c_) : a(a_), b(b_), c(c_) {}
};

// ============================================================================
// Error Types
// ============================================================================

/// Exception hierarchy
class SilentMPCError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

class SetupError : public SilentMPCError {
    using SilentMPCError::SilentMPCError;
};

class EvalError : public SilentMPCError {
    using SilentMPCError::SilentMPCError;
};

class ReuseError : public SilentMPCError {
public:
    ReuseError() : SilentMPCError("Index reuse detected (freshness violation)") {}
};

class EpochMismatchError : public SilentMPCError {
public:
    EpochMismatchError() : SilentMPCError("Epoch mismatch (Contract 1 violation)") {}
};

} // namespace silentmpc
