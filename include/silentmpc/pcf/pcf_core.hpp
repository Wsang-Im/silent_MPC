#pragma once

#include "silentmpc/core/types.hpp"
#include "silentmpc/algebra/gf2.hpp"
#include "silentmpc/algebra/z2k.hpp"
#include <memory>
#include <optional>

namespace silentmpc::pcf {

// ============================================================================
// PCF Core Interface (Section 3)
// ============================================================================

/// Four-Term Decomposition (Section 3.2, Equation 2)
/// y0 ⊙ y1 = (T1) Dense-Dense + (T2) Sparse-Dense +
///           (T3) Sparse-Dense + (T4) Sparse-Sparse
template<Ring R>
struct FourTermResult {
    std::vector<R> term1_dense_dense;      ///< (As0) ⊙ (As1)
    std::vector<R> term2_sparse_dense;     ///< e0 ⊙ (As1)
    std::vector<R> term3_sparse_dense;     ///< e1 ⊙ (As0)
    std::vector<R> term4_sparse_sparse;    ///< e0 ⊙ e1 (collision handling)

    explicit FourTermResult(size_t N)
        : term1_dense_dense(N, R::zero()),
          term2_sparse_dense(N, R::zero()),
          term3_sparse_dense(N, R::zero()),
          term4_sparse_sparse(N, R::zero()) {}
};

// ============================================================================
// Regular Noise Distribution (Definition 1, Section 5.1)
// ============================================================================

/// Regular sparse noise: partition index space into w blocks,
/// exactly one non-zero entry per block
template<Ring R>
class RegularNoise {
private:
    std::vector<size_t> support_;     ///< Non-zero indices
    std::vector<R> values_;           ///< Non-zero values
    size_t total_length_;             ///< Total vector length N
    size_t weight_;                   ///< Number of non-zero entries w

public:
    RegularNoise(size_t N, size_t w);

    /// Sample from D_reg(N, w) using derivation context
    static RegularNoise sample(
        size_t N, size_t w,
        const DerivationContext& ctx,
        const std::vector<uint8_t>& seed
    );

    // Accessors
    std::span<const size_t> support() const { return support_; }
    std::span<const R> values() const { return values_; }
    size_t weight() const { return weight_; }
    size_t length() const { return total_length_; }

    /// Expected collision count: E[|C|] = w²/N (Section 3.2)
    double expected_collisions() const {
        return static_cast<double>(weight_ * weight_) / total_length_;
    }
};

// ============================================================================
// LPN Expansion Core (Section 3.2, Layer 2)
// ============================================================================

/// LPN-style PRG: y = A·s + e over ring R
template<Ring R>
class LPNCore {
private:
    std::vector<R> matrix_A_;  ///< Public matrix A ∈ R^{N×n}
    size_t N_, n_;
    PCFParameters params_;

    // AVX2 optimization: pre-packed matrix for GF2
    std::vector<uint64_t> matrix_A_packed_;  ///< Packed bits (only for GF2)
    bool use_packed_;  ///< Whether to use packed representation

public:
    LPNCore(const PCFParameters& params);

    /// Expand short seed to dense output: y = A·s + e
    /// Section 3.2: PRG_R(r) = A·s + e, s = H·ē
    std::vector<R> expand(
        const std::vector<R>& short_secret,  ///< s ∈ R^n
        const RegularNoise<R>& noise         ///< e ∈ R^N
    ) const;

    /// Expand for a specific range [start, end) - for tiled evaluation
    std::vector<R> expand_range(
        const std::vector<R>& short_secret,
        const RegularNoise<R>& noise,
        size_t start,
        size_t end
    ) const;

    /// Dense matrix-vector multiplication: y = A·s
    std::vector<R> dense_mul(const std::vector<R>& s) const;

    /// Dense matrix-vector multiplication for range [start, end)
    std::vector<R> dense_mul_range(const std::vector<R>& s, size_t start, size_t end) const;

    // Accessors
    size_t batch_size() const { return N_; }
    size_t primal_dim() const { return n_; }
    const PCFParameters& params() const { return params_; }
};

// ============================================================================
// Tiled Evaluation Strategy (Section 3.3)
// ============================================================================

/// Tile-based evaluation for cache locality
template<Ring R>
class TiledEvaluator {
private:
    PCFParameters params_;
    size_t tile_size_;

public:
    explicit TiledEvaluator(const PCFParameters& params)
        : params_(params), tile_size_(params.tile_size) {}

    /// Map absolute index ℓ to (tile_id, offset)
    struct TileMapping {
        size_t tile_id;
        size_t offset;
    };

    TileMapping map_index(CorrelationIndex ell) const {
        return {ell / tile_size_, ell % tile_size_};
    }

    /// Evaluate a single tile: returns correlation for indices [g*B, (g+1)*B)
    FourTermResult<R> evaluate_tile(
        size_t tile_id,
        const LPNCore<R>& lpn_core,
        const CryptoHandle& handle,
        const RuntimeContext& ctx,
        PartyId party
    ) const;
};

// ============================================================================
// PCF Handle (Section 2.1 - Immutable)
// ============================================================================

/// PCF correlation function - main interface
template<Ring R>
class PCFHandle {
private:
    std::unique_ptr<LPNCore<R>> lpn_core_;
    std::unique_ptr<TiledEvaluator<R>> tiled_eval_;
    CryptoHandle crypto_handle_;

public:
    /// Setup phase (Section 2.1, API Definition)
    /// Generates immutable handle and initial epoch
    static std::pair<PCFHandle<R>, RuntimeContext> setup(
        const PCFParameters& params,
        SessionId sid,
        const std::vector<uint8_t>& root_seed
    );

    /// Eval phase (Section 2.1, API Definition)
    /// Deterministic evaluation: Eval(K_i, ε, ℓ) → y_i(ℓ)
    OLECorrelation<R> eval(
        CorrelationIndex ell,
        const RuntimeContext& ctx,
        PartyId party
    ) const;

    /// Batch evaluation for multiple indices
    std::vector<OLECorrelation<R>> eval_batch(
        std::span<const CorrelationIndex> indices,
        const RuntimeContext& ctx,
        PartyId party
    ) const;

    // Accessors
    const CryptoHandle& handle() const { return crypto_handle_; }
    const PCFParameters& params() const { return crypto_handle_.params; }
};

// ============================================================================
// Four-Term Computation (Section 3.2, Equation 2)
// ============================================================================

/// Compute all four terms of the OLE decomposition
/// Returns additive share of y0 ⊙ y1
template<Ring R>
class FourTermComputer {
public:
    /// Compute Term 1: (As0) ⊙ (As1) - Dense-Dense (amortized via Setup)
    /// This is the main compute cost, optimized via tiling
    static std::vector<R> compute_term1_dense_dense(
        const std::vector<R>& As0,
        const std::vector<R>& As1
    );

    /// Compute Term 2: e0 ⊙ (As1) - Sparse-Dense
    /// Only w positions are non-zero, use selective delivery
    static std::vector<R> compute_term2_sparse_dense(
        const RegularNoise<R>& e0,
        const std::vector<R>& As1
    );

    /// Compute Term 3: e1 ⊙ (As0) - Sparse-Dense
    static std::vector<R> compute_term3_sparse_dense(
        const RegularNoise<R>& e1,
        const std::vector<R>& As0
    );

    /// Compute Term 4: e0 ⊙ e1 - Sparse-Sparse (collision handling)
    /// Automatically handled by selective delivery (Section 3.2, 4.1)
    /// When delivering As1[i] + e1[i] at positions in supp(e0),
    /// collision contribution e0[i]·e1[i] is naturally included
    static std::vector<R> compute_term4_sparse_sparse(
        const RegularNoise<R>& e0,
        const RegularNoise<R>& e1
    );

    /// Combine all four terms into final OLE share
    static OLECorrelation<R> combine_terms(
        const FourTermResult<R>& terms
    );
};

// ============================================================================
// Derivation and PRF/PRG Interface (Section C.1)
// ============================================================================

/// Domain-separated PRF/KDF for per-query material
class DerivationEngine {
public:
    /// Derive per-term seed using domain separation
    /// ctx includes (sid, ε, ℓ, party, termID, tileID)
    static std::vector<uint8_t> derive_term_seed(
        const std::vector<uint8_t>& root_key,
        const DerivationContext& ctx
    );

    /// Expand seed to regular noise parameters (support + values)
    template<Ring R>
    static RegularNoise<R> expand_to_noise(
        const std::vector<uint8_t>& seed,
        size_t N, size_t w
    );

    /// Expand seed to short secret s ∈ R^n
    template<Ring R>
    static std::vector<R> expand_to_short_secret(
        const std::vector<uint8_t>& seed,
        size_t n
    );
};

} // namespace silentmpc::pcf
