#pragma once

#include "silentmpc/core/types.hpp"
#include "silentmpc/pcf/pcf_core.hpp"
#include <span>
#include <vector>
#include <memory>

namespace silentmpc::selective_delivery {

// ============================================================================
// PointShare Abstraction (Section 4.1)
// ============================================================================

/// Abstract interface for selective delivery
/// Receiver learns {v[i]}_{i∈I} while hiding I from sender
template<Ring R>
class PointShareBackend {
public:
    virtual ~PointShareBackend() = default;

    /// Setup phase: prepare sender/receiver state for tile
    /// Interactive or one-shot depending on backend
    struct SetupResult {
        std::vector<uint8_t> sender_state;    ///< τ_S
        std::vector<uint8_t> receiver_state;  ///< τ_R
        size_t communication_bytes;           ///< For monitoring
    };

    virtual SetupResult setup(
        const std::vector<size_t>& receiver_indices,  ///< I ⊆ {0,...,B-1}
        size_t tile_size,                              ///< B
        const SessionId& sid,
        EpochId epoch,
        size_t tile_id
    ) = 0;

    /// Reconstruct phase: sender produces response, receiver recovers values
    struct ReconstructResult {
        std::vector<R> recovered_values;  ///< {v[i]}_{i∈I}
        size_t communication_bytes;
    };

    virtual ReconstructResult reconstruct(
        const std::vector<uint8_t>& state,
        const std::vector<R>& dense_vector,  ///< v ∈ R^B (sender's tile)
        bool is_sender
    ) = 0;

    /// Backend identifier
    virtual std::string name() const = 0;
};

// ============================================================================
// Backend A: OT-Centric (Section 4.2)
// ============================================================================

// Forward declaration for cached base OT
template<Ring R>
struct CachedBaseOT;

/// OT extension-based backend
/// Optimized for: low-latency LANs, high compute availability
template<Ring R>
class OTCentricBackend : public PointShareBackend<R> {
private:
    struct OTState {
        std::vector<std::array<uint8_t, 16>> ot_messages;  ///< OT outputs
        std::vector<size_t> choice_bits;                    ///< Receiver choices
    };

public:
    typename PointShareBackend<R>::SetupResult setup(
        const std::vector<size_t>& receiver_indices,
        size_t tile_size,
        const SessionId& sid,
        EpochId epoch,
        size_t tile_id
    ) override;

    /// Fast setup using cached base OT (standard practice in OT literature)
    /// Base OT can be pre-computed once and reused across multiple setup calls
    typename PointShareBackend<R>::SetupResult setup_fast(
        const std::vector<size_t>& receiver_indices,
        size_t tile_size,
        const SessionId& sid,
        EpochId epoch,
        size_t tile_id,
        const CachedBaseOT<R>& base_ot_cache
    );

    typename PointShareBackend<R>::ReconstructResult reconstruct(
        const std::vector<uint8_t>& state,
        const std::vector<R>& dense_vector,
        bool is_sender
    ) override;

    std::string name() const override { return "OT-Centric"; }

private:
    /// IKNP-style OT extension (Section 4.2)
    /// Batches w selections per tile
    void run_ot_extension(
        std::span<const size_t> indices,
        size_t domain_size,
        OTState& state
    );

    /// Encode ring element for OT transfer
    std::vector<uint8_t> encode_element(const R& elem) const;

    /// Decode ring element from OT output
    R decode_element(std::span<const uint8_t> data) const;
};

// ============================================================================
// Backend B: DPF/FSS-Centric (Section 4.3)
// ============================================================================

/// Distributed Point Function-based backend
/// Optimized for: high-latency WANs, asynchronous operation
template<Ring R>
class DPFCentricBackend : public PointShareBackend<R> {
private:
    struct DPFKey {
        std::vector<uint8_t> key_data;  ///< Compact DPF key
        size_t index;                    ///< Target index
        R value;                         ///< Target value (payload)
    };

public:
    typename PointShareBackend<R>::SetupResult setup(
        const std::vector<size_t>& receiver_indices,
        size_t tile_size,
        const SessionId& sid,
        EpochId epoch,
        size_t tile_id
    ) override;

    typename PointShareBackend<R>::ReconstructResult reconstruct(
        const std::vector<uint8_t>& state,
        const std::vector<R>& dense_vector,
        bool is_sender
    ) override;

    std::string name() const override { return "DPF-Centric"; }

private:
    /// Generate DPF key pair for point function
    /// f(i) = value if i == index, else 0
    std::pair<DPFKey, DPFKey> generate_dpf_keys(
        size_t index,
        const R& value,
        size_t domain_size
    );

    /// Evaluate DPF key on full domain
    /// Returns vector of size domain_size
    std::vector<R> evaluate_dpf_key(
        const DPFKey& key,
        size_t domain_size
    ) const;

    /// AES-NI accelerated PRG for DPF expansion
    void aes_prg_expand(
        std::span<const uint8_t> seed,
        std::span<uint8_t> output
    ) const;
};

// ============================================================================
// Backend Factory (Section 4.4)
// ============================================================================

/// Backend selection policy
enum class BackendPolicy {
    OT,       ///< Use OT-centric (recommended for LAN)
    DPF,      ///< Use DPF-centric (recommended for WAN)
    Auto      ///< Auto-select based on network conditions
};

/// Network condition hints for auto-selection
struct NetworkHints {
    size_t bandwidth_mbps;  ///< Estimated bandwidth
    size_t rtt_ms;          ///< Estimated RTT

    /// Heuristic: use DPF for high-latency or low-bandwidth
    BackendPolicy recommend() const {
        if (rtt_ms > 20 || bandwidth_mbps < 200) {
            return BackendPolicy::DPF;
        }
        return BackendPolicy::OT;
    }
};

/// Factory for creating backend instances
template<Ring R>
class BackendFactory {
public:
    static std::unique_ptr<PointShareBackend<R>> create(
        BackendPolicy policy,
        const NetworkHints& hints = {}
    ) {
        BackendPolicy effective_policy = policy;
        if (policy == BackendPolicy::Auto) {
            effective_policy = hints.recommend();
        }

        switch (effective_policy) {
            case BackendPolicy::OT:
                return std::make_unique<OTCentricBackend<R>>();
            case BackendPolicy::DPF:
                return std::make_unique<DPFCentricBackend<R>>();
            default:
                throw std::invalid_argument("Invalid backend policy");
        }
    }
};

// ============================================================================
// Collision-Inclusive Delivery (Section 4.1, clarification)
// ============================================================================

/// Helper to construct sender vector for Term 2/3 sparse corrections
/// Includes collision term automatically (Section 3.2)
template<Ring R>
class CollisionInclusiveHelper {
public:
    /// For Term 2: e0 ⊙ (As1 + e1)
    /// Sender vector is y1 = As1 + e1 (full value including e1)
    /// When delivered at i ∈ supp(e0), multiplication e0[i]·y1[i]
    /// automatically includes e0[i]·e1[i] collision term
    static std::vector<R> prepare_sender_vector_with_collisions(
        const std::vector<R>& As,
        const pcf::RegularNoise<R>& e
    ) {
        std::vector<R> result = As;
        // Add sparse noise: result[i] += e[i] for i ∈ supp(e)
        for (size_t k = 0; k < e.support().size(); ++k) {
            size_t i = e.support()[k];
            result[i] += e.values()[k];
        }
        return result;
    }

    /// Expected collision count for monitoring (Section 3.2)
    static double expected_collisions(size_t N, size_t w) {
        return static_cast<double>(w * w) / N;
    }
};

} // namespace silentmpc::selective_delivery
