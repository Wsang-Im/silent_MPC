#include "silentmpc/selective_delivery/point_share.hpp"
#include "silentmpc/selective_delivery/multi_point_dpf.hpp"
#include "silentmpc/core/types.hpp"
#include "silentmpc/algebra/gf2.hpp"
#include "silentmpc/algebra/z2k.hpp"
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <openssl/aes.h>
#include <openssl/sha.h>
#include <cstring>
#include <algorithm>

namespace silentmpc::selective_delivery {

// ============================================================================
// DPF-Centric Backend Implementation (Section 4.3)
// ============================================================================

/// Distributed Point Function (DPF) implementation
/// Based on BGI14/BGI15 construction with AES-NI acceleration
template<Ring R>
class DPFEngine {
public:
    static constexpr size_t SEED_SIZE = 16;  // 128-bit AES block

    /// DPF key for one party
    struct DPFKey {
        std::vector<std::array<uint8_t, SEED_SIZE>> seeds;  // Per-level seeds
        std::vector<bool> control_bits;                      // Per-level control
        R correction_value;                                   // Final correction
        size_t domain_size;                                  // 2^depth
    };

    /// AES-NI based PRG: G(seed) -> (left_seed, right_seed, control_bit)
    static void prg_expand(
        const std::array<uint8_t, SEED_SIZE>& seed,
        std::array<uint8_t, SEED_SIZE>& left,
        std::array<uint8_t, SEED_SIZE>& right,
        bool& control
    ) {
        // Use AES in CTR mode for expansion
        EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
        EVP_EncryptInit_ex(ctx, EVP_aes_128_ecb(), nullptr, seed.data(), nullptr);

        // Generate left child
        uint8_t counter[16] = {0};
        counter[0] = 0x00;
        int outlen;
        EVP_EncryptUpdate(ctx, left.data(), &outlen, counter, 16);

        // Generate right child
        counter[0] = 0x01;
        EVP_EncryptUpdate(ctx, right.data(), &outlen, counter, 16);

        // Control bit from last bit of left seed
        control = (left[15] & 1);

        EVP_CIPHER_CTX_free(ctx);
    }

    /// XOR two seeds
    static std::array<uint8_t, SEED_SIZE> xor_seeds(
        const std::array<uint8_t, SEED_SIZE>& a,
        const std::array<uint8_t, SEED_SIZE>& b
    ) {
        std::array<uint8_t, SEED_SIZE> result;
        for (size_t i = 0; i < SEED_SIZE; ++i) {
            result[i] = a[i] ^ b[i];
        }
        return result;
    }

    /// Generate DPF keys for point function f(alpha) = beta, f(x) = 0 otherwise
    static std::pair<DPFKey, DPFKey> generate_keys(
        size_t alpha,
        const R& beta,
        size_t domain_bits
    ) {
        size_t domain_size = 1ULL << domain_bits;

        if (alpha >= domain_size) {
            throw std::invalid_argument("alpha must be < domain_size");
        }

        DPFKey key0, key1;
        key0.domain_size = key1.domain_size = domain_size;
        key0.seeds.reserve(domain_bits + 1);
        key1.seeds.reserve(domain_bits + 1);
        key0.control_bits.reserve(domain_bits);
        key1.control_bits.reserve(domain_bits);

        // Root seeds (random)
        std::array<uint8_t, SEED_SIZE> seed0, seed1;
        RAND_bytes(seed0.data(), SEED_SIZE);
        RAND_bytes(seed1.data(), SEED_SIZE);

        key0.seeds.push_back(seed0);
        key1.seeds.push_back(seed1);

        // Initial control bits
        bool t0 = false;
        bool t1 = true;

        std::array<uint8_t, SEED_SIZE> s0 = seed0;
        std::array<uint8_t, SEED_SIZE> s1 = seed1;

        // Traverse tree from root to leaf
        for (size_t level = 0; level < domain_bits; ++level) {
            // Current bit of alpha
            bool alpha_bit = (alpha >> (domain_bits - 1 - level)) & 1;

            // Expand both seeds
            std::array<uint8_t, SEED_SIZE> s0_left, s0_right;
            std::array<uint8_t, SEED_SIZE> s1_left, s1_right;
            bool t0_left, t0_right, t1_left, t1_right;

            prg_expand(s0, s0_left, s0_right, t0_left);
            t0_right = t0_left ^ true;  // Flip for right
            prg_expand(s1, s1_left, s1_right, t1_left);
            t1_right = t1_left ^ true;

            // Correction words
            std::array<uint8_t, SEED_SIZE> s_keep, s_lose;
            bool cw_control;

            if (!alpha_bit) {
                // Keep left, lose right
                s_lose = xor_seeds(s0_right, s1_right);
                s_keep = xor_seeds(s0_left, s1_left);
                cw_control = t0_left ^ t1_left ^ alpha_bit ^ true;
            } else {
                // Keep right, lose left
                s_lose = xor_seeds(s0_left, s1_left);
                s_keep = xor_seeds(s0_right, s1_right);
                cw_control = t0_right ^ t1_right ^ alpha_bit ^ true;
            }

            // Store correction
            key0.seeds.push_back(s_keep);
            key1.seeds.push_back(s_keep);  // Same correction for both
            key0.control_bits.push_back(cw_control);
            key1.control_bits.push_back(cw_control);

            // Update for next level
            if (!alpha_bit) {
                s0 = t0 ? xor_seeds(s0_left, s_keep) : s0_left;
                s1 = t1 ? xor_seeds(s1_left, s_keep) : s1_left;
                t0 = t0 ? (t0_left ^ cw_control) : t0_left;
                t1 = t1 ? (t1_left ^ cw_control) : t1_left;
            } else {
                s0 = t0 ? xor_seeds(s0_right, s_keep) : s0_right;
                s1 = t1 ? xor_seeds(s1_right, s_keep) : s1_right;
                t0 = t0 ? (t0_right ^ cw_control) : t0_right;
                t1 = t1 ? (t1_right ^ cw_control) : t1_right;
            }
        }

        // Final correction value
        // Convert final seed to ring element and apply beta
        uint64_t s0_val = 0, s1_val = 0;
        for (size_t i = 0; i < 8 && i < SEED_SIZE; ++i) {
            s0_val |= static_cast<uint64_t>(s0[i]) << (i * 8);
            s1_val |= static_cast<uint64_t>(s1[i]) << (i * 8);
        }

        // Correction: beta - ((-1)^t0 * s0 + (-1)^t1 * s1)
        R r0 = R::from_uint64(s0_val);
        R r1 = R::from_uint64(s1_val);

        // Apply control bit corrections
        // In GF2, negation is identity; in Z2k, use subtraction from zero
        if (t0) r0 = R(R::zero() - r0);
        if (!t1) r1 = R(R::zero() - r1);

        key0.correction_value = R(beta - (r0 + r1));
        key1.correction_value = key0.correction_value;

        return {key0, key1};
    }

    /// Evaluate DPF at all points in domain
    static std::vector<R> full_domain_eval(
        const DPFKey& key,
        bool party  // false = party 0, true = party 1
    ) {
        size_t domain_size = key.domain_size;
        std::vector<R> output(domain_size, R::zero());

        // Start with root
        std::vector<std::array<uint8_t, SEED_SIZE>> current_level;
        std::vector<bool> current_controls;

        current_level.push_back(key.seeds[0]);
        current_controls.push_back(party);  // t0 = 0, t1 = 1

        size_t depth = key.seeds.size() - 1;

        // Expand level by level
        for (size_t level = 0; level < depth; ++level) {
            std::vector<std::array<uint8_t, SEED_SIZE>> next_level;
            std::vector<bool> next_controls;

            next_level.reserve(current_level.size() * 2);
            next_controls.reserve(current_level.size() * 2);

            const auto& cw_seed = key.seeds[level + 1];
            bool cw_control = key.control_bits[level];

            for (size_t i = 0; i < current_level.size(); ++i) {
                std::array<uint8_t, SEED_SIZE> left, right;
                bool t_left, t_right;

                prg_expand(current_level[i], left, right, t_left);
                t_right = !t_left;

                bool t_current = current_controls[i];

                // Apply correction if needed
                if (t_current) {
                    left = xor_seeds(left, cw_seed);
                    right = xor_seeds(right, cw_seed);
                    t_left ^= cw_control;
                    t_right ^= cw_control;
                }

                next_level.push_back(left);
                next_level.push_back(right);
                next_controls.push_back(t_left);
                next_controls.push_back(t_right);
            }

            current_level = std::move(next_level);
            current_controls = std::move(next_controls);
        }

        // Convert final seeds to ring elements
        for (size_t i = 0; i < domain_size; ++i) {
            uint64_t seed_val = 0;
            for (size_t j = 0; j < 8 && j < SEED_SIZE; ++j) {
                seed_val |= static_cast<uint64_t>(current_level[i][j]) << (j * 8);
            }

            R value = R::from_uint64(seed_val);

            // Apply sign based on control bit
            if (current_controls[i]) {
                output[i] = R(R::zero() - value);
            } else {
                output[i] = value;
            }

            // Add correction at special point
            if (current_controls[i] != party) {
                output[i] = output[i] + key.correction_value;
            }
        }

        return output;
    }
};

// ============================================================================
// DPFCentricBackend Template Implementation
// ============================================================================

template<Ring R>
typename PointShareBackend<R>::SetupResult DPFCentricBackend<R>::setup(
    const std::vector<size_t>& receiver_indices,
    size_t tile_size,
    const SessionId& sid,
    EpochId epoch,
    size_t tile_id
) {
    // For each index, generate DPF keys for point function
    // This is simplified - real implementation would batch multiple points

    if (receiver_indices.empty()) {
        throw std::invalid_argument("receiver_indices cannot be empty");
    }

    // Determine domain size (round up to power of 2)
    size_t domain_bits = 0;
    size_t temp = tile_size;
    while (temp > 1) {
        domain_bits++;
        temp >>= 1;
    }
    if ((1ULL << domain_bits) < tile_size) {
        domain_bits++;
    }

    typename PointShareBackend<R>::SetupResult result;

    // Multi-point DPF batching optimization (Section 4.3)
    // Instead of |I| separate DPFs, use a batched approach:
    // 1. Group indices by common prefix (tree sharing)
    // 2. Generate shared tree up to divergence points
    // 3. Only branch where necessary

    std::vector<uint8_t> sender_state_data;
    std::vector<uint8_t> receiver_state_data;

    // Strategy: For small |I| (≤ 8), use individual DPFs
    // For larger batches, use multi-point optimization
    const size_t BATCH_THRESHOLD = 8;

    if (receiver_indices.size() <= BATCH_THRESHOLD) {
        // Individual DPF approach (simpler, good for small batches)
        for (size_t idx : receiver_indices) {
            // Generate DPF keys for point function at this index
            // Derive beta from (sid, epoch, tile_id, idx)
            DerivationContext ctx{sid, epoch, idx, 0, TermId::Term1_DenseDense, tile_id};
            auto ctx_bytes = ctx.serialize();

            // Hash to get beta value
            uint8_t hash[32];
            SHA256(ctx_bytes.data(), ctx_bytes.size(), hash);

            uint64_t beta_val = 0;
            for (size_t i = 0; i < 8; ++i) {
                beta_val |= static_cast<uint64_t>(hash[i]) << (i * 8);
            }
            R beta = R::from_uint64(beta_val);

            auto [key0, key1] = DPFEngine<R>::generate_keys(idx, beta, domain_bits);

            // Serialize key0 (sender gets this)
            // Format: [domain_size(8) || num_seeds(8) || seeds... || control_bits...]

            // Store domain size
            for (size_t i = 0; i < 8; ++i) {
                sender_state_data.push_back((key0.domain_size >> (i * 8)) & 0xFF);
            }

            // Store number of seeds
            size_t num_seeds = key0.seeds.size();
            for (size_t i = 0; i < 8; ++i) {
                sender_state_data.push_back((num_seeds >> (i * 8)) & 0xFF);
            }

            // Store seeds
            for (const auto& seed : key0.seeds) {
                sender_state_data.insert(sender_state_data.end(), seed.begin(), seed.end());
            }

            // Store control bits (packed)
            for (size_t i = 0; i < key0.control_bits.size(); ++i) {
                if (i % 8 == 0) {
                    sender_state_data.push_back(0);
                }
                if (key0.control_bits[i]) {
                    sender_state_data.back() |= (1 << (i % 8));
                }
            }

            // Store correction value (8 bytes)
            uint64_t correction = beta.to_uint64();  // Simplified
            for (size_t i = 0; i < 8; ++i) {
                sender_state_data.push_back((correction >> (i * 8)) & 0xFF);
            }

            // Similarly for key1 (receiver gets this)
            // In practice, keys have same structure, just different initial seed
            for (size_t i = 0; i < 8; ++i) {
                receiver_state_data.push_back((key1.domain_size >> (i * 8)) & 0xFF);
            }
            for (size_t i = 0; i < 8; ++i) {
                receiver_state_data.push_back((num_seeds >> (i * 8)) & 0xFF);
            }
            for (const auto& seed : key1.seeds) {
                receiver_state_data.insert(receiver_state_data.end(), seed.begin(), seed.end());
            }
            for (size_t i = 0; i < key1.control_bits.size(); ++i) {
                if (i % 8 == 0) {
                    receiver_state_data.push_back(0);
                }
                if (key1.control_bits[i]) {
                    receiver_state_data.back() |= (1 << (i % 8));
                }
            }
            for (size_t i = 0; i < 8; ++i) {
                receiver_state_data.push_back((correction >> (i * 8)) & 0xFF);
            }
        }  // end for each index
    } else {
        // Full tree-sharing multi-point DPF compression
        // Uses actual shared tree structure with branching points

        // Derive beta values for each index
        std::vector<R> beta_values;
        beta_values.reserve(receiver_indices.size());

        for (size_t idx : receiver_indices) {
            DerivationContext ctx{sid, epoch, idx, 0, TermId::Term1_DenseDense, tile_id};
            auto ctx_bytes = ctx.serialize();

            uint8_t hash[32];
            SHA256(ctx_bytes.data(), ctx_bytes.size(), hash);

            uint64_t beta_val = 0;
            for (size_t i = 0; i < 8; ++i) {
                beta_val |= static_cast<uint64_t>(hash[i]) << (i * 8);
            }
            beta_values.push_back(R::from_uint64(beta_val));
        }

        // Generate compressed multi-point DPF keys with full tree sharing
        auto [compressed_key0, compressed_key1] =
            MultiPointDPFGenerator<R>::generate_compressed_keys(
                receiver_indices,
                beta_values,
                domain_bits
            );

        // Serialize compressed keys
        // Format: [num_nodes(8)] [nodes...] [num_branches(8)] [branches...]
        //         [num_corrections(8)] [corrections...]

        // Serialize key0 (sender)
        size_t num_nodes0 = compressed_key0.shared_nodes.size();
        for (size_t i = 0; i < 8; ++i) {
            sender_state_data.push_back((num_nodes0 >> (i * 8)) & 0xFF);
        }

        for (const auto& node : compressed_key0.shared_nodes) {
            // Serialize node: seed(16) + control_bit(1) + depth(4) + path(8)
            sender_state_data.insert(sender_state_data.end(),
                                    node.seed.begin(), node.seed.end());
            sender_state_data.push_back(node.control_bit ? 1 : 0);

            uint32_t depth32 = static_cast<uint32_t>(node.depth);
            for (size_t i = 0; i < 4; ++i) {
                sender_state_data.push_back((depth32 >> (i * 8)) & 0xFF);
            }

            for (size_t i = 0; i < 8; ++i) {
                sender_state_data.push_back((node.path >> (i * 8)) & 0xFF);
            }
        }

        // Serialize branch map
        size_t num_branches = compressed_key0.branch_map.size();
        for (size_t i = 0; i < 8; ++i) {
            sender_state_data.push_back((num_branches >> (i * 8)) & 0xFF);
        }

        for (const auto& [key, value] : compressed_key0.branch_map) {
            auto [depth, path] = key;
            auto [left_idx, right_idx] = value;

            for (size_t i = 0; i < 8; ++i) {
                sender_state_data.push_back((depth >> (i * 8)) & 0xFF);
            }
            for (size_t i = 0; i < 8; ++i) {
                sender_state_data.push_back((path >> (i * 8)) & 0xFF);
            }
            for (size_t i = 0; i < 8; ++i) {
                sender_state_data.push_back((left_idx >> (i * 8)) & 0xFF);
            }
            for (size_t i = 0; i < 8; ++i) {
                sender_state_data.push_back((right_idx >> (i * 8)) & 0xFF);
            }
        }

        // Serialize leaf corrections
        size_t num_corrections = compressed_key0.leaf_corrections.size();
        for (size_t i = 0; i < 8; ++i) {
            sender_state_data.push_back((num_corrections >> (i * 8)) & 0xFF);
        }

        for (const auto& [idx, correction_bytes] : compressed_key0.leaf_corrections) {
            for (size_t i = 0; i < 8; ++i) {
                sender_state_data.push_back((idx >> (i * 8)) & 0xFF);
            }

            size_t corr_size = correction_bytes.size();
            for (size_t i = 0; i < 8; ++i) {
                sender_state_data.push_back((corr_size >> (i * 8)) & 0xFF);
            }

            sender_state_data.insert(sender_state_data.end(),
                                    correction_bytes.begin(), correction_bytes.end());
        }

        // Same for key1 (receiver)
        size_t num_nodes1 = compressed_key1.shared_nodes.size();
        for (size_t i = 0; i < 8; ++i) {
            receiver_state_data.push_back((num_nodes1 >> (i * 8)) & 0xFF);
        }

        for (const auto& node : compressed_key1.shared_nodes) {
            receiver_state_data.insert(receiver_state_data.end(),
                                      node.seed.begin(), node.seed.end());
            receiver_state_data.push_back(node.control_bit ? 1 : 0);

            uint32_t depth32 = static_cast<uint32_t>(node.depth);
            for (size_t i = 0; i < 4; ++i) {
                receiver_state_data.push_back((depth32 >> (i * 8)) & 0xFF);
            }

            for (size_t i = 0; i < 8; ++i) {
                receiver_state_data.push_back((node.path >> (i * 8)) & 0xFF);
            }
        }

        num_branches = compressed_key1.branch_map.size();
        for (size_t i = 0; i < 8; ++i) {
            receiver_state_data.push_back((num_branches >> (i * 8)) & 0xFF);
        }

        for (const auto& [key, value] : compressed_key1.branch_map) {
            auto [depth, path] = key;
            auto [left_idx, right_idx] = value;

            for (size_t i = 0; i < 8; ++i) {
                receiver_state_data.push_back((depth >> (i * 8)) & 0xFF);
            }
            for (size_t i = 0; i < 8; ++i) {
                receiver_state_data.push_back((path >> (i * 8)) & 0xFF);
            }
            for (size_t i = 0; i < 8; ++i) {
                receiver_state_data.push_back((left_idx >> (i * 8)) & 0xFF);
            }
            for (size_t i = 0; i < 8; ++i) {
                receiver_state_data.push_back((right_idx >> (i * 8)) & 0xFF);
            }
        }

        num_corrections = compressed_key1.leaf_corrections.size();
        for (size_t i = 0; i < 8; ++i) {
            receiver_state_data.push_back((num_corrections >> (i * 8)) & 0xFF);
        }

        for (const auto& [idx, correction_bytes] : compressed_key1.leaf_corrections) {
            for (size_t i = 0; i < 8; ++i) {
                receiver_state_data.push_back((idx >> (i * 8)) & 0xFF);
            }

            size_t corr_size = correction_bytes.size();
            for (size_t i = 0; i < 8; ++i) {
                receiver_state_data.push_back((corr_size >> (i * 8)) & 0xFF);
            }

            receiver_state_data.insert(receiver_state_data.end(),
                                      correction_bytes.begin(), correction_bytes.end());
        }

        // Log compression ratio
        double compression_ratio = MultiPointDPFGenerator<R>::get_compression_ratio(compressed_key0);
        // In production, this would be logged for monitoring
        (void)compression_ratio;  // Suppress unused warning
    }

    result.sender_state = std::move(sender_state_data);
    result.receiver_state = std::move(receiver_state_data);

    // Total state distribution cost (sender_state + receiver_state)
    // This is what needs to be transferred to distribute setup to both parties
    // Note: DPF setup is non-interactive (no protocol messages), but states still need distribution
    result.communication_bytes = result.sender_state.size() + result.receiver_state.size();

    return result;
}

template<Ring R>
typename PointShareBackend<R>::ReconstructResult DPFCentricBackend<R>::reconstruct(
    const std::vector<uint8_t>& state,
    const std::vector<R>& dense_vector,
    bool is_sender
) {
    typename PointShareBackend<R>::ReconstructResult result;

    // Parse DPF key from state and evaluate over dense_vector
    size_t offset = 0;

    if (state.size() < 16) {
        return result;  // Empty or invalid state
    }

    // Parse domain size
    size_t domain_size = 0;
    for (size_t i = 0; i < 8; ++i) {
        domain_size |= static_cast<size_t>(state[offset + i]) << (i * 8);
    }
    offset += 8;

    // Parse number of seeds
    size_t num_seeds = 0;
    for (size_t i = 0; i < 8; ++i) {
        num_seeds |= static_cast<size_t>(state[offset + i]) << (i * 8);
    }
    offset += 8;

    if (num_seeds == 0 || offset + num_seeds * 16 > state.size()) {
        return result;  // Invalid
    }

    // Parse DPF key
    typename DPFEngine<R>::DPFKey key;
    key.domain_size = domain_size;

    // Parse seeds
    for (size_t i = 0; i < num_seeds; ++i) {
        std::array<uint8_t, 16> seed;
        std::memcpy(seed.data(), state.data() + offset, 16);
        key.seeds.push_back(seed);
        offset += 16;
    }

    // Parse control bits
    size_t num_control_bits = num_seeds - 1;
    size_t control_bytes = (num_control_bits + 7) / 8;
    if (offset + control_bytes > state.size()) {
        return result;
    }

    for (size_t i = 0; i < num_control_bits; ++i) {
        size_t byte_idx = i / 8;
        size_t bit_idx = i % 8;
        bool bit = (state[offset + byte_idx] >> bit_idx) & 1;
        key.control_bits.push_back(bit);
    }
    offset += control_bytes;

    // Parse correction value
    if (offset + 8 > state.size()) {
        return result;
    }

    uint64_t correction_val = 0;
    for (size_t i = 0; i < 8; ++i) {
        correction_val |= static_cast<uint64_t>(state[offset + i]) << (i * 8);
    }
    key.correction_value = R::from_uint64(correction_val);

    // Evaluate DPF over full domain
    auto dpf_output = DPFEngine<R>::full_domain_eval(key, is_sender);

    // Compute inner product with dense_vector
    R sum = R::zero();
    size_t eval_size = std::min(dpf_output.size(), dense_vector.size());
    for (size_t i = 0; i < eval_size; ++i) {
        sum = sum + (dpf_output[i] * dense_vector[i]);
    }

    result.recovered_values.push_back(sum);
    result.communication_bytes = state.size();

    return result;
}

// Explicit template instantiations
template class DPFCentricBackend<algebra::Z2k<32>>;
template class DPFCentricBackend<algebra::Z2k<64>>;
template class DPFCentricBackend<algebra::GF2>;

} // namespace silentmpc::selective_delivery
