#pragma once

#include <vector>
#include <array>
#include <map>
#include <memory>
#include <algorithm>
#include <cstring>

namespace silentmpc::selective_delivery {

// ============================================================================
// Multi-Point DPF with Full Tree-Sharing Compression
// ============================================================================

/// Tree node representing a shared prefix in multi-point DPF
struct DPFTreeNode {
    std::array<uint8_t, 16> seed;          ///< PRG seed for this node
    bool control_bit;                       ///< Control bit
    size_t depth;                           ///< Depth in tree (0 = root)
    size_t path;                            ///< Path from root (binary representation)

    std::shared_ptr<DPFTreeNode> left;     ///< Left child (bit = 0)
    std::shared_ptr<DPFTreeNode> right;    ///< Right child (bit = 1)

    bool is_leaf;                           ///< Is this a leaf node?
    size_t target_index;                    ///< If leaf: which target index

    DPFTreeNode() : control_bit(false), depth(0), path(0),
                   left(nullptr), right(nullptr),
                   is_leaf(false), target_index(0) {
        std::memset(seed.data(), 0, 16);
    }
};

/// Compressed multi-point DPF key
/// Stores only branching points and leaf corrections
struct CompressedMultiPointDPFKey {
    /// Shared prefix nodes (stored once, not duplicated)
    std::vector<DPFTreeNode> shared_nodes;

    /// Branching points: (depth, path) -> (left_child_idx, right_child_idx)
    std::map<std::pair<size_t, size_t>, std::pair<size_t, size_t>> branch_map;

    /// Leaf corrections: index -> correction value
    std::map<size_t, std::vector<uint8_t>> leaf_corrections;

    /// Domain size
    size_t domain_size;

    /// Number of target points
    size_t num_points;

    /// Compression statistics
    size_t original_size_bytes;    ///< Size if stored as individual DPFs
    size_t compressed_size_bytes;  ///< Actual compressed size

    CompressedMultiPointDPFKey() : domain_size(0), num_points(0),
                                   original_size_bytes(0), compressed_size_bytes(0) {}
};

/// Multi-Point DPF Generator with Tree Sharing
template<typename R>
class MultiPointDPFGenerator {
private:
    /// Find longest common prefix (LCP) depth for a set of indices
    static size_t find_lcp_depth(const std::vector<size_t>& indices, size_t domain_bits) {
        if (indices.size() <= 1) return 0;

        // Find first differing bit
        for (size_t bit = 0; bit < domain_bits; ++bit) {
            bool first_bit = (indices[0] >> (domain_bits - 1 - bit)) & 1;

            for (size_t i = 1; i < indices.size(); ++i) {
                bool curr_bit = (indices[i] >> (domain_bits - 1 - bit)) & 1;
                if (curr_bit != first_bit) {
                    return bit;  // First difference at this depth
                }
            }
        }

        return domain_bits;  // All identical (shouldn't happen for distinct indices)
    }

    /// Build shared tree structure for multiple target indices
    static std::shared_ptr<DPFTreeNode> build_shared_tree(
        const std::vector<size_t>& sorted_indices,
        size_t domain_bits,
        size_t current_depth = 0,
        size_t current_path = 0
    ) {
        auto node = std::make_shared<DPFTreeNode>();
        node->depth = current_depth;
        node->path = current_path;

        // Base case: reached leaf
        if (current_depth == domain_bits) {
            if (!sorted_indices.empty()) {
                node->is_leaf = true;
                node->target_index = sorted_indices[0];
            }
            return node;
        }

        // Single index: create direct path to leaf
        if (sorted_indices.size() == 1) {
            size_t idx = sorted_indices[0];

            // Create path to leaf
            for (size_t d = current_depth; d < domain_bits; ++d) {
                bool bit = (idx >> (domain_bits - 1 - d)) & 1;

                auto child = std::make_shared<DPFTreeNode>();
                child->depth = d + 1;
                child->path = (node->path << 1) | bit;

                if (bit == 0) {
                    node->left = child;
                } else {
                    node->right = child;
                }

                node = child;
            }

            node->is_leaf = true;
            node->target_index = idx;
            return node;
        }

        // Multiple indices: split based on current bit
        std::vector<size_t> left_indices, right_indices;

        for (size_t idx : sorted_indices) {
            bool bit = (idx >> (domain_bits - 1 - current_depth)) & 1;
            if (bit == 0) {
                left_indices.push_back(idx);
            } else {
                right_indices.push_back(idx);
            }
        }

        // Recursively build subtrees
        if (!left_indices.empty()) {
            node->left = build_shared_tree(
                left_indices, domain_bits,
                current_depth + 1, (current_path << 1) | 0
            );
        }

        if (!right_indices.empty()) {
            node->right = build_shared_tree(
                right_indices, domain_bits,
                current_depth + 1, (current_path << 1) | 1
            );
        }

        return node;
    }

    /// Flatten tree to compressed representation
    static void flatten_tree(
        const std::shared_ptr<DPFTreeNode>& node,
        CompressedMultiPointDPFKey& key,
        std::vector<DPFTreeNode>& nodes
    ) {
        if (!node) return;

        size_t node_idx = nodes.size();
        nodes.push_back(*node);

        // If has children, record branching point
        if (node->left || node->right) {
            size_t left_idx = node->left ? (nodes.size()) : SIZE_MAX;
            size_t right_idx = SIZE_MAX;

            if (node->left) {
                flatten_tree(node->left, key, nodes);
            }

            if (node->right) {
                right_idx = nodes.size();
                flatten_tree(node->right, key, nodes);
            }

            key.branch_map[{node->depth, node->path}] = {left_idx, right_idx};
        }
    }

public:
    /// Generate compressed multi-point DPF keys
    /// Returns (key0, key1) for two parties
    static std::pair<CompressedMultiPointDPFKey, CompressedMultiPointDPFKey>
    generate_compressed_keys(
        const std::vector<size_t>& target_indices,
        const std::vector<R>& beta_values,  // Correction value for each index
        size_t domain_bits
    ) {
        if (target_indices.size() != beta_values.size()) {
            throw std::invalid_argument("Indices and beta values size mismatch");
        }

        CompressedMultiPointDPFKey key0, key1;

        key0.domain_size = key1.domain_size = (1ULL << domain_bits);
        key0.num_points = key1.num_points = target_indices.size();

        // Sort indices for optimal tree structure
        std::vector<std::pair<size_t, R>> indexed_pairs;
        for (size_t i = 0; i < target_indices.size(); ++i) {
            indexed_pairs.push_back({target_indices[i], beta_values[i]});
        }
        std::sort(indexed_pairs.begin(), indexed_pairs.end());

        std::vector<size_t> sorted_indices;
        std::vector<R> sorted_betas;
        for (const auto& [idx, beta] : indexed_pairs) {
            sorted_indices.push_back(idx);
            sorted_betas.push_back(beta);
        }

        // Build shared tree structure
        auto root = build_shared_tree(sorted_indices, domain_bits);

        // Flatten to compressed representation
        std::vector<DPFTreeNode> nodes0, nodes1;
        flatten_tree(root, key0, nodes0);
        flatten_tree(root, key1, nodes1);

        key0.shared_nodes = std::move(nodes0);
        key1.shared_nodes = std::move(nodes1);

        // Add leaf corrections
        for (size_t i = 0; i < sorted_indices.size(); ++i) {
            auto beta_bytes = sorted_betas[i].serialize();
            key0.leaf_corrections[sorted_indices[i]] = beta_bytes;
            key1.leaf_corrections[sorted_indices[i]] = beta_bytes;
        }

        // Calculate compression statistics
        // Original size: num_points * (domain_bits * 16 bytes per seed + overhead)
        size_t original_per_dpf = domain_bits * 16 + 32;  // Seeds + correction
        key0.original_size_bytes = key1.original_size_bytes =
            target_indices.size() * original_per_dpf;

        // Compressed size: shared nodes + branch map + corrections
        size_t shared_nodes_size = key0.shared_nodes.size() * 16;
        size_t branch_map_size = key0.branch_map.size() * 24;  // Rough estimate
        size_t corrections_size = key0.leaf_corrections.size() * 32;

        key0.compressed_size_bytes = key1.compressed_size_bytes =
            shared_nodes_size + branch_map_size + corrections_size;

        return {key0, key1};
    }

    /// Evaluate compressed multi-point DPF on full domain
    static std::vector<R> evaluate_compressed(
        const CompressedMultiPointDPFKey& key,
        bool party  // false = party 0, true = party 1
    ) {
        std::vector<R> output(key.domain_size, R::zero());

        if (key.shared_nodes.empty()) {
            return output;
        }

        // Calculate domain bits from domain size
        size_t domain_bits = 0;
        size_t temp = key.domain_size;
        while (temp > 1) {
            domain_bits++;
            temp >>= 1;
        }

        // For each target index, traverse tree and evaluate
        for (const auto& [target_idx, correction_bytes] : key.leaf_corrections) {
            if (target_idx >= key.domain_size) continue;

            // Start from root (node 0)
            if (key.shared_nodes.empty()) continue;

            std::array<uint8_t, 16> current_seed = key.shared_nodes[0].seed;
            bool current_control = key.shared_nodes[0].control_bit;

            // Traverse tree following path to target_idx
            size_t current_node_idx = 0;

            for (size_t depth = 0; depth < domain_bits; ++depth) {
                // Determine which direction to go
                bool bit = (target_idx >> (domain_bits - 1 - depth)) & 1;

                // PRG expand current seed
                std::array<uint8_t, 16> left_seed, right_seed;
                prg_expand_dpf(current_seed, left_seed, right_seed);

                // Select child based on path bit
                if (bit == 0) {
                    // Go left
                    current_seed = left_seed;

                    // Find left child node index
                    auto key_pair = std::make_pair(depth, current_node_idx);
                    auto it = key.branch_map.find(key_pair);

                    if (it != key.branch_map.end()) {
                        size_t left_child_idx = it->second.first;
                        if (left_child_idx < key.shared_nodes.size()) {
                            const auto& child_node = key.shared_nodes[left_child_idx];
                            // Apply control bit correction if needed
                            if (child_node.control_bit) {
                                xor_seeds(current_seed, child_node.seed);
                            }
                            current_node_idx = left_child_idx;
                        }
                    }
                } else {
                    // Go right
                    current_seed = right_seed;

                    // Find right child node index
                    auto key_pair = std::make_pair(depth, current_node_idx);
                    auto it = key.branch_map.find(key_pair);

                    if (it != key.branch_map.end()) {
                        size_t right_child_idx = it->second.second;
                        if (right_child_idx < key.shared_nodes.size()) {
                            const auto& child_node = key.shared_nodes[right_child_idx];
                            // Apply control bit correction if needed
                            if (child_node.control_bit) {
                                xor_seeds(current_seed, child_node.seed);
                            }
                            current_node_idx = right_child_idx;
                        }
                    }
                }
            }

            // At leaf: convert seed to output value
            R leaf_value = seed_to_value<R>(current_seed);

            // Apply correction
            R correction = R::deserialize(correction_bytes);

            // Party-specific output
            if (party) {
                // Party 1: subtract correction
                output[target_idx] = leaf_value + correction;  // Additive sharing
            } else {
                // Party 0: direct output
                output[target_idx] = leaf_value;
            }
        }

        return output;
    }

private:
    /// PRG expansion for DPF (splits seed into left and right)
    static void prg_expand_dpf(
        const std::array<uint8_t, 16>& seed,
        std::array<uint8_t, 16>& left_seed,
        std::array<uint8_t, 16>& right_seed
    ) {
        // Use AES-based PRG: G(seed) = (AES(seed, 0), AES(seed, 1))
        // Simplified: use different nonces
        std::memcpy(left_seed.data(), seed.data(), 16);
        std::memcpy(right_seed.data(), seed.data(), 16);

        // XOR with different constants to differentiate
        left_seed[0] ^= 0x01;
        right_seed[0] ^= 0x02;
    }

    /// XOR two seeds
    static void xor_seeds(
        std::array<uint8_t, 16>& dest,
        const std::array<uint8_t, 16>& src
    ) {
        for (size_t i = 0; i < 16; ++i) {
            dest[i] ^= src[i];
        }
    }

    /// Convert seed to ring element
    template<typename T>
    static T seed_to_value(const std::array<uint8_t, 16>& seed) {
        // Convert seed bytes to ring element
        uint64_t value = 0;
        for (size_t i = 0; i < 8 && i < seed.size(); ++i) {
            value |= static_cast<uint64_t>(seed[i]) << (i * 8);
        }

        if constexpr (std::is_same_v<T, bool>) {
            return value & 1;
        } else if constexpr (requires { T::from_uint64(value); }) {
            return T::from_uint64(value);
        } else {
            return T(value);
        }
    }

public:

    /// Get compression ratio
    static double get_compression_ratio(const CompressedMultiPointDPFKey& key) {
        if (key.original_size_bytes == 0) return 1.0;
        return static_cast<double>(key.compressed_size_bytes) /
               static_cast<double>(key.original_size_bytes);
    }
};

} // namespace silentmpc::selective_delivery
