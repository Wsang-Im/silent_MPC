#pragma once

#include "silentmpc/core/types.hpp"
#include "silentmpc/network/connection.hpp"
#include "silentmpc/selective_delivery/multi_point_dpf.hpp"
#include <vector>
#include <cstdint>
#include <span>
#include <memory>

namespace silentmpc::network {

// ============================================================================
// Message Types
// ============================================================================

enum class MessageType : uint8_t {
    // OT Messages
    OT_BaseSetup = 1,        ///< Base OT sender setup (A value)
    OT_ReceiverChoice = 2,   ///< Base OT receiver choice (B value)
    OT_SenderKeys = 3,       ///< Base OT sender derived keys
    OT_ExtensionSetup = 4,   ///< OT extension matrix
    OT_ExtensionResponse = 5,///< OT extension receiver response

    // DPF Messages
    DPF_Key = 10,            ///< DPF key transfer
    DPF_KeyCompressed = 11,  ///< Compressed multi-point DPF key

    // PCF Messages
    PCF_Setup = 20,          ///< PCF setup phase
    PCF_EvalRequest = 21,    ///< PCF evaluation request
    PCF_EvalResponse = 22,   ///< PCF evaluation response

    // Control Messages
    Handshake = 30,          ///< Initial handshake
    Ack = 31,                ///< Acknowledgment
    Error = 32,              ///< Error message
};

// ============================================================================
// Base Message
// ============================================================================

struct Message {
    MessageType type;
    std::vector<uint8_t> payload;

    Message() : type(MessageType::Handshake) {}
    Message(MessageType t, std::vector<uint8_t> p)
        : type(t), payload(std::move(p)) {}

    /// Serialize to bytes: [type (1 byte)] [payload]
    std::vector<uint8_t> serialize() const;

    /// Deserialize from bytes
    static Message deserialize(std::span<const uint8_t> data);

    /// Send via connection
    void send(TCPConnection& conn) const;

    /// Receive from connection
    static Message receive(TCPConnection& conn);
};

// ============================================================================
// OT Messages
// ============================================================================

/// Base OT Sender Setup Message (contains A = g^a)
struct OTBaseSetupMessage {
    std::vector<uint8_t> A;  ///< Elliptic curve point (sender's public value)

    Message to_message() const;
    static OTBaseSetupMessage from_message(const Message& msg);
};

/// Base OT Receiver Choice Message (contains B = A^b or (A/h)^b)
struct OTReceiverChoiceMessage {
    std::vector<uint8_t> B;  ///< Receiver's response
    bool choice;             ///< Receiver's choice bit

    Message to_message() const;
    static OTReceiverChoiceMessage from_message(const Message& msg);
};

/// Base OT Sender Keys Message (encrypted keys for both choices)
struct OTSenderKeysMessage {
    std::vector<uint8_t> key0;  ///< Key for choice = 0
    std::vector<uint8_t> key1;  ///< Key for choice = 1

    Message to_message() const;
    static OTSenderKeysMessage from_message(const Message& msg);
};

// ============================================================================
// DPF Messages
// ============================================================================

/// DPF Key Message
template<typename R>
struct DPFKeyMessage {
    selective_delivery::CompressedMultiPointDPFKey key;

    Message to_message() const {
        std::vector<uint8_t> payload;

        // Serialize domain_size (8 bytes)
        for (size_t i = 0; i < 8; ++i) {
            payload.push_back(static_cast<uint8_t>((key.domain_size >> (i * 8)) & 0xFF));
        }

        // Serialize num_points (8 bytes)
        for (size_t i = 0; i < 8; ++i) {
            payload.push_back(static_cast<uint8_t>((key.num_points >> (i * 8)) & 0xFF));
        }

        // Serialize shared_nodes
        size_t num_nodes = key.shared_nodes.size();
        for (size_t i = 0; i < 4; ++i) {
            payload.push_back(static_cast<uint8_t>((num_nodes >> (i * 8)) & 0xFF));
        }

        for (const auto& node : key.shared_nodes) {
            // Serialize seed (16 bytes)
            payload.insert(payload.end(), node.seed.begin(), node.seed.end());

            // Serialize control_bit (1 byte)
            payload.push_back(node.control_bit ? 1 : 0);

            // Serialize depth (8 bytes)
            for (size_t i = 0; i < 8; ++i) {
                payload.push_back(static_cast<uint8_t>((node.depth >> (i * 8)) & 0xFF));
            }

            // Serialize path (8 bytes)
            for (size_t i = 0; i < 8; ++i) {
                payload.push_back(static_cast<uint8_t>((node.path >> (i * 8)) & 0xFF));
            }

            // Serialize is_leaf (1 byte)
            payload.push_back(node.is_leaf ? 1 : 0);

            // Serialize target_index (8 bytes)
            for (size_t i = 0; i < 8; ++i) {
                payload.push_back(static_cast<uint8_t>((node.target_index >> (i * 8)) & 0xFF));
            }
        }

        // Serialize branch_map size (4 bytes)
        uint32_t branch_map_size = static_cast<uint32_t>(key.branch_map.size());
        for (size_t i = 0; i < 4; ++i) {
            payload.push_back(static_cast<uint8_t>((branch_map_size >> (i * 8)) & 0xFF));
        }

        // Serialize branch_map entries
        for (const auto& [key_pair, value_pair] : key.branch_map) {
            // Key: (depth, path)
            for (size_t i = 0; i < 8; ++i) {
                payload.push_back(static_cast<uint8_t>((key_pair.first >> (i * 8)) & 0xFF));
            }
            for (size_t i = 0; i < 8; ++i) {
                payload.push_back(static_cast<uint8_t>((key_pair.second >> (i * 8)) & 0xFF));
            }

            // Value: (left_idx, right_idx)
            for (size_t i = 0; i < 8; ++i) {
                payload.push_back(static_cast<uint8_t>((value_pair.first >> (i * 8)) & 0xFF));
            }
            for (size_t i = 0; i < 8; ++i) {
                payload.push_back(static_cast<uint8_t>((value_pair.second >> (i * 8)) & 0xFF));
            }
        }

        // Serialize leaf_corrections size (4 bytes)
        uint32_t corrections_size = static_cast<uint32_t>(key.leaf_corrections.size());
        for (size_t i = 0; i < 4; ++i) {
            payload.push_back(static_cast<uint8_t>((corrections_size >> (i * 8)) & 0xFF));
        }

        // Serialize leaf_corrections
        for (const auto& [idx, correction_bytes] : key.leaf_corrections) {
            // Index (8 bytes)
            for (size_t i = 0; i < 8; ++i) {
                payload.push_back(static_cast<uint8_t>((idx >> (i * 8)) & 0xFF));
            }

            // Correction size (4 bytes)
            uint32_t corr_size = static_cast<uint32_t>(correction_bytes.size());
            for (size_t i = 0; i < 4; ++i) {
                payload.push_back(static_cast<uint8_t>((corr_size >> (i * 8)) & 0xFF));
            }

            // Correction bytes
            payload.insert(payload.end(), correction_bytes.begin(), correction_bytes.end());
        }

        return Message(MessageType::DPF_KeyCompressed, std::move(payload));
    }

    static DPFKeyMessage from_message(const Message& msg) {
        if (msg.type != MessageType::DPF_KeyCompressed) {
            throw NetworkError("Invalid message type for DPFKeyMessage");
        }

        DPFKeyMessage result;
        const auto& data = msg.payload;
        size_t offset = 0;

        // Deserialize domain_size
        result.key.domain_size = 0;
        for (size_t i = 0; i < 8; ++i) {
            result.key.domain_size |= static_cast<size_t>(data[offset++]) << (i * 8);
        }

        // Deserialize num_points
        result.key.num_points = 0;
        for (size_t i = 0; i < 8; ++i) {
            result.key.num_points |= static_cast<size_t>(data[offset++]) << (i * 8);
        }

        // Deserialize shared_nodes
        uint32_t num_nodes = 0;
        for (size_t i = 0; i < 4; ++i) {
            num_nodes |= static_cast<uint32_t>(data[offset++]) << (i * 8);
        }

        result.key.shared_nodes.resize(num_nodes);
        for (auto& node : result.key.shared_nodes) {
            // Deserialize seed
            std::copy_n(data.begin() + offset, 16, node.seed.begin());
            offset += 16;

            // Deserialize control_bit
            node.control_bit = (data[offset++] != 0);

            // Deserialize depth
            node.depth = 0;
            for (size_t i = 0; i < 8; ++i) {
                node.depth |= static_cast<size_t>(data[offset++]) << (i * 8);
            }

            // Deserialize path
            node.path = 0;
            for (size_t i = 0; i < 8; ++i) {
                node.path |= static_cast<size_t>(data[offset++]) << (i * 8);
            }

            // Deserialize is_leaf
            node.is_leaf = (data[offset++] != 0);

            // Deserialize target_index
            node.target_index = 0;
            for (size_t i = 0; i < 8; ++i) {
                node.target_index |= static_cast<size_t>(data[offset++]) << (i * 8);
            }
        }

        // Deserialize branch_map
        uint32_t branch_map_size = 0;
        for (size_t i = 0; i < 4; ++i) {
            branch_map_size |= static_cast<uint32_t>(data[offset++]) << (i * 8);
        }

        for (uint32_t i = 0; i < branch_map_size; ++i) {
            // Key
            size_t key_depth = 0, key_path = 0;
            for (size_t j = 0; j < 8; ++j) {
                key_depth |= static_cast<size_t>(data[offset++]) << (j * 8);
            }
            for (size_t j = 0; j < 8; ++j) {
                key_path |= static_cast<size_t>(data[offset++]) << (j * 8);
            }

            // Value
            size_t left_idx = 0, right_idx = 0;
            for (size_t j = 0; j < 8; ++j) {
                left_idx |= static_cast<size_t>(data[offset++]) << (j * 8);
            }
            for (size_t j = 0; j < 8; ++j) {
                right_idx |= static_cast<size_t>(data[offset++]) << (j * 8);
            }

            result.key.branch_map[{key_depth, key_path}] = {left_idx, right_idx};
        }

        // Deserialize leaf_corrections
        uint32_t corrections_size = 0;
        for (size_t i = 0; i < 4; ++i) {
            corrections_size |= static_cast<uint32_t>(data[offset++]) << (i * 8);
        }

        for (uint32_t i = 0; i < corrections_size; ++i) {
            // Index
            size_t idx = 0;
            for (size_t j = 0; j < 8; ++j) {
                idx |= static_cast<size_t>(data[offset++]) << (j * 8);
            }

            // Correction size
            uint32_t corr_size = 0;
            for (size_t j = 0; j < 4; ++j) {
                corr_size |= static_cast<uint32_t>(data[offset++]) << (j * 8);
            }

            // Correction bytes
            std::vector<uint8_t> correction_bytes(corr_size);
            std::copy_n(data.begin() + offset, corr_size, correction_bytes.begin());
            offset += corr_size;

            result.key.leaf_corrections[idx] = std::move(correction_bytes);
        }

        return result;
    }
};

// ============================================================================
// Handshake Message
// ============================================================================

struct HandshakeMessage {
    uint32_t protocol_version;  ///< Protocol version
    PartyId party_id;           ///< Party identifier (0 or 1)
    SessionId session_id;       ///< Session ID

    Message to_message() const;
    static HandshakeMessage from_message(const Message& msg);
};

} // namespace silentmpc::network
