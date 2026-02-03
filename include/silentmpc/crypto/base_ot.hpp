#pragma once

#include <array>
#include <vector>
#include <cstdint>
#include <openssl/ec.h>
#include <openssl/obj_mac.h>
#include <openssl/bn.h>
#include <openssl/sha.h>
#include <stdexcept>
#include <memory>

namespace silentmpc::crypto {

// ============================================================================
// Chou-Orlandi "Simplest OT" Protocol (2015)
// Most efficient base OT protocol using elliptic curves
// ============================================================================

/// Elliptic curve point wrapper
class ECPoint {
private:
    EC_POINT* point_;
    const EC_GROUP* group_;

public:
    ECPoint(const EC_GROUP* group) : group_(group) {
        point_ = EC_POINT_new(group);
        if (!point_) {
            throw std::runtime_error("Failed to create EC_POINT");
        }
    }

    ~ECPoint() {
        if (point_) {
            EC_POINT_free(point_);
        }
    }

    // Disable copy, enable move
    ECPoint(const ECPoint&) = delete;
    ECPoint& operator=(const ECPoint&) = delete;

    ECPoint(ECPoint&& other) noexcept : point_(other.point_), group_(other.group_) {
        other.point_ = nullptr;
    }

    ECPoint& operator=(ECPoint&& other) noexcept {
        if (this != &other) {
            if (point_) EC_POINT_free(point_);
            point_ = other.point_;
            group_ = other.group_;
            other.point_ = nullptr;
        }
        return *this;
    }

    EC_POINT* get() { return point_; }
    const EC_POINT* get() const { return point_; }

    /// Serialize point to bytes (compressed format)
    std::vector<uint8_t> serialize() const {
        size_t len = EC_POINT_point2oct(group_, point_,
                                       POINT_CONVERSION_COMPRESSED,
                                       nullptr, 0, nullptr);
        std::vector<uint8_t> bytes(len);
        EC_POINT_point2oct(group_, point_,
                          POINT_CONVERSION_COMPRESSED,
                          bytes.data(), len, nullptr);
        return bytes;
    }

    /// Deserialize point from bytes
    void deserialize(const std::vector<uint8_t>& bytes) {
        if (!EC_POINT_oct2point(group_, point_, bytes.data(), bytes.size(), nullptr)) {
            throw std::runtime_error("Failed to deserialize EC_POINT");
        }
    }
};

/// Chou-Orlandi Simplest OT implementation
class ChouOrlandiOT {
private:
    std::unique_ptr<EC_GROUP, decltype(&EC_GROUP_free)> group_;
    std::unique_ptr<BN_CTX, decltype(&BN_CTX_free)> bn_ctx_;

    static constexpr size_t HASH_OUTPUT_SIZE = 32;  // SHA-256

    /// Hash function H(index, point) for OT
    static std::array<uint8_t, 32> hash_point(size_t index, const EC_POINT* point, const EC_GROUP* group) {
        std::array<uint8_t, 32> output;

        // Serialize point
        size_t len = EC_POINT_point2oct(group, point,
                                       POINT_CONVERSION_COMPRESSED,
                                       nullptr, 0, nullptr);
        std::vector<uint8_t> point_bytes(len);
        EC_POINT_point2oct(group, point,
                          POINT_CONVERSION_COMPRESSED,
                          point_bytes.data(), len, nullptr);

        // Hash: SHA-256(index || point)
        SHA256_CTX sha_ctx;
        SHA256_Init(&sha_ctx);

        // Add index
        uint8_t index_bytes[8];
        for (size_t i = 0; i < 8; ++i) {
            index_bytes[i] = (index >> (i * 8)) & 0xFF;
        }
        SHA256_Update(&sha_ctx, index_bytes, 8);

        // Add point
        SHA256_Update(&sha_ctx, point_bytes.data(), point_bytes.size());

        SHA256_Final(output.data(), &sha_ctx);

        return output;
    }

public:
    ChouOrlandiOT()
        : group_(EC_GROUP_new_by_curve_name(NID_X9_62_prime256v1), EC_GROUP_free),
          bn_ctx_(BN_CTX_new(), BN_CTX_free) {
        if (!group_ || !bn_ctx_) {
            throw std::runtime_error("Failed to initialize elliptic curve");
        }
    }

    /// Sender's first message: Generate random elliptic curve point
    /// Returns: A = a·G where a is random scalar, G is generator
    struct SenderSetup {
        std::vector<uint8_t> public_point_A;  // Serialized A
        BIGNUM* private_scalar_a;              // Private scalar a
    };

    SenderSetup sender_setup() {
        SenderSetup result;

        // Generate random scalar a
        result.private_scalar_a = BN_new();
        const BIGNUM* order = EC_GROUP_get0_order(group_.get());
        BN_rand_range(result.private_scalar_a, order);

        // Compute A = a·G
        ECPoint A(group_.get());
        EC_POINT_mul(group_.get(), A.get(), result.private_scalar_a,
                    nullptr, nullptr, bn_ctx_.get());

        result.public_point_A = A.serialize();

        return result;
    }

    /// Receiver's message: Generate B based on choice bit
    /// If choice = 0: B = b·G
    /// If choice = 1: B = b·G + A
    struct ReceiverMessage {
        std::vector<uint8_t> public_point_B;  // Serialized B
        BIGNUM* private_scalar_b;              // Private scalar b
        bool choice;                            // Choice bit
    };

    ReceiverMessage receiver_generate(
        const std::vector<uint8_t>& sender_public_A,
        bool choice
    ) {
        ReceiverMessage result;
        result.choice = choice;

        // Deserialize A
        ECPoint A(group_.get());
        A.deserialize(sender_public_A);

        // Generate random scalar b
        result.private_scalar_b = BN_new();
        const BIGNUM* order = EC_GROUP_get0_order(group_.get());
        BN_rand_range(result.private_scalar_b, order);

        // Compute B = b·G
        ECPoint B(group_.get());
        EC_POINT_mul(group_.get(), B.get(), result.private_scalar_b,
                    nullptr, nullptr, bn_ctx_.get());

        // If choice = 1, add A: B = b·G + A
        if (choice) {
            EC_POINT_add(group_.get(), B.get(), B.get(), A.get(), bn_ctx_.get());
        }

        result.public_point_B = B.serialize();

        return result;
    }

    /// Sender derives both keys from receiver's B
    /// k0 = H(0, a·B)
    /// k1 = H(1, a·(B-A))
    struct SenderKeys {
        std::array<uint8_t, 32> key0;
        std::array<uint8_t, 32> key1;
    };

    SenderKeys sender_derive_keys(
        size_t index,
        const SenderSetup& setup,
        const std::vector<uint8_t>& receiver_public_B
    ) {
        SenderKeys result;

        // Deserialize B
        ECPoint B(group_.get());
        B.deserialize(receiver_public_B);

        // Compute k0 = H(index || 0, a·B)
        ECPoint aB(group_.get());
        EC_POINT_mul(group_.get(), aB.get(), nullptr,
                    B.get(), setup.private_scalar_a, bn_ctx_.get());

        auto hash0 = hash_point(index * 2 + 0, aB.get(), group_.get());
        result.key0 = hash0;

        // Compute k1 = H(index || 1, a·(B-A))
        // First compute A
        ECPoint A(group_.get());
        A.deserialize(setup.public_point_A);

        // B - A
        ECPoint B_minus_A(group_.get());
        EC_POINT_copy(B_minus_A.get(), B.get());
        EC_POINT_invert(group_.get(), A.get(), bn_ctx_.get());  // -A
        EC_POINT_add(group_.get(), B_minus_A.get(),
                    B_minus_A.get(), A.get(), bn_ctx_.get());

        // a·(B-A)
        ECPoint a_B_minus_A(group_.get());
        EC_POINT_mul(group_.get(), a_B_minus_A.get(), nullptr,
                    B_minus_A.get(), setup.private_scalar_a, bn_ctx_.get());

        auto hash1 = hash_point(index * 2 + 1, a_B_minus_A.get(), group_.get());
        result.key1 = hash1;

        return result;
    }

    /// Receiver derives chosen key
    /// k_choice = H(choice, b·A)
    std::array<uint8_t, 32> receiver_derive_key(
        size_t index,
        const ReceiverMessage& msg,
        const std::vector<uint8_t>& sender_public_A
    ) {
        // Deserialize A
        ECPoint A(group_.get());
        A.deserialize(sender_public_A);

        // Compute b·A
        ECPoint bA(group_.get());
        EC_POINT_mul(group_.get(), bA.get(), nullptr,
                    A.get(), msg.private_scalar_b, bn_ctx_.get());

        // H(index || choice, b·A)
        return hash_point(index * 2 + (msg.choice ? 1 : 0), bA.get(), group_.get());
    }

    /// Execute full OT protocol
    struct OTResult {
        std::array<uint8_t, 32> sender_key0;
        std::array<uint8_t, 32> sender_key1;
        std::array<uint8_t, 32> receiver_key;
        bool choice;
    };

    OTResult execute_ot(size_t index, bool choice) {
        OTResult result;
        result.choice = choice;

        // Round 1: Sender setup
        auto sender_setup_msg = sender_setup();

        // Round 2: Receiver response
        auto receiver_msg = receiver_generate(sender_setup_msg.public_point_A, choice);

        // Round 3: Both derive keys
        auto sender_keys = sender_derive_keys(index, sender_setup_msg, receiver_msg.public_point_B);
        result.sender_key0 = sender_keys.key0;
        result.sender_key1 = sender_keys.key1;

        result.receiver_key = receiver_derive_key(index, receiver_msg, sender_setup_msg.public_point_A);

        // Cleanup
        BN_free(sender_setup_msg.private_scalar_a);
        BN_free(receiver_msg.private_scalar_b);

        return result;
    }

    /// Batch OT execution (for base OTs in IKNP)
    struct BatchOTResult {
        std::vector<std::array<uint8_t, 32>> sender_keys0;
        std::vector<std::array<uint8_t, 32>> sender_keys1;
        std::vector<std::array<uint8_t, 32>> receiver_keys;
        std::vector<bool> choices;
    };

    BatchOTResult execute_batch_ot(const std::vector<bool>& choices) {
        BatchOTResult result;
        result.choices = choices;
        result.sender_keys0.reserve(choices.size());
        result.sender_keys1.reserve(choices.size());
        result.receiver_keys.reserve(choices.size());

        for (size_t i = 0; i < choices.size(); ++i) {
            auto ot_result = execute_ot(i, choices[i]);
            result.sender_keys0.push_back(ot_result.sender_key0);
            result.sender_keys1.push_back(ot_result.sender_key1);
            result.receiver_keys.push_back(ot_result.receiver_key);
        }

        return result;
    }
};

} // namespace silentmpc::crypto
