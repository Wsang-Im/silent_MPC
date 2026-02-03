#include "silentmpc/network/message.hpp"
#include <cstring>

namespace silentmpc::network {

// ============================================================================
// Base Message Implementation
// ============================================================================

std::vector<uint8_t> Message::serialize() const {
    std::vector<uint8_t> result;
    result.reserve(1 + payload.size());

    // Type (1 byte)
    result.push_back(static_cast<uint8_t>(type));

    // Payload
    result.insert(result.end(), payload.begin(), payload.end());

    return result;
}

Message Message::deserialize(std::span<const uint8_t> data) {
    if (data.empty()) {
        throw NetworkError("Cannot deserialize empty message");
    }

    Message msg;
    msg.type = static_cast<MessageType>(data[0]);

    if (data.size() > 1) {
        msg.payload.assign(data.begin() + 1, data.end());
    }

    return msg;
}

void Message::send(TCPConnection& conn) const {
    auto serialized = serialize();
    conn.send_with_length(serialized);
}

Message Message::receive(TCPConnection& conn) {
    auto data = conn.receive_with_length();
    return deserialize(data);
}

// ============================================================================
// OT Messages Implementation
// ============================================================================

Message OTBaseSetupMessage::to_message() const {
    return Message(MessageType::OT_BaseSetup, A);
}

OTBaseSetupMessage OTBaseSetupMessage::from_message(const Message& msg) {
    if (msg.type != MessageType::OT_BaseSetup) {
        throw NetworkError("Invalid message type for OTBaseSetupMessage");
    }

    OTBaseSetupMessage result;
    result.A = msg.payload;
    return result;
}

Message OTReceiverChoiceMessage::to_message() const {
    std::vector<uint8_t> payload;
    payload.reserve(B.size() + 1);

    // Choice bit (1 byte)
    payload.push_back(choice ? 1 : 0);

    // B value
    payload.insert(payload.end(), B.begin(), B.end());

    return Message(MessageType::OT_ReceiverChoice, std::move(payload));
}

OTReceiverChoiceMessage OTReceiverChoiceMessage::from_message(const Message& msg) {
    if (msg.type != MessageType::OT_ReceiverChoice) {
        throw NetworkError("Invalid message type for OTReceiverChoiceMessage");
    }

    if (msg.payload.empty()) {
        throw NetworkError("OTReceiverChoiceMessage payload is empty");
    }

    OTReceiverChoiceMessage result;
    result.choice = (msg.payload[0] != 0);
    result.B.assign(msg.payload.begin() + 1, msg.payload.end());
    return result;
}

Message OTSenderKeysMessage::to_message() const {
    std::vector<uint8_t> payload;

    // Key0 size (4 bytes)
    uint32_t key0_size = static_cast<uint32_t>(key0.size());
    for (size_t i = 0; i < 4; ++i) {
        payload.push_back(static_cast<uint8_t>((key0_size >> (i * 8)) & 0xFF));
    }

    // Key0 data
    payload.insert(payload.end(), key0.begin(), key0.end());

    // Key1 size (4 bytes)
    uint32_t key1_size = static_cast<uint32_t>(key1.size());
    for (size_t i = 0; i < 4; ++i) {
        payload.push_back(static_cast<uint8_t>((key1_size >> (i * 8)) & 0xFF));
    }

    // Key1 data
    payload.insert(payload.end(), key1.begin(), key1.end());

    return Message(MessageType::OT_SenderKeys, std::move(payload));
}

OTSenderKeysMessage OTSenderKeysMessage::from_message(const Message& msg) {
    if (msg.type != MessageType::OT_SenderKeys) {
        throw NetworkError("Invalid message type for OTSenderKeysMessage");
    }

    const auto& data = msg.payload;
    size_t offset = 0;

    if (data.size() < 8) {
        throw NetworkError("OTSenderKeysMessage payload too small");
    }

    // Key0 size
    uint32_t key0_size = 0;
    for (size_t i = 0; i < 4; ++i) {
        key0_size |= static_cast<uint32_t>(data[offset++]) << (i * 8);
    }

    // Key0 data
    if (offset + key0_size > data.size()) {
        throw NetworkError("OTSenderKeysMessage key0 size invalid");
    }

    OTSenderKeysMessage result;
    result.key0.assign(data.begin() + offset, data.begin() + offset + key0_size);
    offset += key0_size;

    // Key1 size
    uint32_t key1_size = 0;
    for (size_t i = 0; i < 4; ++i) {
        key1_size |= static_cast<uint32_t>(data[offset++]) << (i * 8);
    }

    // Key1 data
    if (offset + key1_size > data.size()) {
        throw NetworkError("OTSenderKeysMessage key1 size invalid");
    }

    result.key1.assign(data.begin() + offset, data.begin() + offset + key1_size);

    return result;
}

// ============================================================================
// Handshake Message Implementation
// ============================================================================

Message HandshakeMessage::to_message() const {
    std::vector<uint8_t> payload;

    // Protocol version (4 bytes)
    for (size_t i = 0; i < 4; ++i) {
        payload.push_back(static_cast<uint8_t>((protocol_version >> (i * 8)) & 0xFF));
    }

    // Party ID (1 byte)
    payload.push_back(party_id);

    // Session ID (16 bytes)
    payload.insert(payload.end(), session_id.begin(), session_id.end());

    return Message(MessageType::Handshake, std::move(payload));
}

HandshakeMessage HandshakeMessage::from_message(const Message& msg) {
    if (msg.type != MessageType::Handshake) {
        throw NetworkError("Invalid message type for HandshakeMessage");
    }

    if (msg.payload.size() != 21) {
        throw NetworkError("HandshakeMessage invalid size");
    }

    HandshakeMessage result;
    const auto& data = msg.payload;
    size_t offset = 0;

    // Protocol version
    result.protocol_version = 0;
    for (size_t i = 0; i < 4; ++i) {
        result.protocol_version |= static_cast<uint32_t>(data[offset++]) << (i * 8);
    }

    // Party ID
    result.party_id = data[offset++];

    // Session ID
    std::copy_n(data.begin() + offset, 16, result.session_id.begin());

    return result;
}

} // namespace silentmpc::network
