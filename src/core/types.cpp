#include "silentmpc/core/types.hpp"
#include <cstring>

namespace silentmpc {

// ============================================================================
// DerivationContext Implementation
// ============================================================================

std::vector<uint8_t> DerivationContext::serialize() const {
    std::vector<uint8_t> result;
    result.reserve(16 + 8 + 8 + 1 + 1 + 8);  // Total size

    // Session ID (16 bytes)
    result.insert(result.end(), sid.begin(), sid.end());

    // Epoch (8 bytes, little-endian)
    for (size_t i = 0; i < 8; ++i) {
        result.push_back(static_cast<uint8_t>((epoch >> (i * 8)) & 0xFF));
    }

    // Index (8 bytes, little-endian)
    for (size_t i = 0; i < 8; ++i) {
        result.push_back(static_cast<uint8_t>((index >> (i * 8)) & 0xFF));
    }

    // Party ID (1 byte)
    result.push_back(party);

    // Term ID (1 byte)
    result.push_back(static_cast<uint8_t>(term_id));

    // Tile ID (8 bytes, little-endian)
    for (size_t i = 0; i < 8; ++i) {
        result.push_back(static_cast<uint8_t>((tile_id >> (i * 8)) & 0xFF));
    }

    return result;
}

} // namespace silentmpc
