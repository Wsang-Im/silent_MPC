#include "silentmpc/runtime/epoch_manager.hpp"
#include <openssl/hmac.h>
#include <openssl/evp.h>
#include <fstream>
#include <random>
#include <fcntl.h>
#include <unistd.h>

namespace silentmpc::runtime {

// ============================================================================
// EpochManager Implementation
// ============================================================================

EpochManager::EpochManager(const std::filesystem::path& persist_dir)
    : persist_path_(persist_dir / "epoch.dat") {

    // Create directory if needed
    if (!persist_dir.empty() && !std::filesystem::exists(persist_dir)) {
        std::filesystem::create_directories(persist_dir);
    }

    // Load existing epoch or start from 0
    auto loaded = load_from_disk(persist_path_);
    if (loaded) {
        current_epoch_.store(*loaded, std::memory_order_release);
    } else {
        current_epoch_.store(0, std::memory_order_release);
        persist();
    }
}

EpochId EpochManager::increment() {
    EpochId new_epoch = current_epoch_.fetch_add(1, std::memory_order_acq_rel) + 1;

    // Persist immediately (Contract 1)
    atomic_write_epoch(new_epoch);

    return new_epoch;
}

void EpochManager::force_set(EpochId new_epoch) {
    current_epoch_.store(new_epoch, std::memory_order_release);
    atomic_write_epoch(new_epoch);
}

void EpochManager::persist() const {
    atomic_write_epoch(current());
}

std::optional<EpochId> EpochManager::load_from_disk(
    const std::filesystem::path& persist_path
) {
    if (!std::filesystem::exists(persist_path)) {
        return std::nullopt;
    }

    std::ifstream file(persist_path, std::ios::binary);
    if (!file) {
        return std::nullopt;
    }

    EpochId epoch;
    file.read(reinterpret_cast<char*>(&epoch), sizeof(epoch));

    if (file.gcount() == sizeof(epoch)) {
        return epoch;
    }

    return std::nullopt;
}

void EpochManager::atomic_write_epoch(EpochId epoch) const {
    std::lock_guard<std::mutex> lock(persist_mutex_);

    // Write to temporary file
    auto temp_path = persist_path_;
    temp_path += ".tmp";

    {
        std::ofstream file(temp_path, std::ios::binary | std::ios::trunc);
        if (!file) {
            throw std::runtime_error("Failed to open epoch file for writing");
        }

        file.write(reinterpret_cast<const char*>(&epoch), sizeof(epoch));
        file.flush();

        // fsync to ensure data reaches disk
        file.close();
#ifndef _WIN32
        int fd = open(temp_path.c_str(), O_RDONLY);
        if (fd >= 0) {
            fsync(fd);
            close(fd);
        }
#endif
    }

    // Atomic rename
    std::filesystem::rename(temp_path, persist_path_);
}

// ============================================================================
// RandomEpochManager Implementation
// ============================================================================

RandomEpochManager::RandomEpochManager(const std::filesystem::path& persist_dir)
    : persist_path_(persist_dir / "epoch_random.dat") {

    if (!persist_dir.empty() && !std::filesystem::exists(persist_dir)) {
        std::filesystem::create_directories(persist_dir);
    }

    // Try to load existing epoch
    std::ifstream file(persist_path_, std::ios::binary);
    if (file && file.read(reinterpret_cast<char*>(current_epoch_.data()),
                          current_epoch_.size()).gcount() == current_epoch_.size()) {
        // Loaded successfully
    } else {
        // Generate new random epoch
        sample_from_csprng();
        persist();
    }
}

void RandomEpochManager::rotate() {
    sample_from_csprng();
    persist();
}

void RandomEpochManager::persist() const {
    std::lock_guard<std::mutex> lock(persist_mutex_);

    std::ofstream file(persist_path_, std::ios::binary | std::ios::trunc);
    if (!file) {
        throw std::runtime_error("Failed to persist random epoch");
    }

    file.write(reinterpret_cast<const char*>(current_epoch_.data()),
               current_epoch_.size());
    file.flush();
}

void RandomEpochManager::sample_from_csprng() {
#ifdef __linux__
    // Use /dev/urandom on Linux
    std::ifstream urandom("/dev/urandom", std::ios::binary);
    if (urandom) {
        urandom.read(reinterpret_cast<char*>(current_epoch_.data()),
                    current_epoch_.size());
        return;
    }
#endif

    // Fallback: std::random_device (not always cryptographically secure)
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint8_t> dist(0, 255);

    for (auto& byte : current_epoch_) {
        byte = dist(gen);
    }
}

// ============================================================================
// UnifiedEpochManager Implementation
// ============================================================================

UnifiedEpochManager::UnifiedEpochManager(const EpochPolicy& policy)
    : policy_(policy),
      impl_(policy.mode == EpochPolicy::Mode::Monotonic
            ? std::variant<EpochManager, RandomEpochManager>(std::in_place_type<EpochManager>, policy.persist_dir)
            : std::variant<EpochManager, RandomEpochManager>(std::in_place_type<RandomEpochManager>, policy.persist_dir))
{
    // Variant initialized in-place in initializer list
}

EpochId UnifiedEpochManager::current() const {
    return std::visit([](const auto& mgr) -> EpochId {
        if constexpr (std::is_same_v<std::decay_t<decltype(mgr)>, EpochManager>) {
            return mgr.current();
        } else {
            return mgr.as_uint64();
        }
    }, impl_);
}

void UnifiedEpochManager::rotate() {
    std::visit([](auto& mgr) {
        if constexpr (std::is_same_v<std::decay_t<decltype(mgr)>, EpochManager>) {
            mgr.increment();
        } else {
            mgr.rotate();
        }
    }, impl_);
}

bool UnifiedEpochManager::validate(EpochId epoch) const {
    return std::visit([epoch](const auto& mgr) -> bool {
        if constexpr (std::is_same_v<std::decay_t<decltype(mgr)>, EpochManager>) {
            return mgr.validate(epoch);
        } else {
            return mgr.as_uint64() == epoch;
        }
    }, impl_);
}

void UnifiedEpochManager::persist() const {
    std::visit([](const auto& mgr) {
        mgr.persist();
    }, impl_);
}

// ============================================================================
// EpochConsistencyChecker Implementation
// ============================================================================

bool EpochConsistencyChecker::verify_control_message(
    std::span<const uint8_t> message,
    const SessionId& expected_sid,
    EpochId expected_epoch,
    std::span<const uint8_t> hmac_key
) {
    // Message format: [sid(16) || epoch(8) || payload || hmac(32)]
    if (message.size() < 16 + 8 + 32) {
        return false;
    }

    // Verify session ID
    if (!std::equal(message.begin(), message.begin() + 16,
                    expected_sid.begin(), expected_sid.end())) {
        return false;
    }

    // Verify epoch
    EpochId received_epoch = 0;
    for (size_t i = 0; i < 8; ++i) {
        received_epoch |= static_cast<EpochId>(message[16 + i]) << (i * 8);
    }

    if (received_epoch != expected_epoch) {
        return false;
    }

    // Verify HMAC
    if (message.size() < 32) {
        return false;  // Need at least 32 bytes for HMAC
    }

    size_t payload_len = message.size() - 32;
    std::span<const uint8_t> payload = message.subspan(0, payload_len);
    std::span<const uint8_t> received_hmac = message.subspan(payload_len, 32);

    // Compute HMAC-SHA256
    unsigned char computed_hmac[32];
    unsigned int hmac_len;

    HMAC_CTX* hmac_ctx = HMAC_CTX_new();
    HMAC_Init_ex(hmac_ctx, hmac_key.data(), hmac_key.size(), EVP_sha256(), nullptr);
    HMAC_Update(hmac_ctx, payload.data(), payload.size());
    HMAC_Final(hmac_ctx, computed_hmac, &hmac_len);
    HMAC_CTX_free(hmac_ctx);

    // Constant-time comparison
    bool hmac_valid = true;
    for (size_t i = 0; i < 32; ++i) {
        hmac_valid &= (computed_hmac[i] == received_hmac[i]);
    }

    return hmac_valid;
}

} // namespace silentmpc::runtime
