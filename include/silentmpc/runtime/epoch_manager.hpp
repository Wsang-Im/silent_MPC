#pragma once

#include "silentmpc/core/types.hpp"
#include <atomic>
#include <mutex>
#include <filesystem>
#include <fstream>
#include <optional>
#include <variant>

namespace silentmpc::runtime {

// ============================================================================
// Epoch Manager (Contract 1, Section 2.3)
// ============================================================================

/// Manages epoch lifecycle with rollback-resistant persistence
/// Ensures monotonic epoch increments to prevent reuse under VM snapshot/restore
class EpochManager {
private:
    std::atomic<EpochId> current_epoch_{0};
    std::filesystem::path persist_path_;
    mutable std::mutex persist_mutex_;

public:
    /// Initialize with persistent storage path
    /// If path exists, loads last epoch; otherwise starts from 0
    explicit EpochManager(const std::filesystem::path& persist_dir);

    /// Get current epoch (read-only)
    EpochId current() const noexcept {
        return current_epoch_.load(std::memory_order_acquire);
    }

    /// Increment epoch (Contract 1: epoch rotation)
    /// Atomically increments and persists to rollback-resistant storage
    /// Called on: session rotation (Contract 3), recovery events
    EpochId increment();

    /// Force set epoch (for recovery/initialization only)
    /// DANGEROUS: Only use when epoch store is known to be consistent
    void force_set(EpochId new_epoch);

    /// Validate epoch consistency
    /// Returns true if provided epoch matches current
    bool validate(EpochId epoch) const {
        return epoch == current();
    }

    /// Persist current epoch to durable storage
    /// Uses atomic write protocol: write-new, fsync, rename
    void persist() const;

    /// Load epoch from persistent storage
    /// Returns nullopt if file doesn't exist
    static std::optional<EpochId> load_from_disk(
        const std::filesystem::path& persist_path
    );

private:
    /// Atomic file write with fsync
    void atomic_write_epoch(EpochId epoch) const;
};

// ============================================================================
// Random Epoch Mode (Contract 1 alternative)
// ============================================================================

/// Alternative: Random 128-bit epochs for VM-snapshot-unsafe environments
/// Collision probability: ≤ t(t-1)/2^129 for t epochs
class RandomEpochManager {
private:
    std::array<uint8_t, 16> current_epoch_{};
    std::filesystem::path persist_path_;
    mutable std::mutex persist_mutex_;

public:
    explicit RandomEpochManager(const std::filesystem::path& persist_dir);

    /// Sample fresh random epoch (128-bit entropy from CSPRNG)
    void rotate();

    /// Get current epoch (as bytes)
    std::span<const uint8_t> current() const {
        return current_epoch_;
    }

    /// Convert to uint64_t for compatibility (first 64 bits)
    EpochId as_uint64() const {
        EpochId result = 0;
        for (size_t i = 0; i < 8; ++i) {
            result |= static_cast<EpochId>(current_epoch_[i]) << (i * 8);
        }
        return result;
    }

    void persist() const;

private:
    /// Sample from OS CSPRNG
    void sample_from_csprng();
};

// ============================================================================
// Epoch ID Policy Configuration
// ============================================================================

struct EpochPolicy {
    enum class Mode {
        Monotonic,   ///< Default: 64-bit monotonic counter
        Random       ///< Fallback: 128-bit random epochs
    };

    Mode mode = Mode::Monotonic;
    std::filesystem::path persist_dir;

    /// Recommended: rollback-resistant storage
    /// Example: external KV store, persistent volume excluded from snapshots
    bool require_rollback_resistance = true;
};

// ============================================================================
// Unified Epoch Manager (supports both modes)
// ============================================================================

/// High-level epoch manager that switches between modes
class UnifiedEpochManager {
private:
    EpochPolicy policy_;
    std::variant<EpochManager, RandomEpochManager> impl_;

public:
    explicit UnifiedEpochManager(const EpochPolicy& policy);

    /// Get current epoch as EpochId (uint64_t)
    EpochId current() const;

    /// Increment/rotate epoch
    void rotate();

    /// Validate epoch
    bool validate(EpochId epoch) const;

    /// Persist to storage
    void persist() const;

    const EpochPolicy& policy() const { return policy_; }
};

// ============================================================================
// Fail-Closed Consistency Checker (Section 2.3)
// ============================================================================

/// Validates epoch consistency across parties
/// Fail-closed: aborts on mismatch rather than proceeding
class EpochConsistencyChecker {
public:
    /// Check if received epoch matches local epoch
    /// Throws EpochMismatchError if inconsistent
    static void check_or_fail(
        EpochId local_epoch,
        EpochId received_epoch
    ) {
        if (local_epoch != received_epoch) {
            throw EpochMismatchError();
        }
    }

    /// Verify epoch in control-plane message
    /// Message format: [sid || epoch || payload || hmac]
    static bool verify_control_message(
        std::span<const uint8_t> message,
        const SessionId& expected_sid,
        EpochId expected_epoch,
        std::span<const uint8_t> hmac_key
    );
};

} // namespace silentmpc::runtime
