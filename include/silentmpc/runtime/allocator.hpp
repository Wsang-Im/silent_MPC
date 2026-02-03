#pragma once

#include "silentmpc/core/types.hpp"
#include "silentmpc/runtime/epoch_manager.hpp"
#include <atomic>
#include <mutex>
#include <filesystem>
#include <vector>
#include <optional>

namespace silentmpc::runtime {

// ============================================================================
// Waste-Not-Reuse Allocator (Contract 2, Section 2.3)
// ============================================================================

/// Block-based index allocator with crash-safe persistence
/// Enforces: no index range overlap within same (sid, ε)
/// Wastes in-flight blocks on crash, but guarantees zero reuse
class WasteNotReuseAllocator {
public:
    /// Allocated index block [start, end)
    struct IndexBlock {
        CorrelationIndex start;
        CorrelationIndex end;

        size_t size() const { return static_cast<size_t>(end - start); }
        bool contains(CorrelationIndex idx) const {
            return idx >= start && idx < end;
        }
    };

private:
    std::atomic<CorrelationIndex> high_watermark_{0};
    size_t block_size_;
    SessionId sid_;
    EpochId epoch_;

    std::filesystem::path persist_path_;
    mutable std::mutex persist_mutex_;

    // Statistics
    std::atomic<size_t> total_allocated_{0};
    std::atomic<size_t> total_wasted_{0};

public:
    /// Initialize allocator for (sid, epoch)
    /// block_size: default 10^4 (Section 7.4)
    WasteNotReuseAllocator(
        const SessionId& sid,
        EpochId epoch,
        size_t block_size = 10'000,
        const std::filesystem::path& persist_dir = ""
    );

    /// Allocate a block of indices [ℓ_start, ℓ_end)
    /// 1. Increment high-watermark atomically
    /// 2. Persist new high-watermark (fsync)
    /// 3. Return block to caller
    /// Contract 2: high-watermark is persisted BEFORE issuing block
    IndexBlock allocate_block();

    /// Allocate single index (convenience wrapper)
    CorrelationIndex allocate_single() {
        auto block = allocate_block();
        // Waste rest of block (acceptable for single queries)
        return block.start;
    }

    /// Fast-forward after crash recovery
    /// Resumes strictly after persisted high-watermark
    /// Any in-flight indices from crashed block are wasted
    void fast_forward_after_crash();

    /// Current high-watermark (read-only)
    CorrelationIndex current_high_watermark() const {
        return high_watermark_.load(std::memory_order_acquire);
    }

    /// Wasted index ratio (for monitoring/debugging)
    /// Should be negligible under normal operation (< 0.01%, Section 7.4)
    double wasted_ratio() const {
        size_t allocated = total_allocated_.load(std::memory_order_relaxed);
        size_t wasted = total_wasted_.load(std::memory_order_relaxed);
        return (allocated > 0) ?
               static_cast<double>(wasted) / allocated : 0.0;
    }

    // Accessors
    size_t block_size() const { return block_size_; }
    const SessionId& session_id() const { return sid_; }
    EpochId epoch() const { return epoch_; }

private:
    /// Persist high-watermark: atomic write + fsync
    void persist_watermark(CorrelationIndex watermark) const;

    /// Load watermark from disk
    static std::optional<CorrelationIndex> load_watermark(
        const std::filesystem::path& path
    );

    /// Update wasted count (called on fast-forward)
    void record_wasted(size_t count) {
        total_wasted_.fetch_add(count, std::memory_order_relaxed);
    }
};

// ============================================================================
// Session Rotation Manager (Contract 3, Section 2.3)
// ============================================================================

/// Enforces query-bounded security by rotating sessions
/// Before ℓ exceeds Q_max, triggers rotation with fresh sid
class SessionRotationManager {
private:
    CorrelationIndex query_limit_;  ///< Q_max (default: 2^30)
    CorrelationIndex query_count_{0};

    SessionId current_sid_;
    std::atomic<bool> rotation_needed_{false};

public:
    /// Initialize with query limit (Section 5, query-bounded model)
    explicit SessionRotationManager(
        CorrelationIndex query_limit = 1ULL << 30,
        const SessionId& initial_sid = {}
    );

    /// Check if index allocation would exceed limit
    bool should_rotate(CorrelationIndex requested_count) const {
        return (query_count_ + requested_count) > query_limit_;
    }

    /// Record allocated indices
    /// Returns true if rotation is now required
    bool record_allocation(CorrelationIndex count) {
        query_count_ += count;
        if (query_count_ >= query_limit_) {
            rotation_needed_.store(true, std::memory_order_release);
            return true;
        }
        return false;
    }

    /// Generate fresh session ID for rotation
    /// Uses OS CSPRNG for 128-bit entropy
    static SessionId generate_fresh_sid();

    /// Rotate session: returns new sid
    /// Caller must re-run Setup with new sid
    SessionId rotate();

    // Accessors
    CorrelationIndex current_count() const { return query_count_; }
    CorrelationIndex limit() const { return query_limit_; }
    const SessionId& current_sid() const { return current_sid_; }
    bool needs_rotation() const {
        return rotation_needed_.load(std::memory_order_acquire);
    }

    /// Remaining budget before rotation
    CorrelationIndex remaining() const {
        return (query_count_ < query_limit_) ?
               (query_limit_ - query_count_) : 0;
    }
};

// ============================================================================
// Integrated Runtime State (combines Epoch + Allocator + Rotation)
// ============================================================================

/// Complete runtime state manager (Section 2.1, 2.3)
class RuntimeStateManager {
private:
    UnifiedEpochManager epoch_mgr_;
    WasteNotReuseAllocator allocator_;
    SessionRotationManager rotation_mgr_;

    mutable std::mutex state_mutex_;

public:
    RuntimeStateManager(
        const EpochPolicy& epoch_policy,
        const SessionId& sid,
        CorrelationIndex query_limit,
        size_t block_size = 10'000
    );

    /// Allocate index block with all safety checks
    /// Enforces: freshness invariant (no reuse), query limit (Contract 3)
    WasteNotReuseAllocator::IndexBlock allocate_safe();

    /// Handle crash recovery
    /// 1. Load persisted epoch (may increment if rollback detected)
    /// 2. Fast-forward allocator to persisted high-watermark
    void recover_from_crash();

    /// Rotate session when query limit reached
    /// Returns new SessionId; caller must re-run Setup
    SessionId rotate_session();

    /// Get current runtime context for Eval
    RuntimeContext current_context() const {
        RuntimeContext ctx;
        ctx.epoch = epoch_mgr_.current();
        ctx.high_watermark = allocator_.current_high_watermark();
        return ctx;
    }

    // Component access
    UnifiedEpochManager& epoch_manager() { return epoch_mgr_; }
    WasteNotReuseAllocator& allocator() { return allocator_; }
    SessionRotationManager& rotation_manager() { return rotation_mgr_; }

    const UnifiedEpochManager& epoch_manager() const { return epoch_mgr_; }
    const WasteNotReuseAllocator& allocator() const { return allocator_; }
    const SessionRotationManager& rotation_manager() const { return rotation_mgr_; }
};

} // namespace silentmpc::runtime
