#include "silentmpc/runtime/allocator.hpp"
#include <fstream>
#include <random>

namespace silentmpc::runtime {

// ============================================================================
// WasteNotReuseAllocator Implementation
// ============================================================================

WasteNotReuseAllocator::WasteNotReuseAllocator(
    const SessionId& sid,
    EpochId epoch,
    size_t block_size,
    const std::filesystem::path& persist_dir
) : block_size_(block_size), sid_(sid), epoch_(epoch) {

    if (!persist_dir.empty()) {
        persist_path_ = persist_dir / "allocator.dat";

        // Create directory
        if (!std::filesystem::exists(persist_dir)) {
            std::filesystem::create_directories(persist_dir);
        }

        // Try to load existing watermark
        auto loaded = load_watermark(persist_path_);
        if (loaded) {
            high_watermark_.store(*loaded, std::memory_order_release);
        } else {
            persist_watermark(0);
        }
    }
}

WasteNotReuseAllocator::IndexBlock WasteNotReuseAllocator::allocate_block() {
    // Atomically increment and get new range
    CorrelationIndex start = high_watermark_.fetch_add(
        block_size_, std::memory_order_acq_rel
    );
    CorrelationIndex end = start + block_size_;

    // Persist high-watermark BEFORE returning block (Contract 2)
    if (!persist_path_.empty()) {
        persist_watermark(end);
    }

    // Update statistics
    total_allocated_.fetch_add(block_size_, std::memory_order_relaxed);

    return IndexBlock{start, end};
}

void WasteNotReuseAllocator::fast_forward_after_crash() {
    if (persist_path_.empty()) {
        return;  // No persistence configured
    }

    // Load persisted high-watermark
    auto loaded = load_watermark(persist_path_);
    if (!loaded) {
        return;  // No file to recover from
    }

    CorrelationIndex persisted = *loaded;
    CorrelationIndex current = high_watermark_.load(std::memory_order_acquire);

    if (persisted > current) {
        // We rolled back - fast forward to persisted value
        size_t wasted = static_cast<size_t>(persisted - current);
        high_watermark_.store(persisted, std::memory_order_release);
        record_wasted(wasted);
    }
}

void WasteNotReuseAllocator::persist_watermark(CorrelationIndex watermark) const {
    std::lock_guard<std::mutex> lock(persist_mutex_);

    // Atomic write: temp file + rename
    auto temp_path = persist_path_;
    temp_path += ".tmp";

    {
        std::ofstream file(temp_path, std::ios::binary | std::ios::trunc);
        if (!file) {
            throw std::runtime_error("Failed to persist allocator watermark");
        }

        // Write: [sid(16) || epoch(8) || watermark(8)]
        file.write(reinterpret_cast<const char*>(sid_.data()), sid_.size());

        uint8_t epoch_bytes[8];
        for (size_t i = 0; i < 8; ++i) {
            epoch_bytes[i] = static_cast<uint8_t>((epoch_ >> (i * 8)) & 0xFF);
        }
        file.write(reinterpret_cast<const char*>(epoch_bytes), 8);

        uint8_t watermark_bytes[8];
        for (size_t i = 0; i < 8; ++i) {
            watermark_bytes[i] = static_cast<uint8_t>((watermark >> (i * 8)) & 0xFF);
        }
        file.write(reinterpret_cast<const char*>(watermark_bytes), 8);

        file.flush();
    }

    // Atomic rename
    std::filesystem::rename(temp_path, persist_path_);
}

std::optional<CorrelationIndex> WasteNotReuseAllocator::load_watermark(
    const std::filesystem::path& path
) {
    if (!std::filesystem::exists(path)) {
        return std::nullopt;
    }

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return std::nullopt;
    }

    // Read: [sid(16) || epoch(8) || watermark(8)]
    std::array<uint8_t, 16> sid;
    uint8_t epoch_bytes[8];
    uint8_t watermark_bytes[8];

    file.read(reinterpret_cast<char*>(sid.data()), 16);
    file.read(reinterpret_cast<char*>(epoch_bytes), 8);
    file.read(reinterpret_cast<char*>(watermark_bytes), 8);

    if (file.gcount() != 8) {
        return std::nullopt;
    }

    // Reconstruct watermark
    CorrelationIndex watermark = 0;
    for (size_t i = 0; i < 8; ++i) {
        watermark |= static_cast<CorrelationIndex>(watermark_bytes[i]) << (i * 8);
    }

    return watermark;
}

// ============================================================================
// SessionRotationManager Implementation
// ============================================================================

SessionRotationManager::SessionRotationManager(
    CorrelationIndex query_limit,
    const SessionId& initial_sid
) : query_limit_(query_limit), current_sid_(initial_sid) {

    if (current_sid_ == SessionId{}) {
        current_sid_ = generate_fresh_sid();
    }
}

SessionId SessionRotationManager::generate_fresh_sid() {
    SessionId sid;

#ifdef __linux__
    // Use /dev/urandom for cryptographic randomness
    std::ifstream urandom("/dev/urandom", std::ios::binary);
    if (urandom) {
        urandom.read(reinterpret_cast<char*>(sid.data()), sid.size());
        return sid;
    }
#endif

    // Fallback: std::random_device
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint8_t> dist(0, 255);

    for (auto& byte : sid) {
        byte = dist(gen);
    }

    return sid;
}

SessionId SessionRotationManager::rotate() {
    // Generate fresh session ID
    current_sid_ = generate_fresh_sid();

    // Reset counters
    query_count_ = 0;
    rotation_needed_.store(false, std::memory_order_release);

    return current_sid_;
}

// ============================================================================
// RuntimeStateManager Implementation
// ============================================================================

RuntimeStateManager::RuntimeStateManager(
    const EpochPolicy& epoch_policy,
    const SessionId& sid,
    CorrelationIndex query_limit,
    size_t block_size
) : epoch_mgr_(epoch_policy),
    allocator_(sid, epoch_mgr_.current(), block_size, epoch_policy.persist_dir),
    rotation_mgr_(query_limit, sid)
{
}

WasteNotReuseAllocator::IndexBlock RuntimeStateManager::allocate_safe() {
    std::lock_guard<std::mutex> lock(state_mutex_);

    // Check if rotation is needed (Contract 3)
    if (rotation_mgr_.needs_rotation()) {
        throw std::runtime_error(
            "Session rotation required - call rotate_session() first"
        );
    }

    // Allocate block
    auto block = allocator_.allocate_block();

    // Record allocation for rotation tracking
    rotation_mgr_.record_allocation(block.size());

    return block;
}

void RuntimeStateManager::recover_from_crash() {
    std::lock_guard<std::mutex> lock(state_mutex_);

    // 1. Check if epoch rolled back
    // (In production: compare with external rollback-resistant store)

    // 2. Fast-forward allocator
    allocator_.fast_forward_after_crash();
}

SessionId RuntimeStateManager::rotate_session() {
    std::lock_guard<std::mutex> lock(state_mutex_);

    // Increment epoch
    epoch_mgr_.rotate();

    // Rotate session
    SessionId new_sid = rotation_mgr_.rotate();

    // Note: Caller must re-run Setup with new sid and create new allocator

    return new_sid;
}

} // namespace silentmpc::runtime
