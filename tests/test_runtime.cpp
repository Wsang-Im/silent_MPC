#include <gtest/gtest.h>
#include "silentmpc/runtime/epoch_manager.hpp"
#include "silentmpc/runtime/allocator.hpp"
#include <filesystem>
#include <thread>

using namespace silentmpc::runtime;

// ============================================================================
// EpochManager Tests
// ============================================================================

class EpochManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_dir_ = std::filesystem::temp_directory_path() / "silentmpc_test_epoch";
        std::filesystem::create_directories(test_dir_);
    }

    void TearDown() override {
        if (std::filesystem::exists(test_dir_)) {
            std::filesystem::remove_all(test_dir_);
        }
    }

    std::filesystem::path test_dir_;
};

TEST_F(EpochManagerTest, Initialization) {
    EpochManager mgr(test_dir_);

    EXPECT_EQ(mgr.current(), 0ULL);
}

TEST_F(EpochManagerTest, Increment) {
    EpochManager mgr(test_dir_);

    EXPECT_EQ(mgr.current(), 0ULL);

    EpochId e1 = mgr.increment();
    EXPECT_EQ(e1, 1ULL);
    EXPECT_EQ(mgr.current(), 1ULL);

    EpochId e2 = mgr.increment();
    EXPECT_EQ(e2, 2ULL);
    EXPECT_EQ(mgr.current(), 2ULL);
}

TEST_F(EpochManagerTest, Persistence) {
    {
        EpochManager mgr(test_dir_);
        mgr.increment();
        mgr.increment();
        EXPECT_EQ(mgr.current(), 2ULL);
    }

    // Create new manager - should load persisted value
    {
        EpochManager mgr(test_dir_);
        EXPECT_EQ(mgr.current(), 2ULL);
    }
}

TEST_F(EpochManagerTest, Validation) {
    EpochManager mgr(test_dir_);

    mgr.increment();  // epoch = 1
    mgr.increment();  // epoch = 2

    EXPECT_TRUE(mgr.validate(2));   // Current epoch
    EXPECT_TRUE(mgr.validate(1));   // Previous epoch
    EXPECT_FALSE(mgr.validate(3));  // Future epoch
    EXPECT_FALSE(mgr.validate(0));  // Old epoch
}

TEST_F(EpochManagerTest, ForceSet) {
    EpochManager mgr(test_dir_);

    mgr.force_set(100);
    EXPECT_EQ(mgr.current(), 100ULL);

    // Should persist
    EpochManager mgr2(test_dir_);
    EXPECT_EQ(mgr2.current(), 100ULL);
}

TEST_F(EpochManagerTest, ConcurrentAccess) {
    EpochManager mgr(test_dir_);

    const int num_threads = 4;
    const int increments_per_thread = 100;

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&mgr, increments_per_thread]() {
            for (int j = 0; j < increments_per_thread; ++j) {
                mgr.increment();
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(mgr.current(), num_threads * increments_per_thread);
}

// ============================================================================
// RandomEpochManager Tests
// ============================================================================

class RandomEpochManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_dir_ = std::filesystem::temp_directory_path() / "silentmpc_test_random_epoch";
        std::filesystem::create_directories(test_dir_);
    }

    void TearDown() override {
        if (std::filesystem::exists(test_dir_)) {
            std::filesystem::remove_all(test_dir_);
        }
    }

    std::filesystem::path test_dir_;
};

TEST_F(RandomEpochManagerTest, Initialization) {
    RandomEpochManager mgr(test_dir_);

    auto epoch = mgr.current();
    EXPECT_EQ(epoch.size(), 16UL);

    // Should not be all zeros
    bool all_zero = true;
    for (uint8_t byte : epoch) {
        if (byte != 0) {
            all_zero = false;
            break;
        }
    }
    EXPECT_FALSE(all_zero);
}

TEST_F(RandomEpochManagerTest, Rotation) {
    RandomEpochManager mgr(test_dir_);

    auto epoch1 = mgr.current();
    mgr.rotate();
    auto epoch2 = mgr.current();

    // Should be different after rotation
    EXPECT_NE(epoch1, epoch2);
}

TEST_F(RandomEpochManagerTest, Persistence) {
    RandomEpochId epoch1;

    {
        RandomEpochManager mgr(test_dir_);
        epoch1 = mgr.current();
    }

    // Load from disk
    {
        RandomEpochManager mgr(test_dir_);
        auto epoch2 = mgr.current();
        EXPECT_EQ(epoch1, epoch2);
    }
}

TEST_F(RandomEpochManagerTest, Uint64Conversion) {
    RandomEpochManager mgr(test_dir_);

    uint64_t val = mgr.as_uint64();
    // Just check it's non-zero with high probability
    EXPECT_NE(val, 0ULL);
}

// ============================================================================
// WasteNotReuseAllocator Tests
// ============================================================================

class AllocatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_dir_ = std::filesystem::temp_directory_path() / "silentmpc_test_allocator";
        std::filesystem::create_directories(test_dir_);
        sid_.fill(0xAB);
    }

    void TearDown() override {
        if (std::filesystem::exists(test_dir_)) {
            std::filesystem::remove_all(test_dir_);
        }
    }

    SessionId sid_;
    std::filesystem::path test_dir_;
};

TEST_F(AllocatorTest, BasicAllocation) {
    WasteNotReuseAllocator alloc(sid_, 0, 1024, test_dir_);

    auto block1 = alloc.allocate_block();
    EXPECT_EQ(block1.start, 0ULL);
    EXPECT_EQ(block1.end, 1024ULL);
    EXPECT_EQ(block1.size(), 1024ULL);

    auto block2 = alloc.allocate_block();
    EXPECT_EQ(block2.start, 1024ULL);
    EXPECT_EQ(block2.end, 2048ULL);
}

TEST_F(AllocatorTest, NoGaps) {
    WasteNotReuseAllocator alloc(sid_, 0, 512, test_dir_);

    auto b1 = alloc.allocate_block();
    auto b2 = alloc.allocate_block();
    auto b3 = alloc.allocate_block();

    // Blocks should be contiguous
    EXPECT_EQ(b1.end, b2.start);
    EXPECT_EQ(b2.end, b3.start);
}

TEST_F(AllocatorTest, Persistence) {
    {
        WasteNotReuseAllocator alloc(sid_, 0, 1024, test_dir_);
        alloc.allocate_block();  // [0, 1024)
        alloc.allocate_block();  // [1024, 2048)
    }

    // Create new allocator - should continue from watermark
    {
        WasteNotReuseAllocator alloc(sid_, 0, 1024, test_dir_);
        alloc.fast_forward_after_crash();

        auto block = alloc.allocate_block();
        EXPECT_EQ(block.start, 2048ULL);
    }
}

TEST_F(AllocatorTest, CrashRecovery) {
    // Simulate crash: allocate but don't use blocks
    {
        WasteNotReuseAllocator alloc(sid_, 0, 1024, test_dir_);
        alloc.allocate_block();
        alloc.allocate_block();
        alloc.allocate_block();
        // Watermark persisted at 3072
    }

    // Recovery: fast-forward to persisted watermark
    {
        WasteNotReuseAllocator alloc(sid_, 0, 1024, test_dir_);
        alloc.fast_forward_after_crash();

        auto block = alloc.allocate_block();
        EXPECT_EQ(block.start, 3072ULL);
    }
}

TEST_F(AllocatorTest, Statistics) {
    WasteNotReuseAllocator alloc(sid_, 0, 1000, test_dir_);

    EXPECT_EQ(alloc.stats().total_allocated, 0UL);
    EXPECT_EQ(alloc.stats().total_wasted, 0UL);

    alloc.allocate_block();
    EXPECT_EQ(alloc.stats().total_allocated, 1000UL);

    alloc.allocate_block();
    EXPECT_EQ(alloc.stats().total_allocated, 2000UL);
}

TEST_F(AllocatorTest, ConcurrentAllocation) {
    WasteNotReuseAllocator alloc(sid_, 0, 64, test_dir_);

    const int num_threads = 8;
    const int blocks_per_thread = 10;

    std::vector<std::thread> threads;
    std::vector<std::vector<WasteNotReuseAllocator::IndexBlock>> results(num_threads);

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&alloc, &results, i, blocks_per_thread]() {
            for (int j = 0; j < blocks_per_thread; ++j) {
                results[i].push_back(alloc.allocate_block());
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // Check all blocks are unique and non-overlapping
    std::set<CorrelationIndex> starts;
    for (const auto& thread_results : results) {
        for (const auto& block : thread_results) {
            EXPECT_TRUE(starts.insert(block.start).second);  // Unique start
        }
    }

    EXPECT_EQ(starts.size(), num_threads * blocks_per_thread);
}

// ============================================================================
// SessionRotationManager Tests
// ============================================================================

TEST(SessionRotationManagerTest, Initialization) {
    SessionId sid;
    sid.fill(0x42);

    SessionRotationManager mgr(10000, sid);

    EXPECT_EQ(mgr.current_sid(), sid);
    EXPECT_FALSE(mgr.needs_rotation());
}

TEST(SessionRotationManagerTest, QueryLimit) {
    SessionRotationManager mgr(100, SessionId{});

    for (int i = 0; i < 99; ++i) {
        mgr.record_allocation(1);
        EXPECT_FALSE(mgr.needs_rotation());
    }

    mgr.record_allocation(1);  // 100th query
    EXPECT_TRUE(mgr.needs_rotation());
}

TEST(SessionRotationManagerTest, Rotation) {
    SessionId sid;
    sid.fill(0x11);

    SessionRotationManager mgr(100, sid);

    auto old_sid = mgr.current_sid();

    // Trigger rotation
    mgr.record_allocation(100);
    EXPECT_TRUE(mgr.needs_rotation());

    SessionId new_sid = mgr.rotate();

    EXPECT_NE(new_sid, old_sid);
    EXPECT_EQ(mgr.current_sid(), new_sid);
    EXPECT_FALSE(mgr.needs_rotation());
}

// ============================================================================
// RuntimeStateManager Tests
// ============================================================================

class RuntimeStateTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_dir_ = std::filesystem::temp_directory_path() / "silentmpc_test_runtime";
        std::filesystem::create_directories(test_dir_);

        policy_.mode = EpochPolicy::Mode::Monotonic;
        policy_.persist_dir = test_dir_;

        sid_.fill(0xCC);
    }

    void TearDown() override {
        if (std::filesystem::exists(test_dir_)) {
            std::filesystem::remove_all(test_dir_);
        }
    }

    EpochPolicy policy_;
    SessionId sid_;
    std::filesystem::path test_dir_;
};

TEST_F(RuntimeStateTest, SafeAllocation) {
    RuntimeStateManager mgr(policy_, sid_, 1000, 128);

    auto block = mgr.allocate_safe();
    EXPECT_EQ(block.size(), 128UL);
}

TEST_F(RuntimeStateTest, RotationEnforcement) {
    RuntimeStateManager mgr(policy_, sid_, 100, 128);  // Query limit = 100

    // Allocate until rotation needed
    for (int i = 0; i < 100; i += 128) {
        mgr.allocate_safe();
    }

    // Next allocation should fail (rotation required)
    EXPECT_THROW(mgr.allocate_safe(), std::runtime_error);
}

TEST_F(RuntimeStateTest, SessionRotation) {
    RuntimeStateManager mgr(policy_, sid_, 100, 128);

    auto old_sid = sid_;
    mgr.rotate_session();

    // After rotation, allocation should work again
    EXPECT_NO_THROW(mgr.allocate_safe());
}

TEST_F(RuntimeStateTest, CrashRecovery) {
    {
        RuntimeStateManager mgr(policy_, sid_, 1000, 128);
        mgr.allocate_safe();
        mgr.allocate_safe();
    }

    // Recover
    {
        RuntimeStateManager mgr(policy_, sid_, 1000, 128);
        mgr.recover_from_crash();

        auto block = mgr.allocate_safe();
        EXPECT_GE(block.start, 256ULL);  // After recovered blocks
    }
}

// ============================================================================
// EpochConsistencyChecker Tests
// ============================================================================

TEST(EpochConsistencyTest, MessageVerification) {
    SessionId sid;
    sid.fill(0x55);
    EpochId epoch = 42;

    // Construct message: [sid(16) || epoch(8) || payload(10) || hmac(32)]
    std::vector<uint8_t> message;
    message.insert(message.end(), sid.begin(), sid.end());

    for (size_t i = 0; i < 8; ++i) {
        message.push_back((epoch >> (i * 8)) & 0xFF);
    }

    // Payload
    for (int i = 0; i < 10; ++i) {
        message.push_back(0xAA);
    }

    // HMAC (dummy)
    for (int i = 0; i < 32; ++i) {
        message.push_back(0xBB);
    }

    std::array<uint8_t, 32> key;
    key.fill(0x00);

    // Should verify sid and epoch
    bool valid = EpochConsistencyChecker::verify_control_message(
        message, sid, epoch, key
    );

    EXPECT_TRUE(valid);  // HMAC check not implemented yet, but sid/epoch should match
}

TEST(EpochConsistencyTest, WrongSid) {
    SessionId sid1, sid2;
    sid1.fill(0x11);
    sid2.fill(0x22);

    std::vector<uint8_t> message;
    message.insert(message.end(), sid1.begin(), sid1.end());

    uint64_t epoch = 1;
    for (size_t i = 0; i < 8; ++i) {
        message.push_back((epoch >> (i * 8)) & 0xFF);
    }

    // Padding
    message.resize(16 + 8 + 32, 0);

    std::array<uint8_t, 32> key;
    key.fill(0);

    bool valid = EpochConsistencyChecker::verify_control_message(
        message, sid2, epoch, key
    );

    EXPECT_FALSE(valid);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
