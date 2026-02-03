#include <benchmark/benchmark.h>
#include "silentmpc/pcf/pcf_core.hpp"
#include "silentmpc/algebra/gf2.hpp"
#include "silentmpc/algebra/z2k.hpp"

using namespace silentmpc;
using namespace silentmpc::pcf;
using namespace silentmpc::algebra;

// ============================================================================
// PCF Generation Benchmarks
// ============================================================================

static void BM_PCF_DenseGeneration_GF2(benchmark::State& state) {
    PCFGenerator<GF2> gen;
    SessionId sid;
    sid.fill(0x42);
    gen.initialize(sid, 0, 0, PCFParameters::B1);

    for (auto _ : state) {
        auto dense = gen.generate_dense_vector(0, 0);
        benchmark::DoNotOptimize(dense);
    }

    state.SetItemsProcessed(state.iterations() * PCFParameters::B1.N);
    state.SetBytesProcessed(state.iterations() * PCFParameters::B1.N / 8);
}
BENCHMARK(BM_PCF_DenseGeneration_GF2);

static void BM_PCF_DenseGeneration_Z264(benchmark::State& state) {
    PCFGenerator<Z2k<64>> gen;
    SessionId sid;
    sid.fill(0x42);
    gen.initialize(sid, 0, 0, PCFParameters::R2);

    for (auto _ : state) {
        auto dense = gen.generate_dense_vector(0, 0);
        benchmark::DoNotOptimize(dense);
    }

    state.SetItemsProcessed(state.iterations() * PCFParameters::R2.N);
    state.SetBytesProcessed(state.iterations() * PCFParameters::R2.N * 8);
}
BENCHMARK(BM_PCF_DenseGeneration_Z264);

static void BM_PCF_SparseGeneration_GF2(benchmark::State& state) {
    PCFGenerator<GF2> gen;
    SessionId sid;
    sid.fill(0x42);
    gen.initialize(sid, 0, 0, PCFParameters::B1);

    for (auto _ : state) {
        auto indices = gen.generate_sparse_indices(0, 0);
        auto values = gen.generate_sparse_values(0, 0);
        benchmark::DoNotOptimize(indices);
        benchmark::DoNotOptimize(values);
    }

    state.SetItemsProcessed(state.iterations() * PCFParameters::B1.tau);
}
BENCHMARK(BM_PCF_SparseGeneration_GF2);

static void BM_PCF_SparseGeneration_Z264(benchmark::State& state) {
    PCFGenerator<Z2k<64>> gen;
    SessionId sid;
    sid.fill(0x42);
    gen.initialize(sid, 0, 0, PCFParameters::R2);

    for (auto _ : state) {
        auto indices = gen.generate_sparse_indices(0, 0);
        auto values = gen.generate_sparse_values(0, 0);
        benchmark::DoNotOptimize(indices);
        benchmark::DoNotOptimize(values);
    }

    state.SetItemsProcessed(state.iterations() * PCFParameters::R2.tau);
}
BENCHMARK(BM_PCF_SparseGeneration_Z264);

// ============================================================================
// Four-Term Computation Benchmarks
// ============================================================================

static void BM_FourTerm_Compute_GF2(benchmark::State& state) {
    size_t N = state.range(0);
    size_t tau = N / 200;  // ~0.5% noise rate

    std::vector<GF2> dense0(N, GF2(true));
    std::vector<GF2> dense1(N, GF2(false));

    std::vector<size_t> sparse0_idx, sparse1_idx;
    std::vector<GF2> sparse0_val, sparse1_val;

    for (size_t i = 0; i < tau; ++i) {
        sparse0_idx.push_back((i * N) / tau);
        sparse0_val.push_back(GF2(true));
        sparse1_idx.push_back((i * N) / tau + N / (2 * tau));
        sparse1_val.push_back(GF2(false));
    }

    for (auto _ : state) {
        auto result = FourTermComputer<GF2>::compute(
            dense0, dense1, sparse0_idx, sparse0_val, sparse1_idx, sparse1_val
        );
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * N);
}
BENCHMARK(BM_FourTerm_Compute_GF2)->Range(1024, 1 << 20);

static void BM_FourTerm_Compute_Z264(benchmark::State& state) {
    size_t N = state.range(0);
    size_t tau = N / 200;

    std::vector<Z2k<64>> dense0(N, Z2k<64>::from_uint64(123));
    std::vector<Z2k<64>> dense1(N, Z2k<64>::from_uint64(456));

    std::vector<size_t> sparse0_idx, sparse1_idx;
    std::vector<Z2k<64>> sparse0_val, sparse1_val;

    for (size_t i = 0; i < tau; ++i) {
        sparse0_idx.push_back((i * N) / tau);
        sparse0_val.push_back(Z2k<64>::from_uint64(789));
        sparse1_idx.push_back((i * N) / tau + N / (2 * tau));
        sparse1_val.push_back(Z2k<64>::from_uint64(101112));
    }

    for (auto _ : state) {
        auto result = FourTermComputer<Z2k<64>>::compute(
            dense0, dense1, sparse0_idx, sparse0_val, sparse1_idx, sparse1_val
        );
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * N);
    state.SetBytesProcessed(state.iterations() * N * 8);
}
BENCHMARK(BM_FourTerm_Compute_Z264)->Range(1024, 1 << 18);

// ============================================================================
// Tiled Computation Benchmarks
// ============================================================================

static void BM_FourTerm_Tiled_GF2(benchmark::State& state) {
    size_t tile_size = state.range(0);

    PCFGenerator<GF2> gen;
    SessionId sid;
    sid.fill(0x99);
    gen.initialize(sid, 0, 0, PCFParameters::B1);

    for (auto _ : state) {
        auto result = FourTermComputer<GF2>::compute_tiled(gen, tile_size);
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * PCFParameters::B1.N);
}
BENCHMARK(BM_FourTerm_Tiled_GF2)->RangeMultiplier(4)->Range(1024, 1 << 16);

static void BM_FourTerm_Tiled_Z264(benchmark::State& state) {
    size_t tile_size = state.range(0);

    PCFGenerator<Z2k<64>> gen;
    SessionId sid;
    sid.fill(0x99);
    gen.initialize(sid, 0, 0, PCFParameters::R2);

    for (auto _ : state) {
        auto result = FourTermComputer<Z2k<64>>::compute_tiled(gen, tile_size);
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * PCFParameters::R2.N);
}
BENCHMARK(BM_FourTerm_Tiled_Z264)->RangeMultiplier(4)->Range(1024, 1 << 14);

// ============================================================================
// Parameter Set Comparisons
// ============================================================================

static void BM_ParameterSets_B1(benchmark::State& state) {
    PCFGenerator<GF2> gen;
    SessionId sid;
    sid.fill(0xAA);
    gen.initialize(sid, 0, 0, PCFParameters::B1);

    for (auto _ : state) {
        auto dense = gen.generate_dense_vector(0, 0);
        auto indices = gen.generate_sparse_indices(0, 0);
        auto values = gen.generate_sparse_values(0, 0);
        benchmark::DoNotOptimize(dense);
        benchmark::DoNotOptimize(indices);
        benchmark::DoNotOptimize(values);
    }

    state.counters["N"] = PCFParameters::B1.N;
    state.counters["tau"] = PCFParameters::B1.tau;
}
BENCHMARK(BM_ParameterSets_B1);

static void BM_ParameterSets_B2(benchmark::State& state) {
    PCFGenerator<GF2> gen;
    SessionId sid;
    sid.fill(0xAA);
    gen.initialize(sid, 0, 0, PCFParameters::B2);

    for (auto _ : state) {
        auto dense = gen.generate_dense_vector(0, 0);
        auto indices = gen.generate_sparse_indices(0, 0);
        auto values = gen.generate_sparse_values(0, 0);
        benchmark::DoNotOptimize(dense);
        benchmark::DoNotOptimize(indices);
        benchmark::DoNotOptimize(values);
    }

    state.counters["N"] = PCFParameters::B2.N;
    state.counters["tau"] = PCFParameters::B2.tau;
}
BENCHMARK(BM_ParameterSets_B2);

static void BM_ParameterSets_R2(benchmark::State& state) {
    PCFGenerator<Z2k<64>> gen;
    SessionId sid;
    sid.fill(0xAA);
    gen.initialize(sid, 0, 0, PCFParameters::R2);

    for (auto _ : state) {
        auto dense = gen.generate_dense_vector(0, 0);
        auto indices = gen.generate_sparse_indices(0, 0);
        auto values = gen.generate_sparse_values(0, 0);
        benchmark::DoNotOptimize(dense);
        benchmark::DoNotOptimize(indices);
        benchmark::DoNotOptimize(values);
    }

    state.counters["N"] = PCFParameters::R2.N;
    state.counters["tau"] = PCFParameters::R2.tau;
}
BENCHMARK(BM_ParameterSets_R2);

// ============================================================================
// Memory Bandwidth Tests
// ============================================================================

static void BM_Memory_DenseAccess_GF2(benchmark::State& state) {
    size_t N = 1 << 20;
    std::vector<GF2> vec(N, GF2(true));

    for (auto _ : state) {
        GF2 sum = GF2::zero();
        for (size_t i = 0; i < N; ++i) {
            sum = sum + vec[i];
        }
        benchmark::DoNotOptimize(sum);
    }

    state.SetBytesProcessed(state.iterations() * N / 8);
}
BENCHMARK(BM_Memory_DenseAccess_GF2);

static void BM_Memory_DenseAccess_Z264(benchmark::State& state) {
    size_t N = 1 << 18;
    std::vector<Z2k<64>> vec(N, Z2k<64>::from_uint64(42));

    for (auto _ : state) {
        Z2k<64> sum = Z2k<64>::zero();
        for (size_t i = 0; i < N; ++i) {
            sum = sum + vec[i];
        }
        benchmark::DoNotOptimize(sum);
    }

    state.SetBytesProcessed(state.iterations() * N * 8);
}
BENCHMARK(BM_Memory_DenseAccess_Z264);

static void BM_Memory_SparseAccess_Z264(benchmark::State& state) {
    size_t N = 1 << 18;
    size_t tau = 256;

    std::vector<Z2k<64>> dense(N, Z2k<64>::from_uint64(123));
    std::vector<size_t> indices;
    for (size_t i = 0; i < tau; ++i) {
        indices.push_back((i * N) / tau);
    }

    for (auto _ : state) {
        Z2k<64> sum = Z2k<64>::zero();
        for (size_t idx : indices) {
            sum = sum + dense[idx];
        }
        benchmark::DoNotOptimize(sum);
    }

    state.SetItemsProcessed(state.iterations() * tau);
}
BENCHMARK(BM_Memory_SparseAccess_Z264);

BENCHMARK_MAIN();
