#include <benchmark/benchmark.h>
#include "silentmpc/algebra/gf2.hpp"
#include "silentmpc/algebra/z2k.hpp"
#include "silentmpc/algebra/gfq.hpp"

using namespace silentmpc::algebra;

// ============================================================================
// GF(2) Benchmarks
// ============================================================================

static void BM_GF2_Addition(benchmark::State& state) {
    GF2 a(true);
    GF2 b(false);

    for (auto _ : state) {
        GF2 c = a + b;
        benchmark::DoNotOptimize(c);
    }
}
BENCHMARK(BM_GF2_Addition);

static void BM_GF2_Multiplication(benchmark::State& state) {
    GF2 a(true);
    GF2 b(true);

    for (auto _ : state) {
        GF2 c = a * b;
        benchmark::DoNotOptimize(c);
    }
}
BENCHMARK(BM_GF2_Multiplication);

static void BM_GF2Vector_Addition(benchmark::State& state) {
    size_t size = state.range(0);
    GF2Vector v1(size);
    GF2Vector v2(size);

    // Initialize with pattern
    for (size_t i = 0; i < size; i += 2) {
        v1.set(i, true);
        v2.set(i + 1, true);
    }

    for (auto _ : state) {
        GF2Vector v3 = v1 + v2;
        benchmark::DoNotOptimize(v3);
    }

    state.SetItemsProcessed(state.iterations() * size);
    state.SetBytesProcessed(state.iterations() * size / 8);
}
BENCHMARK(BM_GF2Vector_Addition)->Range(1024, 1 << 20);

static void BM_GF2Vector_Hadamard(benchmark::State& state) {
    size_t size = state.range(0);
    GF2Vector v1(size);
    GF2Vector v2(size);

    for (size_t i = 0; i < size; i += 2) {
        v1.set(i, true);
        v2.set(i, true);
    }

    for (auto _ : state) {
        GF2Vector v3 = v1.hadamard(v2);
        benchmark::DoNotOptimize(v3);
    }

    state.SetItemsProcessed(state.iterations() * size);
}
BENCHMARK(BM_GF2Vector_Hadamard)->Range(1024, 1 << 20);

static void BM_GF2Vector_InnerProduct(benchmark::State& state) {
    size_t size = state.range(0);
    GF2Vector v1(size);
    GF2Vector v2(size);

    for (size_t i = 0; i < size; i += 3) {
        v1.set(i, true);
        v2.set(i, true);
    }

    for (auto _ : state) {
        bool result = v1.inner_product(v2);
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * size);
}
BENCHMARK(BM_GF2Vector_InnerProduct)->Range(1024, 1 << 20);

static void BM_GF2_MatrixVectorMul(benchmark::State& state) {
    size_t N = state.range(0);
    size_t n = N / 4;

    std::vector<uint64_t> A((N * n + 63) / 64, 0);
    std::vector<uint64_t> s((n + 63) / 64, 0xAAAAAAAAAAAAAAAAULL);
    std::vector<uint64_t> y((N + 63) / 64, 0);

    // Initialize A with some pattern
    for (size_t i = 0; i < A.size(); ++i) {
        A[i] = 0x5555555555555555ULL;
    }

    for (auto _ : state) {
        GF2_AVX512::matrix_vector_mul(A, s, y, N, n);
        benchmark::DoNotOptimize(y);
    }

    state.SetItemsProcessed(state.iterations() * N * n);
    state.SetBytesProcessed(state.iterations() * (N * n / 8));
}
BENCHMARK(BM_GF2_MatrixVectorMul)->Range(256, 4096);

static void BM_GF2_SparseInnerProduct(benchmark::State& state) {
    size_t dense_len = 1 << 20;
    size_t sparse_count = state.range(0);

    std::vector<uint64_t> dense_vec(dense_len / 64, 0xAAAAAAAAAAAAAAAAULL);
    std::vector<size_t> indices;

    for (size_t i = 0; i < sparse_count; ++i) {
        indices.push_back((i * dense_len) / sparse_count);
    }

    for (auto _ : state) {
        bool result = GF2_AVX512::sparse_inner_product(indices, dense_vec, dense_len);
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * sparse_count);
}
BENCHMARK(BM_GF2_SparseInnerProduct)->Range(64, 1024);

// ============================================================================
// Z2k Benchmarks
// ============================================================================

static void BM_Z2k64_Addition(benchmark::State& state) {
    Z2k<64> a = Z2k<64>::from_uint64(0x123456789ABCDEF0ULL);
    Z2k<64> b = Z2k<64>::from_uint64(0xFEDCBA9876543210ULL);

    for (auto _ : state) {
        Z2k<64> c = a + b;
        benchmark::DoNotOptimize(c);
    }
}
BENCHMARK(BM_Z2k64_Addition);

static void BM_Z2k64_Multiplication(benchmark::State& state) {
    Z2k<64> a = Z2k<64>::from_uint64(0x123456789ABCDEF0ULL);
    Z2k<64> b = Z2k<64>::from_uint64(0xFEDCBA9876543210ULL);

    for (auto _ : state) {
        Z2k<64> c = a * b;
        benchmark::DoNotOptimize(c);
    }
}
BENCHMARK(BM_Z2k64_Multiplication);

static void BM_Z2k32_Addition(benchmark::State& state) {
    Z2k<32> a = Z2k<32>::from_uint64(0x12345678);
    Z2k<32> b = Z2k<32>::from_uint64(0xFEDCBA98);

    for (auto _ : state) {
        Z2k<32> c = a + b;
        benchmark::DoNotOptimize(c);
    }
}
BENCHMARK(BM_Z2k32_Addition);

static void BM_Z2k32_Multiplication(benchmark::State& state) {
    Z2k<32> a = Z2k<32>::from_uint64(0x12345678);
    Z2k<32> b = Z2k<32>::from_uint64(0xFEDCBA98);

    for (auto _ : state) {
        Z2k<32> c = a * b;
        benchmark::DoNotOptimize(c);
    }
}
BENCHMARK(BM_Z2k32_Multiplication);

static void BM_Z2k128_Addition(benchmark::State& state) {
    Z2k<128> a = Z2k<128>::from_uint64(0xFFFFFFFFFFFFFFFFULL);
    Z2k<128> b = Z2k<128>::from_uint64(0xFEDCBA9876543210ULL);

    for (auto _ : state) {
        Z2k<128> c = a + b;
        benchmark::DoNotOptimize(c);
    }
}
BENCHMARK(BM_Z2k128_Addition);

static void BM_Z2k128_Multiplication(benchmark::State& state) {
    Z2k<128> a = Z2k<128>::from_uint64(0xFFFFFFFFFFFFFFFFULL);
    Z2k<128> b = Z2k<128>::from_uint64(0xFEDCBA9876543210ULL);

    for (auto _ : state) {
        Z2k<128> c = a * b;
        benchmark::DoNotOptimize(c);
    }
}
BENCHMARK(BM_Z2k128_Multiplication);

static void BM_Z2k64_VectorAddition(benchmark::State& state) {
    size_t size = state.range(0);
    std::vector<Z2k<64>> v1(size, Z2k<64>::from_uint64(123));
    std::vector<Z2k<64>> v2(size, Z2k<64>::from_uint64(456));
    std::vector<Z2k<64>> result(size);

    for (auto _ : state) {
        for (size_t i = 0; i < size; ++i) {
            result[i] = v1[i] + v2[i];
        }
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * size);
    state.SetBytesProcessed(state.iterations() * size * 8);
}
BENCHMARK(BM_Z2k64_VectorAddition)->Range(1024, 1 << 18);

static void BM_Z2k64_VectorMultiplication(benchmark::State& state) {
    size_t size = state.range(0);
    std::vector<Z2k<64>> v1(size, Z2k<64>::from_uint64(123));
    std::vector<Z2k<64>> v2(size, Z2k<64>::from_uint64(456));
    std::vector<Z2k<64>> result(size);

    for (auto _ : state) {
        for (size_t i = 0; i < size; ++i) {
            result[i] = v1[i] * v2[i];
        }
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * size);
    state.SetBytesProcessed(state.iterations() * size * 8);
}
BENCHMARK(BM_Z2k64_VectorMultiplication)->Range(1024, 1 << 18);

// ============================================================================
// GF(q) Benchmarks
// ============================================================================

static void BM_GFq127_Addition(benchmark::State& state) {
    GFq<127> a = GFq<127>::from_uint64(100);
    GFq<127> b = GFq<127>::from_uint64(50);

    for (auto _ : state) {
        GFq<127> c = a + b;
        benchmark::DoNotOptimize(c);
    }
}
BENCHMARK(BM_GFq127_Addition);

static void BM_GFq127_Multiplication(benchmark::State& state) {
    GFq<127> a = GFq<127>::from_uint64(100);
    GFq<127> b = GFq<127>::from_uint64(50);

    for (auto _ : state) {
        GFq<127> c = a * b;
        benchmark::DoNotOptimize(c);
    }
}
BENCHMARK(BM_GFq127_Multiplication);

static void BM_GFqMersenne_Addition(benchmark::State& state) {
    // 2^61 - 1
    using GFMersenne = GFq<2305843009213693951ULL>;

    GFMersenne a = GFMersenne::from_uint64(1000000000000ULL);
    GFMersenne b = GFMersenne::from_uint64(2000000000000ULL);

    for (auto _ : state) {
        GFMersenne c = a + b;
        benchmark::DoNotOptimize(c);
    }
}
BENCHMARK(BM_GFqMersenne_Addition);

static void BM_GFqMersenne_Multiplication(benchmark::State& state) {
    using GFMersenne = GFq<2305843009213693951ULL>;

    GFMersenne a = GFMersenne::from_uint64(1000000000000ULL);
    GFMersenne b = GFMersenne::from_uint64(2000000000000ULL);

    for (auto _ : state) {
        GFMersenne c = a * b;
        benchmark::DoNotOptimize(c);
    }
}
BENCHMARK(BM_GFqMersenne_Multiplication);

static void BM_GFq_VectorAddition(benchmark::State& state) {
    size_t size = state.range(0);
    std::vector<GFq<127>> v1(size, GFq<127>::from_uint64(100));
    std::vector<GFq<127>> v2(size, GFq<127>::from_uint64(50));
    std::vector<GFq<127>> result(size);

    for (auto _ : state) {
        for (size_t i = 0; i < size; ++i) {
            result[i] = v1[i] + v2[i];
        }
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * size);
}
BENCHMARK(BM_GFq_VectorAddition)->Range(1024, 1 << 18);

static void BM_GFq_VectorMultiplication(benchmark::State& state) {
    size_t size = state.range(0);
    std::vector<GFq<127>> v1(size, GFq<127>::from_uint64(100));
    std::vector<GFq<127>> v2(size, GFq<127>::from_uint64(50));
    std::vector<GFq<127>> result(size);

    for (auto _ : state) {
        for (size_t i = 0; i < size; ++i) {
            result[i] = v1[i] * v2[i];
        }
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * size);
}
BENCHMARK(BM_GFq_VectorMultiplication)->Range(1024, 1 << 18);

// ============================================================================
// Comparison Benchmarks
// ============================================================================

static void BM_Compare_Addition_GF2(benchmark::State& state) {
    GF2 a(true), b(false);
    for (auto _ : state) {
        benchmark::DoNotOptimize(a + b);
    }
}
BENCHMARK(BM_Compare_Addition_GF2);

static void BM_Compare_Addition_Z264(benchmark::State& state) {
    Z2k<64> a = Z2k<64>::from_uint64(123);
    Z2k<64> b = Z2k<64>::from_uint64(456);
    for (auto _ : state) {
        benchmark::DoNotOptimize(a + b);
    }
}
BENCHMARK(BM_Compare_Addition_Z264);

static void BM_Compare_Addition_GFq(benchmark::State& state) {
    GFq<127> a = GFq<127>::from_uint64(100);
    GFq<127> b = GFq<127>::from_uint64(50);
    for (auto _ : state) {
        benchmark::DoNotOptimize(a + b);
    }
}
BENCHMARK(BM_Compare_Addition_GFq);

static void BM_Compare_Multiplication_GF2(benchmark::State& state) {
    GF2 a(true), b(true);
    for (auto _ : state) {
        benchmark::DoNotOptimize(a * b);
    }
}
BENCHMARK(BM_Compare_Multiplication_GF2);

static void BM_Compare_Multiplication_Z264(benchmark::State& state) {
    Z2k<64> a = Z2k<64>::from_uint64(123);
    Z2k<64> b = Z2k<64>::from_uint64(456);
    for (auto _ : state) {
        benchmark::DoNotOptimize(a * b);
    }
}
BENCHMARK(BM_Compare_Multiplication_Z264);

static void BM_Compare_Multiplication_GFq(benchmark::State& state) {
    GFq<127> a = GFq<127>::from_uint64(100);
    GFq<127> b = GFq<127>::from_uint64(50);
    for (auto _ : state) {
        benchmark::DoNotOptimize(a * b);
    }
}
BENCHMARK(BM_Compare_Multiplication_GFq);

BENCHMARK_MAIN();
