# Performance benchmarks (Google Benchmark framework) - Optional
find_package(benchmark QUIET)
if(benchmark_FOUND)
    add_executable(bench_algebra bench_algebra.cpp)
    target_link_libraries(bench_algebra PRIVATE silentmpc_lib ${OPENSSL_LIBRARIES} Threads::Threads benchmark::benchmark)

    add_executable(bench_pcf bench_pcf.cpp)
    target_link_libraries(bench_pcf PRIVATE silentmpc_lib ${OPENSSL_LIBRARIES} Threads::Threads benchmark::benchmark)

    message(STATUS "Google Benchmark found - building performance benchmarks")
else()
    message(STATUS "Google Benchmark not found - skipping bench_algebra and bench_pcf")
endif()

# Paper reproducibility benchmarks (standalone programs)
add_executable(setup_cost_benchmark setup_cost_benchmark.cpp)
target_link_libraries(setup_cost_benchmark PRIVATE silentmpc_lib ${OPENSSL_LIBRARIES} Threads::Threads)

add_executable(tile_sensitivity_benchmark tile_sensitivity_benchmark.cpp)
target_link_libraries(tile_sensitivity_benchmark PRIVATE silentmpc_lib ${OPENSSL_LIBRARIES} Threads::Threads)

add_executable(rollback_safety_benchmark rollback_safety_benchmark.cpp)
target_link_libraries(rollback_safety_benchmark PRIVATE silentmpc_lib ${OPENSSL_LIBRARIES} Threads::Threads)
