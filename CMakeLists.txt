cmake_minimum_required(VERSION 3.20)
project(SilentMPC VERSION 1.0.0 LANGUAGES CXX)

# Require C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization options
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG -flto")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fsanitize=address -fsanitize=undefined")

# Enable Link-Time Optimization (LTO)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)

# AVX-512 support (required for verification)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)
if(COMPILER_SUPPORTS_AVX512)
    add_compile_options(-mavx512f -mavx512bw -mavx512vl -mavx512dq)
    message(STATUS "AVX-512 support enabled for paper verification")
else()
    message(WARNING "AVX-512 not supported - falling back to scalar implementation")
endif()

# Warning options
add_compile_options(
    -Wall -Wextra -Wpedantic
    -Wconversion -Wsign-conversion
    -Wno-unused-parameter
)

# External dependencies
find_package(OpenSSL REQUIRED)
find_package(Threads REQUIRED)

# Directory structure
include_directories(${CMAKE_SOURCE_DIR}/include)

# Library sources (including AVX-512 optimizations)
set(SILENTMPC_SOURCES
    src/core/types.cpp
    src/core/algebra/gf2.cpp
    src/core/algebra/gfq.cpp
    src/core/algebra/z2k.cpp
    src/pcf/pcf_core.cpp
    src/selective_delivery/ot_backend.cpp
    src/selective_delivery/dpf_backend.cpp
    src/runtime/epoch_manager.cpp
    src/runtime/allocator.cpp
    src/network/connection.cpp
    src/network/message.cpp
    src/network/two_party_network.cpp
)

# Main library
add_library(silentmpc_lib STATIC ${SILENTMPC_SOURCES})
target_link_libraries(silentmpc_lib
    PUBLIC OpenSSL::Crypto
    PUBLIC Threads::Threads
)
target_include_directories(silentmpc_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Alias for compatibility
add_library(silentmpc ALIAS silentmpc_lib)

# Tests and benchmarks
enable_testing()
add_subdirectory(tests)
add_subdirectory(benchmarks)

# Installation
install(TARGETS silentmpc_lib
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)
