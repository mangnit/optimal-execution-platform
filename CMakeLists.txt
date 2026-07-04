cmake_minimum_required(VERSION 3.20)
project(optimal_execution_platform LANGUAGES CXX)

# Two-clocks discipline (docs/architecture.md §0): everything under cpp/ that
# will one day sit on the internal clock compiles with the same flags across
# builds, so benchmarks stay comparable and -Werror stays honest.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

add_compile_options(-Wall -Wextra -Wpedantic -Werror)
# Release builds get -O3 -march=native (docs/architecture.md). Debug keeps defaults so
# assertions in the slab allocator can be exercised by tests.
add_compile_options($<$<CONFIG:Release>:-O3>)
add_compile_options($<$<CONFIG:Release>:-march=native>)

# Public headers live under cpp/. Consumers include as "core/slab_allocator.hpp".
add_library(oep_core INTERFACE)
target_include_directories(oep_core INTERFACE ${CMAKE_SOURCE_DIR}/cpp)

enable_testing()
add_subdirectory(cpp/tests)

# Latency benchmarks (Google Benchmark) — internal-clock hot-path only,
# per docs/architecture.md §7. Opt in with -DOEP_BUILD_BENCHMARKS=ON so a
# vanilla test-only build stays fast.
option(OEP_BUILD_BENCHMARKS "Build the internal-clock latency benchmarks" ON)
if(OEP_BUILD_BENCHMARKS)
  add_subdirectory(cpp/benchmarks)
endif()
