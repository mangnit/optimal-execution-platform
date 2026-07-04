include(FetchContent)

# Google Benchmark: pinned tag, matches the GoogleTest FetchContent pattern.
# BENCHMARK_ENABLE_TESTING is turned off so we don't drag in gtest again
# (googletest is already fetched by cpp/tests/CMakeLists.txt, but Google
# Benchmark's tests aren't ours to run).
FetchContent_Declare(
  benchmark
  URL https://github.com/google/benchmark/archive/refs/tags/v1.8.3.tar.gz
  URL_HASH SHA256=6bc180a57d23d4d9515519f92b0c83d61b05b5bab188961f36ac7b06b0d9e9ce
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

# Google Benchmark's own sources aren't -Werror-clean under our strict flags
# (deprecated-copy on gcc 11+). Suppress the escalation *just* for the
# benchmark target so we don't relax the flag for our code.
if(TARGET benchmark)
  target_compile_options(benchmark PRIVATE -Wno-error)
endif()
if(TARGET benchmark_main)
  target_compile_options(benchmark_main PRIVATE -Wno-error)
endif()

# One executable per component keeps failures triage-friendly and lets CI
# run them in parallel jobs when the time comes.
add_executable(slab_allocator_bench slab_allocator_bench.cpp)
target_link_libraries(slab_allocator_bench
  PRIVATE oep_core benchmark::benchmark pthread)

add_executable(spsc_ring_bench spsc_ring_bench.cpp)
target_link_libraries(spsc_ring_bench
  PRIVATE oep_core benchmark::benchmark pthread)

add_executable(order_book_bench order_book_bench.cpp)
target_link_libraries(order_book_bench
  PRIVATE oep_core benchmark::benchmark pthread)
