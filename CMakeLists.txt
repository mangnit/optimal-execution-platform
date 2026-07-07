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

# Execution baselines (Almgren-Chriss, VWAP — task P5). Static library because
# the schedule ctors carry `<cmath>` and non-trivial state; keeping the .cpp
# out of headers lets the CRTP call sites in strategy.hpp stay header-only.
add_library(oep_exec STATIC
  cpp/exec/almgren_chriss.cpp
  cpp/exec/vwap.cpp
)
target_link_libraries(oep_exec PUBLIC oep_core)

# LibTorch RL inference (task P5.6). Off by default so a vanilla
# `-DOEP_USE_TORCH=OFF` checkout stays fast and CI does not need the
# ~2 GiB libtorch drop on disk. Turn on with:
#   -DOEP_USE_TORCH=ON -DCMAKE_PREFIX_PATH=/path/to/libtorch
# Once enabled, rl_policy.cpp joins the oep_exec library and every
# consumer (tests, benchmarks, sim_runner) transitively links torch.
option(OEP_USE_TORCH "Build the LibTorch RL inference module (P5.6)" OFF)
if(OEP_USE_TORCH)
  find_package(Torch REQUIRED)
  target_sources(oep_exec PRIVATE cpp/exec/rl_policy.cpp)
  target_link_libraries(oep_exec PUBLIC "${TORCH_LIBRARIES}")
  target_compile_definitions(oep_exec PUBLIC OEP_HAS_TORCH=1)
  # LibTorch headers pull in a lot of code that trips -Werror under
  # -Wall -Wextra -Wpedantic (sign-compare, deprecated-decl, unused-parameter).
  # Scope the relaxation to the single translation unit that actually
  # includes <torch/script.h>.
  set_source_files_properties(cpp/exec/rl_policy.cpp PROPERTIES
    COMPILE_OPTIONS "-Wno-sign-compare;-Wno-deprecated-declarations;-Wno-unused-parameter")
endif()

enable_testing()
add_subdirectory(cpp/tests)

# Latency benchmarks (Google Benchmark) — internal-clock hot-path only,
# per docs/architecture.md §7. Opt in with -DOEP_BUILD_BENCHMARKS=ON so a
# vanilla test-only build stays fast.
option(OEP_BUILD_BENCHMARKS "Build the internal-clock latency benchmarks" ON)
if(OEP_BUILD_BENCHMARKS)
  add_subdirectory(cpp/benchmarks)
endif()

# KDB+ IPC logger (task P2_kdb_ipc). Off by default so CI stays green without
# k.h / c.o present and without a running q ticker plant. Turn on with
# -DOEP_USE_KDB=ON once cpp/external/kdb/{k.h,c.o} are downloaded (see
# the KX Systems kdb repository for the k.h/c.o downloads).
option(OEP_USE_KDB "Build the real KDB+ IPC logger (requires cpp/external/kdb/{k.h,c.o})" OFF)
if(OEP_USE_KDB)
  set(OEP_KDB_C_OBJ "${CMAKE_SOURCE_DIR}/cpp/external/kdb/c.o")
  set(OEP_KDB_HEADER "${CMAKE_SOURCE_DIR}/cpp/external/kdb/k.h")
  if(NOT EXISTS "${OEP_KDB_C_OBJ}" OR NOT EXISTS "${OEP_KDB_HEADER}")
    message(FATAL_ERROR
      "OEP_USE_KDB=ON but cpp/external/kdb/{k.h,c.o} are missing. "
      "See the KX Systems kdb repository for the k.h/c.o downloads.")
  endif()

  add_library(oep_kdb_ipc STATIC cpp/external/ipc_kdb_logger.cpp)
  target_include_directories(oep_kdb_ipc
    PUBLIC
      ${CMAKE_SOURCE_DIR}/cpp
    PRIVATE
      ${CMAKE_SOURCE_DIR}/cpp/external/kdb
  )
  # `c.o` is the prebuilt KX Systems Linux 64-bit client object. Linking it as
  # a plain object file works for GCC/Ninja; it pulls in libc symbols only.
  target_link_libraries(oep_kdb_ipc
    PUBLIC
      oep_core
      ${OEP_KDB_C_OBJ}
      pthread
  )
  target_compile_definitions(oep_kdb_ipc PUBLIC OEP_USE_KDB=1 KXVER=3)
endif()

# Sim runner (task P3_sim_runner). Drives a deterministic synthetic workload
# through OrderBook → TelemetryPublisher → SpscRing → EventRelay → KdbLogger
# with the production TSC clock plumbed into TelemetryPublisher::ClockFn.
# Links against the real IPC logger when OEP_USE_KDB=ON; otherwise falls back
# to the in-memory sink so a `docker compose`-less checkout can still `run`.
add_executable(sim_runner cpp/main/sim_runner.cpp)
target_link_libraries(sim_runner PRIVATE oep_core pthread)
if(OEP_USE_KDB)
  target_link_libraries(sim_runner PRIVATE oep_kdb_ipc)
endif()

# Pybind11 bridge (task P5.2). Off by default so a vanilla `-DOEP_BUILD_PYBIND=OFF`
# checkout stays fast and CI does not need Python/pybind11 on PATH. Turn on with
# -DOEP_BUILD_PYBIND=ON (and optionally -DPython_EXECUTABLE=... to pin the
# interpreter). Produces a shared module named `oep_env` that Python can import.
option(OEP_BUILD_PYBIND "Build the pybind11 bridge module oep_env" OFF)
if(OEP_BUILD_PYBIND)
  # Prefer the Development.Module component — enough to build a Python
  # extension without needing the full Python.h shared-library link.
  find_package(Python 3.11 COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)

  pybind11_add_module(oep_env MODULE cpp/bindings/py_module.cpp)
  # Bridge is on the boundary between clocks — the same header set as sim_runner
  # is required (order book, ring, publisher, relay, TSC clock).
  target_link_libraries(oep_env PRIVATE oep_core pthread)
  # -Werror is inherited from add_compile_options above; pybind11's own headers
  # are clean under it. Suppress the visibility warning the pybind11 harness
  # emits when combined with GCC's -Wextra so -Werror does not trip on it.
  target_compile_options(oep_env PRIVATE -fvisibility=hidden)
endif()
