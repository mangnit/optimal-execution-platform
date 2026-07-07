include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
  URL_HASH SHA256=8ad598c73ad796e0d8280b082cebd82a630d73e73cd3c70057938a6501bba5d7
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(slab_allocator_test slab_allocator_test.cpp)
target_link_libraries(slab_allocator_test
  PRIVATE
    oep_core
    GTest::gtest
    GTest::gtest_main
    pthread
)

add_executable(spsc_ring_test spsc_ring_test.cpp)
target_link_libraries(spsc_ring_test
  PRIVATE
    oep_core
    GTest::gtest
    GTest::gtest_main
    pthread
)

add_executable(order_book_test order_book_test.cpp)
target_link_libraries(order_book_test
  PRIVATE
    oep_core
    GTest::gtest
    GTest::gtest_main
    pthread
)

add_executable(relay_test relay_test.cpp)
target_link_libraries(relay_test
  PRIVATE
    oep_core
    GTest::gtest
    GTest::gtest_main
    pthread
)

add_executable(system_integration_test system_integration_test.cpp)
target_link_libraries(system_integration_test
  PRIVATE
    oep_core
    GTest::gtest
    GTest::gtest_main
    pthread
)

add_executable(almgren_chriss_test almgren_chriss_test.cpp)
target_link_libraries(almgren_chriss_test
  PRIVATE
    oep_core
    oep_exec
    GTest::gtest
    GTest::gtest_main
    pthread
)

add_executable(vwap_test vwap_test.cpp)
target_link_libraries(vwap_test
  PRIVATE
    oep_core
    oep_exec
    GTest::gtest
    GTest::gtest_main
    pthread
)

# RL policy test (task P5.6) — gated on OEP_USE_TORCH so a non-LibTorch
# build stays green. Links torch transitively through oep_exec's PUBLIC
# link when OEP_USE_TORCH=ON.
if(OEP_USE_TORCH)
  add_executable(rl_policy_test rl_policy_test.cpp)
  target_link_libraries(rl_policy_test
    PRIVATE
      oep_core
      oep_exec
      GTest::gtest
      GTest::gtest_main
      pthread
  )
  set_source_files_properties(rl_policy_test.cpp PROPERTIES
    COMPILE_OPTIONS "-Wno-sign-compare;-Wno-deprecated-declarations;-Wno-unused-parameter")
endif()

include(GoogleTest)
gtest_discover_tests(slab_allocator_test)
gtest_discover_tests(spsc_ring_test)
gtest_discover_tests(order_book_test)
gtest_discover_tests(relay_test)
gtest_discover_tests(system_integration_test)
gtest_discover_tests(almgren_chriss_test)
gtest_discover_tests(vwap_test)
if(OEP_USE_TORCH)
  gtest_discover_tests(rl_policy_test)
endif()
