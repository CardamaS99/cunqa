# =====================================================================
#  C++ unit tests (configure with -DCUNQA_BUILD_TESTS=ON, run with ctest)
#  GoogleTest is fetched in cmake/dependencies/TestDeps.cmake.
# =====================================================================
include(GoogleTest)

add_executable(cunqa_cpp_tests
  test_random_delay.cpp
  test_channels.cpp
  test_comm_model.cpp
  test_qc_engine.cpp
)

target_include_directories(cunqa_cpp_tests
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

# logger_executor goes first: it provides the logger every library logs through.
target_link_libraries(cunqa_cpp_tests
  PRIVATE
    logger_executor
    GTest::gtest_main
    comm_model
    qc_engine
    noise
    timing
    AER_simulator_adapter
    cunqa_simulator_adapter
    munich_simulator_adapter
    qsim_simulator_adapter
    quest_simulator_adapter
    qulacs_simulator_adapter
)

gtest_discover_tests(cunqa_cpp_tests DISCOVERY_MODE PRE_TEST)
