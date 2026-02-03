# Test executables (require GTest)
find_package(GTest QUIET)
if(GTest_FOUND)
    add_executable(test_algebra test_algebra.cpp)
    target_link_libraries(test_algebra PRIVATE silentmpc_lib ${OPENSSL_LIBRARIES} Threads::Threads GTest::gtest GTest::gtest_main)

    add_executable(test_pcf test_pcf.cpp)
    target_link_libraries(test_pcf PRIVATE silentmpc_lib ${OPENSSL_LIBRARIES} Threads::Threads GTest::gtest GTest::gtest_main)

    add_executable(test_runtime test_runtime.cpp)
    target_link_libraries(test_runtime PRIVATE silentmpc_lib ${OPENSSL_LIBRARIES} Threads::Threads GTest::gtest GTest::gtest_main)

    # Enable testing
    enable_testing()
    add_test(NAME AlgebraTests COMMAND test_algebra)
    add_test(NAME PCFTests COMMAND test_pcf)
    add_test(NAME RuntimeTests COMMAND test_runtime)

    message(STATUS "GTest found - building unit tests")
else()
    message(STATUS "GTest not found - skipping unit tests")
endif()
