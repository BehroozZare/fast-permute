if(HOMA_BUILD_TESTS)
    add_executable(test_patch_ordering_etree test_patch_ordering_etree.cpp)
    target_link_libraries(test_patch_ordering_etree PRIVATE homa GTest::gtest_main)
    target_include_directories(test_patch_ordering_etree PRIVATE
        ${HOMA_SOURCE_DIR}/include
    )
    homa_configure_runtime(test_patch_ordering_etree)
    gtest_discover_tests(test_patch_ordering_etree)

    add_executable(test_linsys_solver_api test_linsys_solver_api.cpp)
    target_link_libraries(test_linsys_solver_api PRIVATE homa GTest::gtest_main)
    target_include_directories(test_linsys_solver_api PRIVATE
        ${HOMA_SOURCE_DIR}/include
    )
    homa_configure_runtime(test_linsys_solver_api)
    gtest_discover_tests(test_linsys_solver_api)
endif()
