if(HOMA_BUILD_TESTS)
    add_executable(test_patch_ordering_etree test_patch_ordering_etree.cpp)
    target_link_libraries(test_patch_ordering_etree PRIVATE homa GTest::gtest_main)
    target_include_directories(test_patch_ordering_etree PRIVATE
        ${HOMA_SOURCE_DIR}/include
    )
    if(COMMAND homa_configure_mkl_runtime)
        homa_configure_mkl_runtime(test_patch_ordering_etree)
    endif()
    if(COMMAND homa_configure_cudss_runtime)
        homa_configure_cudss_runtime(test_patch_ordering_etree)
    endif()
    gtest_discover_tests(test_patch_ordering_etree)
endif()
