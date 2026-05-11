if(HOMA_BUILD_TESTS)
    add_executable(test_patch_ordering_etree test_patch_ordering_etree.cpp)
    target_link_libraries(test_patch_ordering_etree PRIVATE homa GTest::gtest_main)
    target_include_directories(test_patch_ordering_etree PRIVATE
        ${CMAKE_SOURCE_DIR}/include
    )
    gtest_discover_tests(test_patch_ordering_etree)
endif()
