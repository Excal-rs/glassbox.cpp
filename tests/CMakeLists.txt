add_executable(test_token test_token.cpp)
target_link_libraries(test_token PRIVATE glassbox)

# token_test_data.h lives next to this test.
target_include_directories(test_token PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_test(NAME token_encode COMMAND test_token)
