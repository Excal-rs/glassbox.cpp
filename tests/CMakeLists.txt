add_executable(test_token test_token.cpp)
target_link_libraries(test_token PRIVATE glassbox)

# token_test_data.h lives next to this test.
target_include_directories(test_token PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Absolute path to the GPT-2 model files, so the test finds them from any cwd.
target_compile_definitions(test_token PRIVATE MODEL_DIR="${CMAKE_SOURCE_DIR}/model")

add_test(NAME token_encode COMMAND test_token)
