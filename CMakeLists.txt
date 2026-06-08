cmake_minimum_required(VERSION 3.16)
project(glassbox CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PCRE2 REQUIRED COMPONENTS 8BIT)

enable_testing()

add_subdirectory(src)
add_subdirectory(app)
add_subdirectory(tests)
