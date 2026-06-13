cmake_minimum_required(VERSION 3.16)
project(glassbox CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Prefer a system-installed PCRE2; otherwise fetch and build it from source.
find_package(PCRE2 QUIET COMPONENTS 8BIT)

if(NOT PCRE2_FOUND)
    message(STATUS "PCRE2 not found on system; fetching from source")
    include(FetchContent)

    # Only build the 8-bit library we actually use.
    set(PCRE2_BUILD_PCRE2_8 ON CACHE BOOL "" FORCE)
    set(PCRE2_BUILD_PCRE2_16 OFF CACHE BOOL "" FORCE)
    set(PCRE2_BUILD_PCRE2_32 OFF CACHE BOOL "" FORCE)
    set(PCRE2_BUILD_PCRE2GREP OFF CACHE BOOL "" FORCE)
    set(PCRE2_BUILD_TESTS OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
        pcre2
        GIT_REPOSITORY https://github.com/PCRE2Project/pcre2.git
        GIT_TAG pcre2-10.44
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(pcre2)

    # Match the imported-target name find_package would have provided, so
    # consumers can link PCRE2::8BIT either way.
    if(TARGET pcre2-8-static)
        add_library(PCRE2::8BIT ALIAS pcre2-8-static)
    elseif(TARGET pcre2-8-shared)
        add_library(PCRE2::8BIT ALIAS pcre2-8-shared)
    endif()
endif()

enable_testing()

add_subdirectory(src)
add_subdirectory(app)
add_subdirectory(tests)
