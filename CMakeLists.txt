cmake_minimum_required(VERSION 3.16)
project(glassbox CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to an optimized build — the forward pass is unusably slow at -O0.
# Override with -DCMAKE_BUILD_TYPE=Debug when debugging.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Emit build/compile_commands.json so editors/IntelliSense pick up include paths
# (e.g. fetched deps like nlohmann_json) without manual configuration.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Warning flags for our code only. Carried by an INTERFACE target rather than
# CMAKE_CXX_FLAGS so the fetched dependencies (pcre2, nlohmann_json) keep their
# own settings — their warnings aren't ours to fix, and burying real ones under
# third-party noise is how a warning-clean build stops meaning anything.
add_library(glassbox_warnings INTERFACE)
if(MSVC)
    target_compile_options(glassbox_warnings INTERFACE /W4)
else()
    target_compile_options(glassbox_warnings INTERFACE -Wall -Wextra)
endif()

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

# Header-only JSON; prefer a system install, otherwise fetch from source.
find_package(nlohmann_json QUIET)

if(NOT nlohmann_json_FOUND)
    message(STATUS "nlohmann_json not found on system; fetching from source")
    include(FetchContent)

    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

enable_testing()

add_subdirectory(src)
add_subdirectory(app)

# tests/ is kept out of the published repository, so only descend into it when
# it is actually there; a fresh clone would otherwise fail at configure time.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    add_subdirectory(tests)
endif()
