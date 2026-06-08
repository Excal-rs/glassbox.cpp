add_library(glassbox
    token.cpp
)

# Exposes include/ to anything linking glassbox, so includes look like
# #include "glassbox/token.h".
target_include_directories(glassbox PUBLIC ${CMAKE_SOURCE_DIR}/include)


target_link_libraries(glassbox PRIVATE PCRE2::8BIT)
target_compile_definitions(glassbox PRIVATE PCRE2_CODE_UNIT_WIDTH=8)

