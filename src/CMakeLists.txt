add_library(glassbox
    token.cpp
)

# Exposes include/ to anything linking glassbox, so includes look like
# #include "glassbox/token.h".
target_include_directories(glassbox PUBLIC ${CMAKE_SOURCE_DIR}/include)
