add_executable(glassbox_cli main.cpp input.cpp)
target_link_libraries(glassbox_cli PRIVATE glassbox glassbox_warnings)
