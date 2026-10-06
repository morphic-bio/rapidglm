cmake_minimum_required(VERSION 3.16)
project(rapidglm_consumer LANGUAGES C CXX)
find_package(rapidglm CONFIG REQUIRED)
add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE rapidglm::rapidglm)
# A C program linked against the shared C interface.
add_executable(c_consumer main.c)
target_link_libraries(c_consumer PRIVATE rapidglm::rapidglm_c)
