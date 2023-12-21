cmake_minimum_required(VERSION 3.22)
project(collective_api LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(Arrow REQUIRED)

add_executable(collective_api main.cpp)
target_link_libraries(collective_api PRIVATE Arrow::arrow_shared)


