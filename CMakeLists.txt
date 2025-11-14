cmake_minimum_required(VERSION 3.30)
project(Tensor)

set(CMAKE_CXX_STANDARD 20)

add_executable(Tensor main.cpp Tensor.cpp Arena.cpp)
