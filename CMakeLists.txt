cmake_minimum_required(VERSION 3.20)
project(MicrojetModel CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find Heyoka
find_package(heyoka REQUIRED)

add_executable(microjet main.cpp acousticWave.cpp)
target_link_libraries(microjet PRIVATE heyoka::heyoka)
target_compile_options(microjet PRIVATE -O2)
