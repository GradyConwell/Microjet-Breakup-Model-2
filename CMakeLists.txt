cmake_minimum_required(VERSION 3.20)
project(microjet LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(heyoka REQUIRED)

add_executable(microjet
    main.cpp
    acousticForcing.cpp
)

target_link_libraries(microjet PRIVATE heyoka::heyoka)

if(MSVC)
    target_compile_definitions(microjet PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(microjet PRIVATE /O2 /W3)
else()
    target_compile_options(microjet PRIVATE -O2 -Wall -Wextra -Wno-pragma-clang-attribute)
endif()
