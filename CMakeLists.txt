cmake_minimum_required(VERSION 2.25)

project(k-means_using_OpenMP LANGUAGES C)

set(CMAKE_C_STANDARD 90)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(k-means_using_OpenMP src/main.c src/help.h src/help.c src/kmeans.h src/kmeans.c)

target_link_libraries(k-means_using_OpenMP m)
