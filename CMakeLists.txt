cmake_minimum_required (VERSION 2.8.11)

project (graphflow)

find_package(HPX REQUIRED)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(Eigen3 REQUIRED)

include_directories(${graphflow_SOURCE_DIR}/graphflow ${EIGEN3_INCLUDE_DIR})

add_compile_options(-std=c++14 -Wno-unused-parameter)

add_hpx_executable(
	main
	ESSENTIAL
	SOURCES src/main.cpp)

