cmake_minimum_required(VERSION 3.5)
project(Feature_Matching)
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})


add_library(lib src/Sift.cpp src/visualizer.cpp)

add_executable(${PROJECT_NAME} main.cpp )
target_link_libraries(${PROJECT_NAME} lib ${OpenCV_LIBS} ${EiGEN3_LIBS})