cmake_minimum_required(VERSION 3.5)
project(openMPITest)

set(CMAKE_CXX_STANDARD 14)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(openMPITest main.cpp)

target_link_libraries(openMPITest OpenMP::OpenMP_CXX)
target_link_libraries(openMPITest MPI::MPI_CXX)

target_link_libraries(openMPITest ${OpenCV_LIBS})
