cmake_minimum_required(VERSION 3.5)
project(parallel)

set(CMAKE_CXX_STANDARD 14)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

add_executable(parallelMPI MPI/main.cpp)
add_executable(parallelOMP OMP/main.cpp)
add_executable(serial SERIAL/main.cpp)

target_link_libraries(parallelOMP OpenMP::OpenMP_CXX)
target_link_libraries(parallelMPI MPI::MPI_CXX)
target_link_libraries(serial OpenMP::OpenMP_CXX)

target_link_libraries(parallelOMP ${OpenCV_LIBS})
target_link_libraries(parallelMPI ${OpenCV_LIBS})
target_link_libraries(serial ${OpenCV_LIBS})
