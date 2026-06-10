cmake_minimum_required(VERSION 3.10)
project(MyHNSWProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

SET( CMAKE_CXX_FLAGS  "-O3 -lpthread -march=native -lrt -DHAVE_CXX0X -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )

add_executable(build buildFvecs.cpp)
add_executable(query queryFvecs.cpp)