# Copyright (c) 2018, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

cmake_minimum_required(VERSION 3.0)
project(NN_from_Scratch
  VERSION 3.5.0
  LANGUAGES CXX)
set(ArrayFire_DIR /opt/arrayfire/share/ArrayFire/cmake)
find_package(ArrayFire)

if(ArrayFire_CPU_FOUND)
  # Vectorization examples
  add_executable(try_cpu try.cpp)
  target_link_libraries(try_cpu ArrayFire::afcpu)
endif()

if(ArrayFire_CUDA_FOUND)
  add_executable(try_cuda try.cpp)
  target_link_libraries(try_cuda ArrayFire::afcuda)
endif()

if(ArrayFire_OpenCL_FOUND)
  add_executable(try_opencl try.cpp)
  target_link_libraries(try_opencl ArrayFire::afopencl)
endif()
