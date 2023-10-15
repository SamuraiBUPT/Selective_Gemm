cmake_minimum_required(VERSION 3.10)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

project(MultiLoraKernel)

find_package(CUDA REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CUDA_INCLUDE_DIRS})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-G;-gencode arch=compute_86,code=sm_86)

cuda_add_executable(main 
                    src/main.cc 
                    src/NaiveKernel.cu
                    src/cublasOp.cc
                    src/mbmmKernel.cu)


target_link_libraries(main ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY})
