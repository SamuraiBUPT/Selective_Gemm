#include "common.h"

#pragma once

cublasHandle_t getCublasTensorOpHandle();

void cublasTensorOp(cublasHandle_t handle, half *A, half *B, half *C, size_t M, size_t N, size_t K);

void cublasBench(half* d_A, half* d_B, half* d_C, int M, int N, int K, std::map<std::string, std::pair<int, int>> lora_info);