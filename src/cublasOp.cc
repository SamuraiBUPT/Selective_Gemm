#include "cublasOp.h"

cublasHandle_t getCublasTensorOpHandle() {
    cublasHandle_t handle = nullptr;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    return handle;
}

void cublasTensorOp(cublasHandle_t handle, half *A, half *B, half *C, size_t M, size_t N, size_t K) {
  static half alpha = 1.0;
  static half beta = 0.0;

  cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16F, K, A,
                                        CUDA_R_16F, K, &beta, C, CUDA_R_16F, N, CUBLAS_COMPUTE_16F,
                                        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

void cublasBench(half* d_A, half* d_B, half* d_C, int M, int N, int K, std::map<std::string, std::pair<int, int>> lora_info) {
  int idx = 0;
  int m_start_idx = 0;
  int n_start_idx = 0;

  cudaStream_t stream1;
  cudaStreamCreate(&stream1);
  cudaStream_t stream2;
  cudaStreamCreate(&stream2);
  cudaStream_t stream3;
  cudaStreamCreate(&stream3);

  std::vector<cudaStream_t> repo;
  repo.push_back(stream1);
  repo.push_back(stream2);
  repo.push_back(stream3);
  
  int iii = 0;

  auto handle = getCublasTensorOpHandle();
  
  for (auto& e: lora_info) {
    int m = e.second.second;  // lora rank
    int n = e.second.first; // request
    int k = K;
    std::cout << "m: " << m << " n: " << n << " k: " << k << std::endl;

    cublasSetStream(handle, repo[iii]);
    cublasTensorOp(handle, d_A + m_start_idx, d_B + n_start_idx, d_C + idx, m, n, k);
    idx += m * n;
    m_start_idx += m * k;
    n_start_idx += n * k;
    iii++;
  }

  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
  cudaStreamDestroy(stream3);
}