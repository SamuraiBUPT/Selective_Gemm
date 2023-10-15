#include "mbmmKernel.h"

template <typename T>
__global__ void MBMMkernel(const int          m,
                    const int          n,
                    const int          k,
                    const T*             fused_lora, 
                    const T*             input,
                    T*                   result,
                    const int*           segments,
                    const int*           ranks)
{
  int block_id = blockIdx.x * blockDim.x + blockIdx.y;
  if(block_id != 0 || block_id != 4 || block_id != 8) return;

  // we have 16 * 4 per block
  int BLOCK_ID = block_id / 4;
  int row = BLOCK_ID * 16 + threadIdx.x;
  int col = BLOCK_ID * 3 + threadIdx.y; // 0-4 3-7 6-10

  // 边界检查
  T sum = 0;
  if(row >= BLOCK_ID * 16 && row < BLOCK_ID * 16 + 16 && col >= BLOCK_ID * 3 && col < BLOCK_ID == 2 ? BLOCK_ID * 3 + 2 : BLOCK_ID * 3 + 3) {
    #pragma unroll
    for (int i = 0; i < k; i++) {
      sum += fused_lora[row * k + i] * input[col * k + i];
    }

    // figure out the idx of result
    int idx = BLOCK_ID * 16 * 3 + (row - BLOCK_ID * 16) * 3 + col - BLOCK_ID * 3;
    result[idx] = sum;
  }
} 

template <typename T>
void launchMBMMKernel(const int          m,
                      const int          n,
                      const int          k,
                      const T*             fused_lora, 
                      const T*             input,
                      T*                   result,
                      std::map<std::string, std::pair<int, int>> name_count_rank)
{
  const int N = name_count_rank.size();
  int segments[N+1] = {0};       // get the requests account and types
  int ranks[N+1] = {0};          // get the lora ranks

  // 前缀和算法 
  // 比如lora rank分别是：16 128 32，那么这些切割点就是：0, 16, 144, 176
  int idx = 1;
  for(auto& item : name_count_rank) {
    segments[idx] = item.second.first + segments[idx-1];  // 0, 3, 6, 8, 10
    ranks[idx] = item.second.second + ranks[idx-1];       // 0, 16, 144, 176, 208
    idx++;
  }

  dim3 ThreadPerBlock(16, 4);
  dim3 BlockPerGrid(3, 3);

  int* d_seg;
  int* d_rank;
  cudaMalloc((void**)&d_seg, sizeof(int) * (N+1));
  cudaMalloc((void**)&d_rank, sizeof(int) * (N+1));
  cudaMemcpy(d_seg, segments, sizeof(int) * (N+1), cudaMemcpyHostToDevice);
  cudaMemcpy(d_rank, ranks, sizeof(int) * (N+1), cudaMemcpyHostToDevice);

  MBMMkernel<<<BlockPerGrid, ThreadPerBlock>>>(m, n, k, fused_lora, input, result, d_seg, d_rank);


  cudaFree(d_seg);
  cudaFree(d_rank);
}

template void launchMBMMKernel<float>(const int          m,
                                      const int          n,
                                      const int          k,
                                      const float*             fused_lora, 
                                      const float*             input,
                                      float*                   result,
                                      std::map<std::string, std::pair<int, int>> name_count_rank);

template void launchMBMMKernel<half>(const int          m,
                                       const int          n,
                                       const int          k,
                                       const half*             fused_lora, 
                                       const half*             input,
                                       half*                   result,
                                       std::map<std::string, std::pair<int, int>> name_count_rank);