#include "NaiveKernel.h"

template<typename T>
__global__ void LoraKernel_A(const int         width,    
                            const T*               fused_lora, 
                            const T*               input,
                            T*                     result,
                            int*                   ranks,  // 0, 16, 144, 176
                            int*                   tokens, // 0, 3, 6, 8
                            const int              array_size)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;  // thread: 8, 22
  int col = blockIdx.x * blockDim.x + threadIdx.x;  // block: 1, 8

  T sum = 0;
  if (row < ranks[array_size-1] && col < tokens[array_size-1]) { //确保不超出矩阵C的大小
    // figure out the bound
    int bound = 1;
    #pragma unroll
    for(; bound < array_size; bound++) {
      if (row < ranks[bound]) break;
    }

    int row_bound_r = ranks[bound-1];
    int row_bound_l = ranks[bound];

    int col_bound_r = tokens[bound-1];
    int col_bound_l = tokens[bound];
    
    // kernel section
    if(row < row_bound_l && row >= row_bound_r && col < col_bound_l && col >= col_bound_r){
      #pragma unroll
      for (int k = 0; k < width; k++) {
        sum += fused_lora[row * width + k] * input[col * width + k];
      }

      // figure out the index
      int m_idx = 0;
      #pragma unroll
      for (int i = 1; i < bound; i++) {
        m_idx += (ranks[i] - ranks[i-1]) * (tokens[i] - tokens[i-1]);
      }
      m_idx += (row - row_bound_r) * (col_bound_l - col_bound_r) + (col - col_bound_r);
      result[m_idx] = sum;
    }
  }

}

template<typename T>
void launchNaiveKernel(const int         m,   // lora_rank_sum: 16 + 128 + 32
                            const int         n,   // token_num: default 8
                            const int         k,   // 4096
                            const T*               fused_lora, 
                            const T*               input,
                            T*                     result,
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

  dim3 blocksPerGrid(1, n);     // 分配8个blocks，因为C矩阵有8列
  int thread_y = m%n == 0 ? m/n : m/n + 2;
  dim3 threadsPerBlock(n, thread_y);  // 8 * 22 = 176

  int* d_seg;
  int* d_rank;
  cudaMalloc((void**)&d_seg, sizeof(int) * (N+1));
  cudaMalloc((void**)&d_rank, sizeof(int) * (N+1));
  cudaMemcpy(d_seg, segments, sizeof(int) * (N+1), cudaMemcpyHostToDevice);
  cudaMemcpy(d_rank, ranks, sizeof(int) * (N+1), cudaMemcpyHostToDevice);

  LoraKernel_A<<<blocksPerGrid, threadsPerBlock>>>(k, fused_lora, input, result, d_rank, d_seg, idx);

  cudaFree(d_seg);
  cudaFree(d_rank);
}

// instantiate
template void launchNaiveKernel<float>(const int         m,   // lora_rank_sum: 16 + 128 + 32
                                          const int         n,   // token_num: default 8
                                          const int         k,   // 4096
                                          const float*               fused_lora, 
                                          const float*               input,
                                          float*                     result,
                                          std::map<std::string, std::pair<int, int>> name_count_rank);

template void launchNaiveKernel<half>(const int         m,   // lora_rank_sum: 16 + 128 + 32
                                          const int         n,   // token_num: default 8
                                          const int         k,   // 4096
                                          const half*               fused_lora, 
                                          const half*               input,
                                          half*                     result,
                                          std::map<std::string, std::pair<int, int>> name_count_rank);

