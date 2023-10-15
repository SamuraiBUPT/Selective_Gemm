#include "common.h"

template<typename T>
void launchMBMMKernel(const int          m,
                      const int          n,
                      const int          k,
                      const T*             fused_lora, 
                      const T*             input,
                      T*                   result,
                      std::map<std::string, std::pair<int, int>> name_count_rank);