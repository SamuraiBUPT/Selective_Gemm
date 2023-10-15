#include "NaiveKernel.h"
#include "cublasOp.h"
#include "MBMMKernel.h"

bool findNum(int target, int arr[], int size){
  for (int i = 0; i < size; i++) {
    if (arr[i] == target) return true;
  }
  return false;
}

// global variables
std::map<std::string, std::pair<int, int>> lora_info;

half* h_input;
half* h_input2;
half* h_output;

half* d_input;
half* d_input2;
half* d_output;

half* d_input_cublas;
half* d_input2_cublas;
half* d_output_cublas;

int main() {
    int token_num = 0;    // change here to modify B matrix
    int units = 4096;

    // change here to modify A matrix

    std::pair<int, int> lora_1 = {3, 16};
    std::pair<int, int> lora_2 = {3, 16};
    std::pair<int, int> lora_3 = {2, 16};

    // std::pair<int, int> lora_4 = {2, 16};
    // std::pair<int, int> lora_5 = {2, 16};
    // std::pair<int, int> lora_6 = {2, 16};


    lora_info.insert({"1", lora_1});
    lora_info.insert({"2", lora_2});
    lora_info.insert({"3", lora_3});

    // lora_info.insert({"4", lora_4});
    // lora_info.insert({"5", lora_5});
    // lora_info.insert({"6", lora_6});
    // lora_info.insert({"7", lora_6});

    // for(int i = 0; i<43; i++) {
    //   lora_info.insert({std::to_string(i+8), {1, 16}});
    // }    

    int lora_ranks = 0;
    const int N = lora_info.size();
    int segs_prefix[N] = {0};
    int idxx = 1;
    for(auto& e : lora_info) {
      lora_ranks += e.second.second;
      token_num += e.second.first;
      segs_prefix[idxx] = e.second.second * e.second.first + segs_prefix[idxx-1];
      idxx++;
    }

    int size_lora = units * lora_ranks; // A matrix: lora concat
    int size2 = units * token_num;      // B matrix: input concat

    int size3 = 0;                      // C matrix: output concat
    for (auto& e : lora_info) {
      size3 += e.second.first * e.second.second;
    }


    h_input = new half[size_lora];
    h_input2 = new half[size2];
    h_output = new half[size3];

    // 从dyz_q_content.txt中读取数据
    std::ifstream infile;
    int i = 0;
    for (int ii = 0; ii < lora_ranks / 16; ii++){
      infile.open("../data/dyz_q_content.txt");
      if (!infile) {
          std::cout << "open file error" << std::endl;
          return 0;
      }
      while (!infile.eof()) {
          float tmp;
          infile >> tmp;
          h_input[i] = __float2half(tmp);
          i++;
      }
      infile.close();
    }

    // 从dyz_input_content.txt中读取数据
    std::ifstream infile2;
    int j = 0;
    for (int ii = 0; ii < token_num / 2; ii++) {
      infile2.open("../data/dyz_input_content.txt");
      if (!infile2) {
          std::cout << "open file error" << std::endl;
          return 0;
      }
      while (!infile2.eof()) {
        float tmp;
        infile2 >> tmp;
        h_input2[j] = __float2half(tmp);
        j++;
      }
      infile2.close();
    }


    // 分配设备内存
    cudaMalloc((void**)&d_input, size_lora * sizeof(half));
    cudaMalloc((void**)&d_input2, size2 * sizeof(half));
    cudaMalloc((void**)&d_output, size3 * sizeof(half));

    cudaMemcpy(d_input, h_input, size_lora * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2, h_input2, size2 * sizeof(half), cudaMemcpyHostToDevice);

    // 1. Naive Kernel
    for (int z = 0; z < 5; z++){
      launchNaiveKernel<half>(lora_ranks, token_num, units, d_input, d_input2, d_output, lora_info);
    }

    // 2. Block masked MBMM kernel
    for (int z = 0; z < 5; z++){
      launchMBMMKernel<half>(lora_ranks, token_num, units, d_input, d_input2, d_output, lora_info);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    // param: dest, src
    cudaMemcpy(h_output, d_output, size3 * sizeof(half), cudaMemcpyDeviceToHost);

    // 打印结果，以检查kernel的正确性
    // for (int i = 0; i < size3; i++) {
    //     if (findNum(i, segs_prefix, idxx - 1)) {
    //       std::cout<< "\n idx: "<< i << std::endl;
    //     }
    //     std::cout << h_output[i] << " ";
    // }
    std::cout << std::endl<<"I'm done." <<std::endl;
    std::cout << std::endl;

    cudaFree(d_input);
    cudaFree(d_input2);
    cudaFree(d_output);


    // metric cublas kernel

    half* h_output_cublas = new half[size3];

    cudaMalloc((void**)&d_input_cublas, size_lora * sizeof(half));
    cudaMalloc((void**)&d_input2_cublas, size2 * sizeof(half));
    cudaMalloc((void**)&d_output_cublas, size3 * sizeof(half));

    cudaMemcpy(d_input_cublas, h_input, size_lora * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2_cublas, h_input2, size2 * sizeof(half), cudaMemcpyHostToDevice);


    for (int z = 0; z < 5; z++) {
      cublasBench(d_input_cublas, d_input2_cublas, d_output_cublas, lora_ranks, token_num, units, lora_info);
    }

    cudaMemcpy(h_output_cublas, d_output_cublas, size3 * sizeof(half), cudaMemcpyDeviceToHost);
    // 打印结果，以检查kernel的正确性
    // for (int i = 0; i < size3; i++) {
    //     if (findNum(i, segs_prefix, idxx - 1)) {
    //       std::cout<< "\n idx: "<< i << std::endl;
    //     }
    //     std::cout << h_output_cublas[i] << " ";
    // }


    // 清理资源
    delete[] h_input;
    std::cout <<"I'm done." <<std::endl;
    delete[] h_input2;
    std::cout <<"I'm done." <<std::endl;
    delete[] h_output;
    std::cout <<"I'm done." <<std::endl;

    delete[] h_output_cublas;

    cudaFree(d_input_cublas);
    cudaFree(d_input2_cublas);
    cudaFree(d_output_cublas);


    return 0;
}
