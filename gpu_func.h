#ifndef GPU_FUNC_H_
#define GPU_FUNC_H_

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

struct event_pair
{
  cudaEvent_t start;
  cudaEvent_t end;
};

inline void check_launch(const char * kernel_name)
{
  cudaThreadSynchronize();
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess)
  {
    std::cerr << "error in " << kernel_name << " kernel" << std::endl;
    std::cerr << "error was: " << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
}

inline void start_timer(event_pair * p)
{
  cudaEventCreate(&p->start);
  cudaEventCreate(&p->end);
  cudaEventRecord(p->start, 0);
}


inline double stop_timer(event_pair * p)
{
  cudaEventRecord(p->end, 0);
  cudaEventSynchronize(p->end);
  
  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, p->start, p->end);
  cudaEventDestroy(p->start);
  cudaEventDestroy(p->end);
  return elapsed_time;
}

int useless_gpu_add_one (int t);

int myGEMM_no_overwrite_transposeB(double* A, double* B, double* C, double* D, 
            double alpha, double beta, int M, int N, int K);

int myGEMM_no_overwrite(double* A, double* B, double* C, double* D,
            double alpha, double beta, int M, int N, int K);

__global__
void myGEMM_kernel(double* A, double* B, double* C, 
           double alpha, double beta, int M, int N, int K);

int myGEMM(double* A, double* B, double* C, double* alpha, double* beta, int M, int N, int K);

int gpu_train(double* X, double* y, double* W0, double* W1, double* b0, double* b1, 
        double* DW0, double* DW1, double* Db0, double* Db1,
        const int n_images, const int n_0, const int n_1, const int n_2, double reg, double learning_rate);

#endif