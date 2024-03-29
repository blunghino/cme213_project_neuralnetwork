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

int Dz1_schur_GPU(double* Da1, double* a1, double* Dz1, int M, int N);

int sigmoid_GPU(double* z1, double* a1, int M, int N);

int softmax_GPU(double* z2, double* a2, double* y, int M, int N, int scale);

int myGEMM_no_overwrite_no_add_transposeA(double* A, double* B, double* C, 
                      double alpha, int M, int N, int K);

int myGEMM_no_overwrite_transposeB(double* A, double* B, double* C, double* D, 
            double alpha, double beta, int M, int N, int K);

int myGEMM_no_overwrite(double* A, double* B, double* C, double* D,
            double alpha, double beta, int M, int N, int K);

int shared_myGEMM_no_overwrite_no_add_transposeA(double* A, double* B, double* C, 
                      double alpha, int M, int N, int K);

int shared_myGEMM_no_overwrite_transposeB(double* A, double* B, double* C, double* D, 
            double alpha, double beta, int M, int N, int K);

int shared_myGEMM_no_overwrite(double* A, double* B, double* C, double* D, 
            double alpha, double beta, int M, int N, int K);

int myGEMM_no_na_tA(double* A, double* B, double* C, 
                      double alpha, int M, int N, int K);

int myGEMM_no_tB(double* A, double* B, double* C, double* D, 
            double alpha, double beta, int M, int N, int K);

int myGEMM_no(double* A, double* B, double* C, double* D,
            double alpha, double beta, int M, int N, int K);

int myGEMM(double* A, double* B, double* C, double* alpha, double* beta, int M, int N, int K);

#endif