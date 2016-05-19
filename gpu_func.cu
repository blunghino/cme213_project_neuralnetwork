#include "gpu_func.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include "cublas_v2.h"

__global__
void device_add_one (int* d_result, int t)
{
	*d_result = t + 1;
}

/*
Just a dummy function the can be used to warm up GPU
*/
int useless_gpu_add_one (int t)
{
	int result;
	int *d_result;

	checkCudaErrors (cudaMalloc((void **)&d_result, 1 * sizeof (int)));

	event_pair timer;
	start_timer (&timer);
	device_add_one<<<1,1>>>(d_result, t);
	check_launch ("device_add_one");
	double time = stop_timer (&timer);

	std::cout << "device_add_one took: " << time << " seconds" << std::endl;

	checkCudaErrors (cudaMemcpy(&result, d_result, 1 * sizeof (int), cudaMemcpyDeviceToHost));
	return result;
}

/*
Kernel function called by my GEMM
*/
template <int side>
__global__
void myGEMM_kernel(double* A, double* B, double* C, const double alpha, const double beta, 
				   const int M, const int N, const int K) {
	// side is BLOCK_SIZE
	// N is C.stride
	// N is B.stride
	// K is A.stride

	const int block_row = blockDim.y;
	const int block_col = blockDim.x;

	const int row = threadIdx.y;
	const int col = threadIdx.x;

	double Cval = 0;

    // get pointer into sub matrix for this kernel
	double* Csub = &((*C)[N * side * block_row + side * block_col]);

	for (int k = 0; k < (K / side); ++k) {

		// index into sub
		double* Asub = &((*A)[K * side * block_row + side * k]);
		double* Bsub = B[N * side * k + side * block_col];

		// allocate shared memory
		__shared__ double Ashared[side][side];
		__shared__ double Bshared[side][side];

		Ashared[row][col] = (*Asub)[K * row + col]
		Bshared[row][col] = (*Bsub)[N * row + col]
	}

}

/* 
Routine to perform an in-place GEMM operation, i.e., C := alpha*A*B + beta*C 
*/
int myGEMM(double* A, double* B, double* C, double* alpha, double* beta, int M, int N, int K){
	/* TODO: Write an efficient GEMM implementation on GPU */

	// A, B, C are already memcopied to device ie we already have device pointers
	// first set up threads_per_block and blocks_per_grid
	int side = 32;
	int block_x = (N + side) / side;
	int block_y = (M + side) / side;
	dim3 threads_per_block(side, side);
	dim3 blocks_per_grid(block_x, block_y);

	// set up streams ?
	myGEMM_kernel <side> <<<blocks_per_grid, threads_per_block>>> 
		(A, B, C, *alpha, *beta, M, N, K);



	return 1;
}
