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
void myGEMM_kernel(double* A, double* B, double* C, 
				   double alpha, double beta, int M, int N, int K) {
	// side is BLOCK_SIZE
	// M is C.stride
	// K is B.stride
	// M is A.stride
	const int block_row = blockIdx.y;
	const int block_col = blockIdx.x;

	const int row = threadIdx.y;
	const int col = threadIdx.x;

	double Cval = 0;
	
	const int C_idx = M * side * block_col + side * block_row;
	// // (probably don't need)
	// if (C_idx >= M * N) {
	// 	printf("uh oh");
	// 	return;
	// }

    // get pointer into sub matrix for this kernel
	double* Csub = &(C[C_idx]);

	const int B_size = K * N;
	const int A_size = M * K;
	const int Asub_idx = M * col + row;
	const int Bsub_idx = K * col + row;

	// check bounds
	if (Asub_idx + C_idx >= M * N) {
		return;
	}

	// printf("\n\n block_row: %d \n block_col: %d \n row: %d \n col: %d \n Asub_idx: %d \n Bsub_idx %d \n",
	//        block_row, block_col, row, col, Asub_idx, Bsub_idx);

	// check in bounds
	// if (Asub_idx >= A_size || Bsub_idx >= B_size) {
	// 	return;	
	// }

	// loop over sub matrices (K is width of A)
	for (int k = 0; k < ((K + side - 1) / side); ++k) {

		//  CHECK IN BOUNDS (probably don't need)
		int B_idx = K * side * block_col + side * k;
		int A_idx = M * side * k + side * block_row;
		// if (B_idx + >= B_size || A_idx >= A_size) {
		// 	printf("uh oh 2");
		// 	return;
		// }

		// address to location of sub
		double* Asub = &(A[A_idx]);
		double* Bsub = &(B[B_idx]);

		// allocate shared memory
		__shared__ double Ashared[side][side];
		__shared__ double Bshared[side][side];

		// assign elements to shared memory
		if (A_idx + Asub_idx < A_size) {
			Ashared[row][col] = Asub[Asub_idx];
		}
		else {
			Ashared[row][col] = 0;
		}
		if (B_idx + Bsub_idx < B_size) {
			Bshared[row][col] = Bsub[Bsub_idx];
		}
		else {
			Bshared[row][col] = 0;
		}

		__syncthreads();
		
		const int idx_check = side * k;

		// do matrix multiply
		for (int idx = 0; idx < side; ++idx) {
			// bounds check (do you still need this?)
			if (idx + idx_check < K) {
				Cval += Ashared[row][idx] * Bshared[idx][col];
			}
			// Cval += Ashared[row][idx] * Bshared[idx][col];
		}

		__syncthreads();


		// printf("\n\n loop \n block_row: %d \n block_col: %d \n row: %d \n col: %d \n A_idx: %d \n B_idx: %d \n k%d \n idx_check: %d \n",
	 //           block_row, block_col, row, col, A_idx, B_idx, k, idx_check);
	}
	// evaluate the rest of the GEMM equation
	Cval = alpha * Cval + beta * Csub[Asub_idx];
	// set value
	Csub[Asub_idx] = Cval;
}

/* 
Routine to perform an in-place GEMM operation, i.e., C := alpha*A*B + beta*C 
*/
int myGEMM(double* A, double* B, double* C, double* alpha, double* beta, int M, int N, int K){
	/* TODO: Write an efficient GEMM implementation on GPU */

	// A, B, C are already memcopied to device ie we already have device pointers
	// first set up threads_per_block and blocks_per_grid
	const int side = 16;

	const int n_threads_per_block = 256;
	int threads_x = side;
	int threads_y = side;

	int block_x = (N + threads_x - 1) / threads_x;
	int block_y = (M + threads_y - 1) / threads_y;

	dim3 threads_per_block(threads_x, threads_y);
	dim3 blocks_per_grid(block_x, block_y);

	// set up streams ??
	myGEMM_kernel <side> <<<blocks_per_grid, threads_per_block>>> 
		(A, B, C, *alpha, *beta, M, N, K);

	check_launch("myGEMM_kernel");

	return 0;
}

// X and y have been subdivided
int gpu_train(double* X, double* y, double* W0, double* W1, double* b0, double* b1, 
			  const int n_images, const int n_0, const int n_1, const int n_2) {

	// create pointers
	double* d_X;
	double* d_y;
	double* d_W0;
	double* d_b0;
	double* d_W1;
	double* d_b1;
	// data only on device
	double* d_a1;
	double* d_a2;
	double* d_z1;
	double* d_z2;

	// calc sizes
	const size_t size_d = sizeof(double);
	const size_t X_size = n_images * n_0 * size_d; // 800 x 784
	const size_t y_size = n_images * n_2 * size_d; // 800 x 10
	const size_t W0_size = n_1 * n_0 * size_d; // 100 x 784
	const size_t W1_size = n_2 * n_1 * size_d; // 10 x 100
	const size_t b0_size = n_1 * size_d; // 1 x 100
	const size_t b1_size = n_2 * size_d; // 1 x 10
	const size_t a1_size = n_images * n_1 * size_d; // 800 x 100
	const size_t a2_size = n_images * n_2 * size_d; // 800 x 10
	const size_t z1_size = n_images * n_1 * size_d; // 800 x 100
	const size_t z2_size = n_images * n_2 * size_d; // 800 x 10

	// malloc
	checkCudaErrors(cudaMalloc(&d_X, X_size));
	checkCudaErrors(cudaMalloc(&d_y, y_size));
	checkCudaErrors(cudaMalloc(&d_W0, W0_size));
	checkCudaErrors(cudaMalloc(&d_W1, W1_size));
	checkCudaErrors(cudaMalloc(&d_b0, b0_size));
	checkCudaErrors(cudaMalloc(&d_b1, b1_size));
	checkCudaErrors(cudaMalloc(&d_a1, a1_size));
	checkCudaErrors(cudaMalloc(&d_a2, a2_size));
	checkCudaErrors(cudaMalloc(&d_z1, z1_size));
	checkCudaErrors(cudaMalloc(&d_z2, z2_size));

	// memcpy
	cudaMemcpy(d_X, X, X_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, y_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_W0, W0, W0_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_W1, W1, W1_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b0, b0, b0_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b1, b1, b1_size, cudaMemcpyHostToDevice);

	// feedforward steps to calc a1, a2, z1, z2 all on device

	// backprop steps to calc dW0/1 and db0/1 all on device
	// calls to myGEMM_kernel through myGEMM or directly?
	// calls to other __global__ functions?

	// free!
	cudaFree(d_X);
	cudaFree(d_y);
	cudaFree(d_W0);
	cudaFree(d_W1);
	cudaFree(d_b0);
	cudaFree(d_b1);
	cudaFree(d_a1);
	cudaFree(d_a2);
	cudaFree(d_z1);
	cudaFree(d_z2);

	return 1;
}
