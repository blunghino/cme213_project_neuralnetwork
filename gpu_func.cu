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

// Dz1.T = Da1.T .* a1.T .* (1 - a1.T)
__global__
void Dz1_schur_kernel(double* Da1, double* a1, double* Dz1, int M, int N) {

    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int k = col * M + row;

    if (k < M * N) {
    	double a1_k = a1[k];
    	Dz1[k] = Da1[k] * a1_k * (1 - a1_k);
    }
}

// Dz1.T = Da1.T .* a1.T .* (1 - a1.T)
int Dz1_schur_GPU(double* Da1, double* a1, double* Dz1, int M, int N) {

    int threads_per_block = 256;
    int threads_x = 32;
    int threads_y = threads_per_block / threads_x;

    int blocks_x = (N + threads_x - 1) / threads_x;
    int blocks_y = (M + threads_y - 1) / threads_y;

    dim3 threads(threads_x, threads_y);
    dim3 blocks(blocks_x, blocks_y);

	Dz1_schur_kernel<<<blocks, threads>>> (Da1, a1, Dz1, M, N);

	check_launch("Dz1_schur_kernel");

	return 0;
}

// W0 = W0 + learning_rate * DW0 
__global__
void in_place_linear_combination_kernel(double* W0, double* DW0, 
										double learning_rate, int M, int N) {

    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int k = col * M + row;

    if (k < M * N) {
    	W0[k] += learning_rate * DW0[k];
    }
}

// W0 = W0 + learning_rate * DW0 
int in_place_linear_combination_GPU(double* W0, double* DW0, 
									double learning_rate, int M, int N) {

    int threads_per_block = 256;
    int threads_x = 32;
    int threads_y = threads_per_block / threads_x;

    int blocks_x = (N + threads_x - 1) / threads_x;
    int blocks_y = (M + threads_y - 1) / threads_y;

    dim3 threads(threads_x, threads_y);
    dim3 blocks(blocks_x, blocks_y);

	in_place_linear_combination_kernel<<<blocks, threads>>> (W0, DW0, learning_rate, M, N);

	check_launch("in_place_linear_combination_kernel");

	return 0;
}

__global__
void sigmoid_kernel(double* z1, double* a1, int M, int N) {

    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int k = col * M + row;

    if (k < M * N) {
    	a1[k] = 1.0 / (1.0 + exp(-z1[k]));
    }
}

// 
int sigmoid_GPU(double* z1, double* a1, int M, int N) {

    int threads_per_block = 256;
    int threads_x = 32;
    int threads_y = threads_per_block / threads_x;

    int blocks_x = (N + threads_x - 1) / threads_x;
    int blocks_y = (M + threads_y - 1) / threads_y;

    dim3 threads(threads_x, threads_y);
    dim3 blocks(blocks_x, blocks_y);

	sigmoid_kernel<<<blocks, threads>>> (z1, a1, M, N);

	check_launch("sigmoid_kernel");

	return 0;
}

// this kernel does soft max and subtracts y AND scales by 1/N
__global__
void softmax_kernel(double* z2, double* a2, double* y, int M, int N, int scale) {

	int col = threadIdx.x;
    int row = blockIdx.x;
    int k = col * M + row;

    a2[k] = exp(z2[k]);

    __syncthreads();

    double denom = 0;
    int idx;

    for (int i = 0; i < M; ++i) {
    	idx = col * M + i;
    	denom += a2[idx];
    }

    __syncthreads();

    a2[k] /= denom;

    // (y^ - y)
    a2[k] -= y[k];
    // factor of 1/N
    a2[k] /= (double)scale;

}

int softmax_GPU(double* z2, double* a2, double* y, int M, int N, int scale) {

    softmax_kernel<<<M, N>>> (z2, a2, y, M, N, scale);

    check_launch("softmax_kernel");

    return 0;
}

// Kernel function called by my GEMM no overwrite - alpha * A * B = C
template <int side>
__global__
void myGEMM_no_overwrite_no_add_transposeA_kernel(double* A, double* B, double* C,
				                				  double alpha, int M, int N, int K) {
	// side is BLOCK_SIZE
	// M is C.stride
	// K is B.stride
	// K is A.stride
	// A = K x M, B = K x N, C = M x N
	const int block_row = blockIdx.y;
	const int block_col = blockIdx.x;

	const int row = threadIdx.y;
	const int col = threadIdx.x;

	double Cval = 0;
	
	const int C_idx = M * side * block_col + side * block_row;

	const int Asub_idx = K * row + col;
	const int Csub_idx = M * col + row;
	const int Bsub_idx = K * col + row;

    // get pointer into sub matrix for this kernel
	double* Csub = &(C[C_idx]);

	const int B_size = K * N;
	const int A_size = M * K;

	// loop over sub matrices (K is width of A)
	for (int k = 0; k < ((K + side - 1) / side); ++k) {

		//  to CHECK IN BOUNDS 
		int B_idx = K * side * block_col + side * k;
		int A_idx = K * side * block_row + side * k;

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
		
		// do matrix multiply
		for (int idx = 0; idx < side; ++idx) {
			Cval += Ashared[row][idx] * Bshared[idx][col];
		}

		__syncthreads();

	}
	
	// check bounds
	if (Csub_idx + C_idx < M * N) {
		// set value
		Csub[Csub_idx] = Cval;
	}

}


// Routine to perform a GEMM operation without addition, transposing A
//  not in place, i.e., D := alpha*A.T*B
int myGEMM_no_overwrite_no_add_transposeA(double* A, double* B, double* C, 
										  double alpha, int M, int N, int K){

	// A, B are already memcopied to device ie we already have device pointers
	// D is already malloced
	const int side = 16;
	//const int n_threads_per_block = 256;
	int threads_x = side;
	int threads_y = side;

	int block_x = (N + threads_x - 1) / threads_x;
	int block_y = (M + threads_y - 1) / threads_y;

	dim3 threads_per_block(threads_x, threads_y);
	dim3 blocks_per_grid(block_x, block_y);

	// set up streams ??
	myGEMM_no_overwrite_no_add_transposeA_kernel <side> <<<blocks_per_grid, threads_per_block>>> 
		(A, B, C, alpha, M, N, K);

	check_launch("myGEMM_no_overwrite_no_add_transposeA");

	return 0;
}

// Kernel function called by my GEMM no overwrite
// TRANSPOSING B to acheive matrix multiply dimensions
template <int side>
__global__
void myGEMM_no_overwrite_transposeB_kernel(double* A, double* B, double* C, double* D,
				                double alpha, double beta, int M, int N, int K) {
	// side is BLOCK_SIZE
	// M is C.stride
	// N is B.stride
	// M is A.stride
	// A = M x K, B = N x K, C = M x N, D = M x N
	const int block_row = blockIdx.y;
	const int block_col = blockIdx.x;

	const int row = threadIdx.y;
	const int col = threadIdx.x;

	double Cval = 0;
	
	const int C_idx = M * side * block_col + side * block_row;

	const int Asub_idx = M * col + row;
	const int Bsub_idx = N * row + col;

    // get pointer into sub matrix for this kernel
	double* Csub = &(C[C_idx]);
	double* Dsub = &(D[C_idx]);

	const int B_size = K * N;
	const int A_size = M * K;

	// loop over sub matrices (K is width of A)
	for (int k = 0; k < ((K + side - 1) / side); ++k) {

		//  to CHECK IN BOUNDS 
		int B_idx = N * side * k + side * block_col;
		int A_idx = M * side * k + side * block_row;

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
		
		// do matrix multiply
		for (int idx = 0; idx < side; ++idx) {
			Cval += Ashared[row][idx] * Bshared[idx][col];
		}

		__syncthreads();

	}
	
	// check bounds
	if (Asub_idx + C_idx < M * N) {
		Dsub[Asub_idx] = alpha * Cval + beta * Csub[Asub_idx];
	}
}


// Routine to perform a GEMM operation TRANSPOSING B,
// A = M x K, B = N x K, C = M x N, D = M x N
// not in place, D := alpha * A * B.T + beta*C 
int myGEMM_no_overwrite_transposeB(double* A, double* B, double* C, double* D, 
						double alpha, double beta, int M, int N, int K){

	// A, B, C are already memcopied to device ie we already have device pointers
	// D is already malloced
	const int side = 16;
	//const int n_threads_per_block = 256;
	int threads_x = side;
	int threads_y = side;

	int block_x = (N + threads_x - 1) / threads_x;
	int block_y = (M + threads_y - 1) / threads_y;

	dim3 threads_per_block(threads_x, threads_y);
	dim3 blocks_per_grid(block_x, block_y);

	// set up streams ??
	myGEMM_no_overwrite_transposeB_kernel <side> <<<blocks_per_grid, threads_per_block>>> 
		(A, B, C, D, alpha, beta, M, N, K);

	check_launch("myGEMM_no_overwrite_transposeB_kernel");

	return 0;
}

// Kernel function called by my GEMM no overwrite
template <int side>
__global__
void myGEMM_no_overwrite_kernel(double* A, double* B, double* C, double* D,
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

	const int Asub_idx = M * col + row;
	const int Bsub_idx = K * col + row;

    // get pointer into sub matrix for this kernel
	double* Csub = &(C[C_idx]);
	double* Dsub = &(D[C_idx]);

	const int B_size = K * N;
	const int A_size = M * K;

	// loop over sub matrices (K is width of A)
	for (int k = 0; k < ((K + side - 1) / side); ++k) {

		//  to CHECK IN BOUNDS 
		int B_idx = K * side * block_col + side * k;
		int A_idx = M * side * k + side * block_row;

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
		
		// do matrix multiply
		for (int idx = 0; idx < side; ++idx) {
			Cval += Ashared[row][idx] * Bshared[idx][col];
		}

		__syncthreads();

	}
	
	// check bounds
	if (Asub_idx + C_idx < M * N) {
		Cval = alpha * Cval + beta * Csub[Asub_idx];
		// set value
		Dsub[Asub_idx] = Cval;	}
}


// Routine to perform a GEMM operation, not in place, i.e., D := alpha*A*B + beta*C 
int myGEMM_no_overwrite(double* A, double* B, double* C, double* D, 
						double alpha, double beta, int M, int N, int K){

	// A, B, C are already memcopied to device ie we already have device pointers
	// D is already malloced
	const int side = 16;
	//const int n_threads_per_block = 256;
	int threads_x = side;
	int threads_y = side;

	int block_x = (N + threads_x - 1) / threads_x;
	int block_y = (M + threads_y - 1) / threads_y;

	dim3 threads_per_block(threads_x, threads_y);
	dim3 blocks_per_grid(block_x, block_y);

	// set up streams ??
	myGEMM_no_overwrite_kernel <side> <<<blocks_per_grid, threads_per_block>>> 
		(A, B, C, D, alpha, beta, M, N, K);

	check_launch("myGEMM_no_overwrite_kernel");

	return 0;
}

/*
Kernel function called by my GEMM
*/
template <int side>
__global__
void myGEMM_shared_kernel(double* A, double* B, double* C, 
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

	const int Asub_idx = M * col + row;
	const int Bsub_idx = K * col + row;

    // get pointer into sub matrix for this kernel
	double* Csub = &(C[C_idx]);

	const int B_size = K * N;
	const int A_size = M * K;

	// loop over sub matrices (K is width of A)
	for (int k = 0; k < ((K + side - 1) / side); ++k) {

		//  to CHECK IN BOUNDS 
		int B_idx = K * side * block_col + side * k;
		int A_idx = M * side * k + side * block_row;

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
		
		// const int idx_check = side * k;

		// do matrix multiply
		for (int idx = 0; idx < side; ++idx) {
			// bounds check (do you still need this?)
			// if (idx + idx_check < K) {
			// 	Cval += Ashared[row][idx] * Bshared[idx][col];
			// }
			Cval += Ashared[row][idx] * Bshared[idx][col];
		}

		__syncthreads();


	// printf("\n\n loop \n block_row: %d \n block_col: %d \n row: %d \n col: %d \n A_idx: %d \n B_idx: %d \n k%d \n idx_check: %d \n",
	//           block_row, block_col, row, col, A_idx, B_idx, k, idx_check);
	}
	
	// check bounds
	if (Asub_idx + C_idx < M * N) {
		// evaluate the rest of the GEMM equation
		Cval = alpha * Cval + beta * Csub[Asub_idx];
		// set value
		Csub[Asub_idx] = Cval;
	}
}

/* 
Routine to perform an in-place GEMM operation, i.e., C := alpha*A*B + beta*C 
*/
int myGEMM_shared(double* A, double* B, double* C, double* alpha, double* beta, int M, int N, int K){
	/* TODO: Write an efficient GEMM implementation on GPU */

	// A, B, C are already memcopied to device ie we already have device pointers
	// first set up threads_per_block and blocks_per_grid
	const int side = 16;
	//const int n_threads_per_block = 256;
	int threads_x = side;
	int threads_y = side;

	int block_x = (N + threads_x - 1) / threads_x;
	int block_y = (M + threads_y - 1) / threads_y;

	dim3 threads_per_block(threads_x, threads_y);
	dim3 blocks_per_grid(block_x, block_y);

	// set up streams ??
	myGEMM_shared_kernel <side> <<<blocks_per_grid, threads_per_block>>> 
		(A, B, C, *alpha, *beta, M, N, K);

	check_launch("myGEMM_shared_kernel");

	return 0;
}

__global__
void myGEMM_no_na_tA_kernel(double* A, double* B, double* C, double alpha, int M, int N, int K) {

	int row = blockIdx.x;
	int col = threadIdx.x;
	int idx = col * M + row;

	double Cval = 0;

	for (int k = 0; k < K; ++k) {
		Cval += alpha * A[K * row + k] * B[K * col + k];
	}

	C[idx] = Cval;
}

int myGEMM_no_na_tA(double* A, double* B, double* C, 
                      double alpha, int M, int N, int K) {

	myGEMM_no_na_tA_kernel <<<M, N>>> (A, B, C, alpha, M, N, K);

	check_launch("myGEMM_no_tb_kernel");
	
	return 0;	
}

__global__
void myGEMM_no_tB_kernel(double* A, double* B, double* C, double* D,
                         double alpha, double beta, int M, int N, int K) {

	int row = blockIdx.x;
	int col = threadIdx.x;
	int idx = col * M + row;

	double Cval = 0;

	for (int k = 0; k < K; ++k) {
		Cval += alpha * A[k * M + row] * B[N * k + col];
	}

	D[idx] = Cval + C[idx] * beta;
}

int myGEMM_no_tB(double* A, double* B, double* C, double* D, 
            double alpha, double beta, int M, int N, int K) {

	myGEMM_no_tB_kernel <<<M, N>>> (A, B, C, D, alpha, beta, M, N, K);

	check_launch("myGEMM_no_tb_kernel");
	
	return 0;
}

__global__
void myGEMM_no_kernel(double* A, double* B, double* C, double* D, 
	                  double alpha, double beta, int M, int N, int K) {

	int row = blockIdx.x;
	int col = threadIdx.x;
	int idx = col * M + row;

	double Cval = 0;

	for (int k = 0; k < K; ++k) {
		Cval += alpha * A[k * M + row] * B[K * col + k];
	}

	D[idx] = Cval + C[idx] * beta;
}

int myGEMM_no(double* A, double* B, double* C, double* D,
            double alpha, double beta, int M, int N, int K) {

	myGEMM_no_kernel <<<M, N>>> (A, B, C, D, alpha, beta, M, N, K);

	check_launch("myGEMM_no_kernel");

	return 0;	
}

__global__
void myGEMM_simple_kernel(double* A, double* B, double* C, double alpha, double beta, int M, int N, int K) {

	int row = blockIdx.x;
	int col = threadIdx.x;
	int idx = col * M + row;

	double Cval = 0;

	for (int k = 0; k < K; ++k) {
		Cval += alpha * A[k * M + row] * B[K * col + k];
	}

	C[idx] = Cval + C[idx] * beta;
}

int myGEMM(double* A, double* B, double* C, double* alpha, double* beta, int M, int N, int K){
	myGEMM_simple_kernel <<<M, N>>> (A, B, C, *alpha, *beta, M, N, K);
	check_launch("myGEMM_simple_kernel");
	return 0;
}
