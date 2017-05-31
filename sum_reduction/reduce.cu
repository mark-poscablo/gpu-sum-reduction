#include "reduce.h"

#include <cmath>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils.h"

__global__
void block_sum_reduce(unsigned int* const d_block_sums, 
	const unsigned int* const d_in,
	const unsigned int d_in_len)
{
	extern __shared__ unsigned int s_out[];

	unsigned int glbl_t_idx = blockDim.x * blockIdx.x + threadIdx.x;

	// Zero out shared memory
	// Especially important when padding shmem for
	//  non-power of 2 sized input
	s_out[2 * threadIdx.x] = 0;
	s_out[2 * threadIdx.x + 1] = 0;

	__syncthreads();

	// Copy d_in to shared memory per block
	if (2 * glbl_t_idx < d_in_len)
	{
		s_out[2 * threadIdx.x] = d_in[2 * glbl_t_idx];
		if (2 * glbl_t_idx + 1 < d_in_len)
			s_out[2 * threadIdx.x + 1] = d_in[2 * glbl_t_idx + 1];
	}

	__syncthreads();

	// 2^11 = 2048, the max amount of data a block can blelloch scan
	unsigned int max_steps = 11;

	unsigned int r_idx = 0;
	unsigned int l_idx = 0;
	unsigned int sum = 0; // global sum can be passed to host if needed
	unsigned int t_active = 0;
	for (unsigned int s = 0; s < max_steps; ++s)
	{
		t_active = 0;

		// calculate necessary indexes
		// right index must be (t+1) * 2^(s+1)) - 1
		r_idx = ((threadIdx.x + 1) * (1 << (s + 1))) - 1;
		if (r_idx >= 0 && r_idx < 2048)
			t_active = 1;

		if (t_active)
		{
			// left index must be r_idx - 2^s
			l_idx = r_idx - (1 << s);

			// do the actual add operation
			sum = s_out[l_idx] + s_out[r_idx];
		}
		__syncthreads();

		if (t_active)
			s_out[r_idx] = sum;
		__syncthreads();
	}

	// Copy last element (total sum of block) to block sums array
	// Then, reset last element to operation's identity (sum, 0)
	if (threadIdx.x == 0)
	{
		d_block_sums[blockIdx.x] = s_out[r_idx];
	}
}

unsigned int gpu_sum_reduce(unsigned int* d_in, unsigned int d_in_len)
{
	unsigned int total_sum = 0;

	// Set up number of threads and blocks
	// If input size is not power of two, the remainder will still need a whole block
	// Thus, number of blocks must be the least number of 2048-blocks greater than the input size
	unsigned int block_sz = MAX_BLOCK_SZ;
	unsigned int max_elems_per_block = block_sz * 2; // due to binary tree nature of algorithm
	unsigned int grid_sz = (unsigned int)std::ceil(float(d_in_len) / float(max_elems_per_block));

	// Allocate memory for array of total sums produced by each block
	// Array length must be the same as number of blocks / grid size
	unsigned int* d_block_sums;
	checkCudaErrors(cudaMalloc(&d_block_sums, sizeof(unsigned int) * grid_sz));
	checkCudaErrors(cudaMemset(d_block_sums, 0, sizeof(unsigned int) * grid_sz));

	// Sum data allocated for each block
	block_sum_reduce<<<grid_sz, block_sz, sizeof(unsigned int) * max_elems_per_block>>>(d_block_sums, d_in, d_in_len);

	// Sum each block's total sums (to get global total sum)
	// Use basic implementation if number of total sums is <= 2048
	// Else, recurse on this same function
	if (grid_sz <= 2048)
	{
		unsigned int* d_total_sum;
		checkCudaErrors(cudaMalloc(&d_total_sum, sizeof(unsigned int)));
		checkCudaErrors(cudaMemset(d_total_sum, 0, sizeof(unsigned int)));
		block_sum_reduce<<<1, block_sz, sizeof(unsigned int) * max_elems_per_block>>>(d_total_sum, d_block_sums, grid_sz);
		checkCudaErrors(cudaMemcpy(&total_sum, d_total_sum, sizeof(unsigned int), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaFree(d_total_sum));
	}
	else
	{
		unsigned int* d_in_block_sums;
		checkCudaErrors(cudaMalloc(&d_in_block_sums, sizeof(unsigned int) * grid_sz));
		checkCudaErrors(cudaMemcpy(d_in_block_sums, d_block_sums, sizeof(unsigned int) * grid_sz, cudaMemcpyDeviceToDevice));
		total_sum = gpu_sum_reduce(d_in_block_sums, grid_sz);
		checkCudaErrors(cudaFree(d_in_block_sums));
	}

	checkCudaErrors(cudaFree(d_block_sums));
	return total_sum;
}
