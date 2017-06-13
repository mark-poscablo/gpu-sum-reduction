#include "reduce.h"

#include <cmath>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils.h"

// Bandwidth: (((2^27) + 1) unsigned ints * 4 bytes/unsigned int)/(38.716 * 10^-3 s)
//  13.867 GB/s = 96.297% -> excellent memory bandwidth
// Reasonable point to stop working on this implementation's optimization
// Algorithm is not compute-intensive, so acheiving >75% of theoretical bandwidth is goal
// Main strategies used:
// - Process as much data as possible (in terms of algorithm correctness) in shared memory
// - Use sequential addressing to get rid of bank conflicts
__global__
void block_sum_reduce(unsigned int* const d_block_sums, 
	const unsigned int* const d_in,
	const unsigned int d_in_len)
{
	extern __shared__ unsigned int s_out[];

	unsigned int max_elems_per_block = blockDim.x * 2;
	unsigned int glbl_tid = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int tid = threadIdx.x;
	
	// Zero out shared memory
	// Especially important when padding shmem for
	//  non-power of 2 sized input
	s_out[threadIdx.x] = 0;
	s_out[threadIdx.x + blockDim.x] = 0;

	__syncthreads();

	// Copy d_in to shared memory per block
	if (glbl_tid < d_in_len)
	{
		s_out[threadIdx.x] = d_in[glbl_tid];
		if (glbl_tid + blockDim.x < d_in_len)
			s_out[threadIdx.x + blockDim.x] = d_in[glbl_tid + blockDim.x];
	}
	__syncthreads();

	// Actually do the reduction
	for (unsigned int s = blockDim.x; s > 0; s >>= 1) {
		if (tid < s) {
			s_out[tid] += s_out[tid + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0)
		d_block_sums[blockIdx.x] = s_out[0];
}

// On my current laptop with GTX 850M, theoretical peak bandwidth is 14.4 GB/s
// Shared memory of GTX 850M has 32 memory banks
// Succeeding measurements are for the Release build

// Bandwidth: (((2^27) + 1) unsigned ints * 4 bytes/unsigned int)/(173.444 * 10^-3 s)
//  3.095 GB/s = 21.493% -> bad kernel memory bandwidth
__global__ void reduce0(unsigned int* g_odata, unsigned int* g_idata, unsigned int len) {
	extern __shared__ unsigned int sdata[];

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	sdata[tid] = 0;

	if (i < len)
	{
		sdata[tid] = g_idata[i];
	}

	__syncthreads();

	// do reduction in shared mem
	// Interleaved addressing, which causes huge thread divergence
	//  because threads are active/inactive according to their thread IDs
	//  being powers of two. The if conditional here is guaranteed to diverge
	//  threads within a warp.
	for (unsigned int s = 1; s < 2048; s <<= 2) {
		if (tid % (2 * s) == 0) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0) 
		g_odata[blockIdx.x] = sdata[0];
}

// Bandwidth: (((2^27) + 1) unsigned ints * 4 bytes/unsigned int)/(81.687 * 10^-3 s)
//  6.572 GB/s = 45.639% -> bad kernel memory bandwidth, but better than last time
__global__ void reduce1(unsigned int* g_odata, unsigned int* g_idata, unsigned int len) {
	extern __shared__ unsigned int sdata[];

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	sdata[tid] = 0;

	if (i < len)
	{
		sdata[tid] = g_idata[i];
	}

	__syncthreads();

	// do reduction in shared mem
	// Interleaved addressing, but threads being active/inactive
	//  is no longer based on thread IDs being powers of two. Consecutive
	//  threadIDs now run, and thus solves the thread diverging issue within
	//  a warp
	// However, this introduces shared memory bank conflicts, as threads start 
	//  out addressing with a stride of two 32-bit words (unsigned ints),
	//  and further increase the stride as the current power of two grows larger
	//  (which can worsen or lessen bank conflicts, depending on the amount
	//  of stride)
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		unsigned int index = 2 * s * tid;

		if (index < blockDim.x) {
			sdata[index] += sdata[index + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = sdata[0];
}

// Bandwidth: (((2^27) + 1) unsigned ints * 4 bytes/unsigned int)/(67.699 * 10^-3 s)
//  7.930 GB/s = 55.069% -> good kernel memory bandwidth
__global__ void reduce2(unsigned int* g_odata, unsigned int* g_idata, unsigned int len) {
	extern __shared__ unsigned int sdata[];

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	sdata[tid] = 0;

	if (i < len)
	{
		sdata[tid] = g_idata[i];
	}

	__syncthreads();

	// do reduction in shared mem
	// Sequential addressing. This solves the bank conflicts as
	//  the threads now access shared memory with a stride of one
	//  32-bit word (unsigned int) now, which does not cause bank 
	//  conflicts
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = sdata[0];
}

// Bandwidth: (((2^27) + 1) unsigned ints * 4 bytes/unsigned int)/(58.505 * 10^-3 s)
//  9.176 GB/s = 63.722% -> good kernel memory bandwidth, better than last time
__global__ void reduce3(unsigned int* g_odata, unsigned int* g_idata, unsigned int len) {
	extern __shared__ unsigned int sdata[];

	// each thread loads one element from global to shared mem
	// Do the first stage of the reduction on the global-to-shared load step
	// This reduces the previous inefficiency of having half of the threads being
	//  inactive on the first for-loop iteration below (previous first step of reduction)
	// Previously, only less than or equal to 512 out of 1024 threads in a block are active.
	// Now, all 512 threads in a block are active from the start
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	sdata[tid] = 0;

	if (i < len)
	{
		sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
	}

	__syncthreads();

	// do reduction in shared mem
	// this loop now starts with s = 512 / 2 = 256
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = sdata[0];
}

// Bandwidth: (((2^27) + 1) unsigned ints * 4 bytes/unsigned int)/(37.255 * 10^-3 s)
//  14.411 GB/s = 100% -> perfect bandwidth? is this even possible?
// ***In my laptop, measurements are wrong since the release version oddly outputs an incorrect value***
__global__ void reduce4(unsigned int* g_odata, unsigned int* g_idata, unsigned int len) {
	extern __shared__ unsigned int sdata[];

	// each thread loads one element from global to shared mem
	// Do the first stage of the reduction on the global-to-shared load step
	// This reduces the previous inefficiency of having half of the threads being
	//  inactive on the first for-loop iteration below (previous first step of reduction)
	// Previously, only less than or equal to 512 out of 1024 threads in a block are active.
	// Now, all 512 threads in a block are active from the start
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	sdata[tid] = 0;

	if (i < len)
	{
		sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
	}

	__syncthreads();

	// do reduction in shared mem
	// this loop now starts with s = 512 / 2 = 256
	for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	if (tid < 32)
	{
		sdata[tid] += sdata[tid + 32];
		sdata[tid] += sdata[tid + 16];
		sdata[tid] += sdata[tid + 8];
		sdata[tid] += sdata[tid + 4];
		sdata[tid] += sdata[tid + 2];
		sdata[tid] += sdata[tid + 1];
	}

	// write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = sdata[0];
}

void print_d_array(unsigned int* d_array, unsigned int len)
{
	unsigned int* h_array = new unsigned int[len];
	checkCudaErrors(cudaMemcpy(h_array, d_array, sizeof(unsigned int) * len, cudaMemcpyDeviceToHost));
	for (unsigned int i = 0; i < len; ++i)
	{
		std::cout << h_array[i] << " ";
	}
	std::cout << std::endl;

	delete[] h_array;
}

unsigned int gpu_sum_reduce(unsigned int* d_in, unsigned int d_in_len)
{
	unsigned int total_sum = 0;

	// Set up number of threads and blocks
	// If input size is not power of two, the remainder will still need a whole block
	// Thus, number of blocks must be the least number of 2048-blocks greater than the input size
	unsigned int block_sz = MAX_BLOCK_SZ; // Halve the block size due to reduce3() and further 
											  //  optimizations from there
	// our block_sum_reduce()
	unsigned int max_elems_per_block = block_sz * 2; // due to binary tree nature of algorithm
	// NVIDIA's reduceX()
	//unsigned int max_elems_per_block = block_sz;
	
	unsigned int grid_sz = 0;
	if (d_in_len <= max_elems_per_block)
	{
		grid_sz = (unsigned int)std::ceil(float(d_in_len) / float(max_elems_per_block));
	}
	else
	{
		grid_sz = d_in_len / max_elems_per_block;
		if (d_in_len % max_elems_per_block != 0)
			grid_sz++;
	}

	// Allocate memory for array of total sums produced by each block
	// Array length must be the same as number of blocks / grid size
	unsigned int* d_block_sums;
	checkCudaErrors(cudaMalloc(&d_block_sums, sizeof(unsigned int) * grid_sz));
	checkCudaErrors(cudaMemset(d_block_sums, 0, sizeof(unsigned int) * grid_sz));

	// Sum data allocated for each block
	block_sum_reduce<<<grid_sz, block_sz, sizeof(unsigned int) * max_elems_per_block>>>(d_block_sums, d_in, d_in_len);
	//reduce4<<<grid_sz, block_sz, sizeof(unsigned int) * block_sz>>>(d_block_sums, d_in, d_in_len);
	//print_d_array(d_block_sums, grid_sz);

	// Sum each block's total sums (to get global total sum)
	// Use basic implementation if number of total sums is <= 2048
	// Else, recurse on this same function
	if (grid_sz <= max_elems_per_block)
	{
		unsigned int* d_total_sum;
		checkCudaErrors(cudaMalloc(&d_total_sum, sizeof(unsigned int)));
		checkCudaErrors(cudaMemset(d_total_sum, 0, sizeof(unsigned int)));
		block_sum_reduce<<<1, block_sz, sizeof(unsigned int) * max_elems_per_block>>>(d_total_sum, d_block_sums, grid_sz);
		//reduce4<<<1, block_sz, sizeof(unsigned int) * block_sz>>>(d_total_sum, d_block_sums, grid_sz);
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
