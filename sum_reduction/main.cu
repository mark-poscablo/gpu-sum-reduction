#include <iostream>
#include <ctime>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils.h"

#include "reduce.h"

void generate_input(unsigned int* input, unsigned int input_len)
{
	for (unsigned int i = 0; i < input_len; ++i)
	{
		input[i] = i;
	}
}

unsigned int cpu_simple_sum(unsigned int* h_in, unsigned int h_in_len)
{
	unsigned int total_sum = 0;

	for (unsigned int i = 0; i < h_in_len; ++i)
	{
		total_sum = total_sum + h_in[i];
	}

	return total_sum;
}

int main()
{
	// Set up clock for timing comparisons
	std::clock_t start;
	double duration;

	for (int k = 1; k < 28; ++k)
	{
		unsigned int h_in_len = (1 << k);
		//unsigned int h_in_len = 2048;
		std::cout << "h_in_len: " << h_in_len << std::endl;
		unsigned int* h_in = new unsigned int[h_in_len];
		generate_input(h_in, h_in_len);
		//for (unsigned int i = 0; i < input_len; ++i)
		//{
		//	std::cout << input[i] << " ";
		//}
		//std::cout << std::endl;

		// Set up device-side memory for input
		unsigned int* d_in;
		checkCudaErrors(cudaMalloc(&d_in, sizeof(unsigned int) * h_in_len));
		checkCudaErrors(cudaMemcpy(d_in, h_in, sizeof(unsigned int) * h_in_len, cudaMemcpyHostToDevice));

		// Do CPU sum for reference
		start = std::clock();
		unsigned int cpu_total_sum = cpu_simple_sum(h_in, h_in_len);
		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		std::cout << cpu_total_sum << std::endl;
		std::cout << "CPU time: " << duration << " s" << std::endl;

		// Do GPU scan
		start = std::clock();
		unsigned int gpu_total_sum = gpu_sum_reduce(d_in, h_in_len);
		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		std::cout << gpu_total_sum << std::endl;
		std::cout << "GPU time: " << duration << " s" << std::endl;

		bool match = cpu_total_sum == gpu_total_sum;
		std::cout << "Match: " << match << std::endl;

		checkCudaErrors(cudaFree(d_in));
		delete[] h_in;

		std::cout << std::endl;
	}
}
