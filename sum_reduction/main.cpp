#include <iostream>
#include <ctime>

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

	unsigned int input_len = 4096;
	unsigned int* input = new unsigned int[input_len];
	generate_input(input, input_len);
	//for (unsigned int i = 0; i < input_len; ++i)
	//{
	//	std::cout << input[i] << " ";
	//}
	//std::cout << std::endl;
	start = std::clock();
	unsigned int cpu_total_sum = cpu_simple_sum(input, input_len);
	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	std::cout << cpu_total_sum << std::endl;
	std::cout << "CPU time: " << duration << " s" << std::endl;

	delete[] input;
}
