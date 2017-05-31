#ifndef REDUCE_H__
#define REDUCE_H__

#define MAX_BLOCK_SZ 1024

unsigned int gpu_sum_reduce(unsigned int* d_in, unsigned int d_in_len);

#endif // !REDUCE_H__


