CUDA_INCLUDEPATH=/usr/local/cuda-5.0/include
NVCC_OPTS=-O3 -arch=sm_37 -Xcompiler -Wall -Xcompiler -Wextra -m64

reduce: main.cu reduce.o Makefile
	nvcc -o reduce main.cu reduce.o $(NVCC_OPTS)

reduce.o: reduce.cu
	nvcc -c reduce.cu $(NVCC_OPTS)

clean:
	rm -f *.o reduce
