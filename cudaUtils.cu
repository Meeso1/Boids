#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#ifndef CUDAUTILS
#define CUDAUTILS

size_t max_threads = 128;

struct kernelConfig{
	size_t blocks;
	size_t threads;
};

void deviceMalloc(void** pointer, size_t size){
	cudaError_t error = cudaMalloc(pointer, size);
	if(error != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector (error code: %s)\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
}

void deviceCopy(void* destination, const void* source, size_t size, cudaMemcpyKind kind){
	cudaError_t error = cudaMemcpy(destination, source, size, kind);
	if(error != cudaSuccess){
		fprintf(stderr, "Failed to copy vector (error code: %s)\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
}

void deviceFree(void* pointer){
	cudaError_t err = cudaFree(pointer);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to free device vector (error code: %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void deviceCheckErrors(const char* name){
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to launch %s (error code: %s)\n", name, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void deviceSynchronize(){
	cudaError_t err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to sync device (error code: %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

kernelConfig calculateKernelConfig(size_t total_num_of_threads, size_t max_threads_per_block_x){
	if(total_num_of_threads <= max_threads_per_block_x){
		return {1, total_num_of_threads};
	}

	double b = total_num_of_threads / ((double) max_threads_per_block_x);
	size_t num_of_blocks = (int)b == b ? (int)b : (int)b + 1;

	return {num_of_blocks, max_threads_per_block_x};
}

void setMaxThreads(){
	int device;
	cudaError_t err = cudaGetDevice(&device);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to get device (error code: %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	struct cudaDeviceProp props;
	err = cudaGetDeviceProperties(&props, device);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to get device properties (error code: %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	max_threads = props.maxThreadsDim[0] / 2;
}

#endif