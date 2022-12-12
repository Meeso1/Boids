#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#ifndef CUDAUTILS
#define CUDAUTILS

void deviceMalloc(void** pointer, size_t size){
	cudaError_t error = cudaMalloc(pointer, size);
	if(error != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
}

void deviceCopy(void* destination, const void* source, size_t size, cudaMemcpyKind kind){
	cudaError_t error = cudaMemcpy(destination, source, size, kind);
	if(error != cudaSuccess){
		fprintf(stderr, "Failed to copy vector (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
}

void deviceFree(void* pointer){
	cudaError_t err = cudaFree(pointer);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to free device vector (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

#endif