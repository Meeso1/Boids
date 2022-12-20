#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cudaUtils.cu"

int main(){
	int device;
	cudaGetDevice(&device);

	struct cudaDeviceProp props;
	cudaGetDeviceProperties(&props, device);

	printf("Device [%d]: %s (computability %d.%d)\n", device, props.name, props.major, props.minor);
	printf("Clock rate: %d kHz\n", props.clockRate);
	printf("Global memory: %zu\n", props.totalGlobalMem);
	printf("Shared memory per block: %zu\n", props.sharedMemPerBlock);
	printf("Max threads per block: %d\n", props.maxThreadsPerBlock);
	printf("Max threads per dimension: %d x %d x %d\n", props.maxThreadsDim[0], props.maxThreadsDim[1], props.maxThreadsDim[2]);
	printf("Max grid size: %d x %d x %d\n", props.maxGridSize[0], props.maxGridSize[1], props.maxGridSize[2]);

	return 0;
}