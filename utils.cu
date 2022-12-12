#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <time.h>
#include "cudaUtils.cu"

struct Fish{
	double* x;
	double* y;
	double* vx;
	double* vy;
};

struct Vector2{
	double x;
	double y;
};

void printVector(double* vector, size_t size){
	fprintf(stdout, "[");
	for(int i = 0; i < size; i++){
		fprintf(stdout, "%7.3f", vector[i]);
		if(i != size - 1) fprintf(stdout, ", ");
	}
	fprintf(stdout, "]\n");
}

void printFrame(const Fish* fishes, int num_of_fishes, double time){
	size_t vector_size = num_of_fishes*sizeof(double);
	double* tmp = (double*)malloc(vector_size);

	fprintf(stdout, "t = %6.3f:\n", time);

	deviceCopy(tmp, fishes->x, vector_size, cudaMemcpyDeviceToHost);
	printVector(tmp, num_of_fishes);

	deviceCopy(tmp, fishes->y, vector_size, cudaMemcpyDeviceToHost);
	printVector(tmp, num_of_fishes);

	deviceCopy(tmp, fishes->vx, vector_size, cudaMemcpyDeviceToHost);
	printVector(tmp, num_of_fishes);

	deviceCopy(tmp, fishes->vy, vector_size, cudaMemcpyDeviceToHost);
	printVector(tmp, num_of_fishes);

	free(tmp);
}

void print_elapsed(clock_t start, clock_t stop)
{
  double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
  printf("Elapsed time: %.3fs\n", elapsed);
}