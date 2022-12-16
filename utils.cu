#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <time.h>
#include "cudaUtils.cu"

#ifndef UTILS
#define UTILS

#ifndef DEBUG_MSG
#define DEBUG_MSG 0
#endif

#define DEBUG(...) if(DEBUG_MSG != 0) printf(__VA_ARGS__)
#define IN_DEBUG(expr) if(DEBUG_MSG != 0) do{expr;}while(0)
#define B(i) DEBUG("[%2d]======>\n", i)
#define T(s) DEBUG("[%s]\n", s)

#ifdef USE_3D
#define VECT_0 {0, 0, 0}
#else
#define VECT_0 {0, 0}
#endif

struct Fish{
	double* x;
	double* y;
	double* vx;
	double* vy;
	#ifdef USE_3D
	double* z;
	double* vz;
	#endif
	#ifdef TYPES
	int type;
	#endif
};

struct Vector{
	double x;
	double y;
	#ifdef USE_3D
	double z;
	#endif
};

void print_int_array(int* array, size_t length){
	printf("[");
	for(int i = 0; i < length; i++){
		printf("%2d", array[i]);
		if(i != length - 1){
			printf(", ");
		}
	}
	printf("]\n");
}

void print_double_array(double* vector, size_t size){
	fprintf(stdout, "[");
	for(int i = 0; i < size; i++){
		fprintf(stdout, "%7.3f", vector[i]);
		if(i != size - 1) fprintf(stdout, ", ");
	}
	fprintf(stdout, "]\n");
}

void print_dev_int_array(int* dev_array, size_t size){
	int* tmp = (int*)malloc(size*sizeof(int));
	printf("[%p]: ", dev_array);
	deviceCopy(tmp, dev_array, size*sizeof(int), cudaMemcpyDeviceToHost);
	print_int_array(tmp, size);
	free(tmp);
}

void printFrame(const Fish* fishes, int num_of_fishes, double time){
	size_t vector_size = num_of_fishes*sizeof(double);
	double* tmp = (double*)malloc(vector_size);

	fprintf(stdout, "t = %6.3f:\n", time);

	deviceCopy(tmp, fishes->x, vector_size, cudaMemcpyDeviceToHost);
	print_double_array(tmp, num_of_fishes);

	deviceCopy(tmp, fishes->y, vector_size, cudaMemcpyDeviceToHost);
	print_double_array(tmp, num_of_fishes);

	#ifdef USE_3D
	deviceCopy(tmp, fishes->z, vector_size, cudaMemcpyDeviceToHost);
	print_double_array(tmp, num_of_fishes);
	#endif

	deviceCopy(tmp, fishes->vx, vector_size, cudaMemcpyDeviceToHost);
	print_double_array(tmp, num_of_fishes);

	deviceCopy(tmp, fishes->vy, vector_size, cudaMemcpyDeviceToHost);
	print_double_array(tmp, num_of_fishes);

	#ifdef USE_3D
	deviceCopy(tmp, fishes->vz, vector_size, cudaMemcpyDeviceToHost);
	print_double_array(tmp, num_of_fishes);
	#endif

	free(tmp);
}

void print_elapsed(clock_t start, clock_t stop){
  double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
  printf("Elapsed time: %.3fs\n", elapsed);
}

#endif