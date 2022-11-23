#include <stdio.h>
#include <cuda_runtime.h>

#define NUM_OF_FISHES 10

struct Fish{
  double* x;
  double* y;
  double* vx;
  double* vy;
};

__global__ void updateFish(const Fish in, Fish out, double dt, int numElements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < numElements) {
    out.x[i]  = in.x[i]  + in.vx[i] * dt;
    out.y[i]  = in.y[i]  + in.vy[i] * dt;
    out.vy[i] = in.vy[i] - 1.0 * dt;
  }
}

Fish* initFish(int number){
  Fish* fish = (Fish*)malloc(sizeof(Fish));
  
  fish->x = (double*)malloc(number*sizeof(double));
  fish->y = (double*)malloc(number*sizeof(double));
  fish->vx = (double*)malloc(number*sizeof(double));
  fish->vy = (double*)malloc(number*sizeof(double));

  for(int i = 0; i < number; i++){
    fish->x[i] = i;
    fish->y[i] = 0;
    fish->vx[i] = 0;
    fish->vy[i] = 5;
  }

  return fish;
}

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

void printVector(double* vector, size_t size){
  fprintf(stdout, "[");
  for(int i = 0; i < size; i++){
    fprintf(stdout, "%7.3f", vector[i]);
    if(i != size - 1) fprintf(stdout, ", ");
  }
  fprintf(stdout, "]\n");
}

int main(void) {
  // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;

  Fish* fish = initFish(NUM_OF_FISHES);
  size_t vector_size = NUM_OF_FISHES*sizeof(double);

  // Allocate the device input vector
  Fish* d_in_fish = (Fish*)malloc(sizeof(Fish));
  d_in_fish->x = NULL;
  d_in_fish->y = NULL;
  d_in_fish->vx = NULL;
  d_in_fish->vy = NULL;
  deviceMalloc((void**) &(d_in_fish->x),  vector_size);
  deviceMalloc((void**) &(d_in_fish->y),  vector_size);
  deviceMalloc((void**) &(d_in_fish->vx), vector_size);
  deviceMalloc((void**) &(d_in_fish->vy), vector_size);

  // Allocate the device output vector
  Fish* d_out_fish = (Fish*)malloc(sizeof(Fish));
  d_out_fish->x = NULL;
  d_out_fish->y = NULL;
  d_out_fish->vx = NULL;
  d_out_fish->vy = NULL;
  deviceMalloc((void**) &(d_out_fish->x),  vector_size);
  deviceMalloc((void**) &(d_out_fish->y),  vector_size);
  deviceMalloc((void**) &(d_out_fish->vx), vector_size);
  deviceMalloc((void**) &(d_out_fish->vy), vector_size);

  // Copy to input vector
  deviceCopy(d_in_fish->x, fish->x, vector_size, cudaMemcpyHostToDevice);
  deviceCopy(d_in_fish->y, fish->y, vector_size, cudaMemcpyHostToDevice);
  deviceCopy(d_in_fish->vx, fish->vx, vector_size, cudaMemcpyHostToDevice);
  deviceCopy(d_in_fish->vy, fish->vy, vector_size, cudaMemcpyHostToDevice);

  // Launch the Vector Add CUDA Kernel
  int threadsPerBlock = NUM_OF_FISHES;
  int blocksPerGrid = 1;
  
  double dt = 0.01;
  double time = 0;
  double max_time = 15;
  double* tmp = (double*)malloc(vector_size);

  fprintf(stdout, "t = %6.3f:\n", time);
  deviceCopy(tmp, d_in_fish->x, vector_size, cudaMemcpyDeviceToHost);
  printVector(tmp, NUM_OF_FISHES);
  deviceCopy(tmp, d_in_fish->y, vector_size, cudaMemcpyDeviceToHost);
  printVector(tmp, NUM_OF_FISHES);
  deviceCopy(tmp, d_in_fish->vx, vector_size, cudaMemcpyDeviceToHost);
  printVector(tmp, NUM_OF_FISHES);
  deviceCopy(tmp, d_in_fish->vy, vector_size, cudaMemcpyDeviceToHost);
  printVector(tmp, NUM_OF_FISHES);

  while(time < max_time){
    updateFish<<<blocksPerGrid, threadsPerBlock>>>(*d_in_fish, *d_out_fish, dt, NUM_OF_FISHES);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    deviceCopy(d_in_fish->x, d_out_fish->x, vector_size, cudaMemcpyDeviceToDevice);
    deviceCopy(d_in_fish->y, d_out_fish->y, vector_size, cudaMemcpyDeviceToDevice);
    deviceCopy(d_in_fish->vx, d_out_fish->vx, vector_size, cudaMemcpyDeviceToDevice);
    deviceCopy(d_in_fish->vy, d_out_fish->vy, vector_size, cudaMemcpyDeviceToDevice);

    time += dt;

    fprintf(stdout, "t = %6.3f:\n", time);
    deviceCopy(tmp, d_out_fish->x, vector_size, cudaMemcpyDeviceToHost);
    printVector(tmp, NUM_OF_FISHES);
    deviceCopy(tmp, d_out_fish->y, vector_size, cudaMemcpyDeviceToHost);
    printVector(tmp, NUM_OF_FISHES);
    deviceCopy(tmp, d_out_fish->vx, vector_size, cudaMemcpyDeviceToHost);
    printVector(tmp, NUM_OF_FISHES);
    deviceCopy(tmp, d_out_fish->vy, vector_size, cudaMemcpyDeviceToHost);
    printVector(tmp, NUM_OF_FISHES);
  }

  // Free device memory
  deviceFree(d_in_fish->x);
  deviceFree(d_in_fish->y);
  deviceFree(d_in_fish->vx);
  deviceFree(d_in_fish->vy);
  deviceFree(d_out_fish->x);
  deviceFree(d_out_fish->y);
  deviceFree(d_out_fish->vx);
  deviceFree(d_out_fish->vy);

  // Free host memory
  free(fish->x);
  free(fish->y);
  free(fish->vx);
  free(fish->vy);
  free(fish);
  free(tmp);

  return 0;
}