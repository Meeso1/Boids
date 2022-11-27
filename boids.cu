#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define NUM_OF_FISHES 50
#define SCENE_SIZE 100
#define MAX_INIT_V 2
#define MAX_V 8
#define INTERACTION1_RADIUS 5
#define INTERACTION2_RADIUS 5
#define INTERACTION3_RADIUS 10
#define INTERACTION4_RADIUS 5
#define ATTRACTION_STR 2
#define SEPARATION_STR 2
#define ALIGNMENT_STR 5
#define REPULSION_STR 1

struct Fish{
  double* x;
  double* y;
  double* vx;
  double* vy;
};

__host__ __device__ double length(double x, double y){
  return sqrt(x*x + y*y);
}

__device__ void clip_speed(double max, double* vx, double* vy){
  double len = length(*vx, *vy);
  
  if(len < max || len <= 0){
    return;
  }

  *vx = *vx / len * max;
  *vy = *vy / len * max;
}

__global__ void updateFish(const Fish in, Fish out, double dt, int numElements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i >= numElements){
    return;
  }

  // Rule 1: Attraction
  double centerX = 0;
  double centerY = 0;
  int rule1Neighbours = 0;
  for(int k = 0; k < numElements; k++){
    if(k == i) continue;

    double distance = length(in.x[i] - in.x[k], in.y[i] - in.y[k]);
    if(distance > INTERACTION1_RADIUS) continue;

    centerX += in.x[k];
    centerY += in.y[k];
    rule1Neighbours++;
  }

  double vx1 = 0;
  double vy1 = 0;
  if(rule1Neighbours != 0){
    centerX /= rule1Neighbours;
    centerY /= rule1Neighbours;

    vx1 = (centerX - in.x[i]) * ATTRACTION_STR;
    vy1 = (centerY - in.y[i]) * ATTRACTION_STR;
  }

  // Rule 2: Separation
  double sepX = 0;
  double sepY = 0;
  for(int k = 0; k < numElements; k++){
    if(k == i) continue;

    double distance = length(in.x[i] - in.x[k], in.y[i] - in.y[k]);
    if(distance > INTERACTION2_RADIUS) continue;

    sepX -= in.x[k] - in.x[i];
    sepY -= in.y[k] - in.y[i];
  }

  double vx2 = sepX * SEPARATION_STR;
  double vy2 = sepY * SEPARATION_STR;

  // Rule 3: Alignment
  double vx3 = in.vx[i];
  double vy3 = in.vy[i];
  int rule3Neighbours = 0;
  for(int k = 0; k < numElements; k++){
    if(k == i) continue;

    double distance = length(in.x[i] - in.x[k], in.y[i] - in.y[k]);
    if(distance > INTERACTION1_RADIUS) continue;

    vx3 += in.vx[k];
    vy3 += in.vy[k];
    rule3Neighbours++;
  }

  if(rule1Neighbours != 0){
    vx3 -= in.vx[i];
    vy3 -= in.vy[i];

    vx3 /= rule3Neighbours;
    vy3 /= rule3Neighbours;
  }

  // Rule 4: Avoidance
  double vx4 = 0;
  double vy4 = 0;
  if(in.x[i] < INTERACTION4_RADIUS){
    vx4 = (INTERACTION4_RADIUS - in.x[i]) * REPULSION_STR;
  }
  else if(in.x[i] > SCENE_SIZE - INTERACTION4_RADIUS){
    vx4 = - (in.x[i] - SCENE_SIZE + INTERACTION4_RADIUS) * REPULSION_STR;
  }

  if(in.y[i] < INTERACTION4_RADIUS){
    vy4 = (INTERACTION4_RADIUS - in.y[i]) * REPULSION_STR;
  }
  else if(in.y[i] > SCENE_SIZE - INTERACTION4_RADIUS){
    vy4 = - (in.y[i] - SCENE_SIZE + INTERACTION4_RADIUS) * REPULSION_STR;
  }

  double ax = (vx1 + vx2 + vx3 + vx4) - in.vx[i];
  double ay = (vy1 + vy2 + vy3 + vy4) - in.vy[i];

  // Doesn't work :(
  out.vx[i] = in.vx[i] + ax; //* dt;
  out.vy[i] = in.vy[i] + ay; //* dt;

  clip_speed(MAX_V, &(out.vx[i]), &(out.vy[i]));

  out.x[i]  = in.x[i]  + out.vx[i] * dt;
  out.y[i]  = in.y[i]  + out.vy[i] * dt;
}

Fish* initFish(int number){
  srand(1234567890);

  Fish* fish = (Fish*)malloc(sizeof(Fish));
  
  fish->x = (double*)malloc(number*sizeof(double));
  fish->y = (double*)malloc(number*sizeof(double));
  fish->vx = (double*)malloc(number*sizeof(double));
  fish->vy = (double*)malloc(number*sizeof(double));

  for(int i = 0; i < number; i++){
    fish->x[i]  = (rand() % (SCENE_SIZE * 100)) / 100.0;
    fish->y[i]  = (rand() % (SCENE_SIZE * 100)) / 100.0;
    fish->vx[i] = (rand() % (MAX_INIT_V * 100)) / 100.0 - (MAX_INIT_V / 2.0);
    fish->vy[i] = (rand() % (MAX_INIT_V * 100)) / 100.0 - (MAX_INIT_V / 2.0);
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
  double max_time = 100;
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