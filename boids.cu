#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include "cudaUtils.cu"
#include "utils.cu"

#define NUM_OF_FISHES 50
#define SCENE_SIZE 100

#define MIN_V 1
#define MAX_V 10

#define ATTRACTION_STR 0.3
#define INTERACTION1_RADIUS 8

#define SEPARATION_STR 0.5
#define INTERACTION2_RADIUS 3

#define ALIGNMENT_STR 0.2
#define INTERACTION3_RADIUS 5

#define REPULSION_STR 8
#define INTERACTION4_RADIUS 8

#define COS_FOV 0

__host__ __device__ double length(double x, double y){
  return sqrt(x*x + y*y);
}

__host__ __device__ void clip_speed(double min, double max, double* vx, double* vy){
  double len = length(*vx, *vy);
  
  if(len <= 0){
    return;
  }

  if(len > max){
    *vx = *vx / len * max;
    *vy = *vy / len * max;
  }
  else if(len < min){
    *vx = *vx / len * min;
    *vy = *vy / len * min;
  }
}

__device__ bool is_in_front(Vector2 fish, Vector2 other, Vector2 v, double cos_fov){
  Vector2 r = {other.x - fish.x, other.y - fish.y};
  double r_length = length(r.x, r.y);
  double v_length = length(v.x, v.y);
  if(r_length == 0 || v_length == 0) return true;
  double cos_r_v = (r.x * v.x + r.y * v.y) / (r_length * v_length);
  return cos_r_v >= cos_fov;
}

__global__ void updateFish(const Fish in, Fish out, double dt, int numElements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i >= numElements){
    return;
  }

  // Rule 1: Attraction
  Vector2 center = {0, 0};
  int rule1Neighbours = 0;
  for(int k = 0; k < numElements; k++){
    if(k == i) continue;

    double distance = length(in.x[i] - in.x[k], in.y[i] - in.y[k]);
    if(distance > INTERACTION1_RADIUS) continue;

    if(!is_in_front({in.x[i], in.y[i]}, {in.x[k], in.y[k]}, {in.vx[i], in.vy[i]}, COS_FOV)) continue;

    center.x += in.x[k];
    center.y += in.y[k];
    rule1Neighbours++;
  }

  Vector2 v1 = {0, 0};
  if(rule1Neighbours != 0){
    center.x /= rule1Neighbours;
    center.y /= rule1Neighbours;

    v1.x = (center.x - in.x[i]) * ATTRACTION_STR;
    v1.y = (center.y - in.y[i]) * ATTRACTION_STR;
  }

  // Rule 2: Separation
  Vector2 sep = {0, 0};
  for(int k = 0; k < numElements; k++){
    if(k == i) continue;

    double distance = length(in.x[i] - in.x[k], in.y[i] - in.y[k]);
    if(distance > INTERACTION2_RADIUS) continue;

    if(!is_in_front({in.x[i], in.y[i]}, {in.x[k], in.y[k]}, {in.vx[i], in.vy[i]}, COS_FOV)) continue;

    sep.x -= in.x[k] - in.x[i];
    sep.y -= in.y[k] - in.y[i];
  }

  Vector2 v2 = {sep.x * SEPARATION_STR, sep.y * SEPARATION_STR};

  // Rule 3: Alignment
  Vector2 v_sum = {0, 0};
  int rule3Neighbours = 0;
  for(int k = 0; k < numElements; k++){
    if(k == i) continue;

    double distance = length(in.x[i] - in.x[k], in.y[i] - in.y[k]);
    if(distance > INTERACTION1_RADIUS) continue;

    if(!is_in_front({in.x[i], in.y[i]}, {in.x[k], in.y[k]}, {in.vx[i], in.vy[i]}, COS_FOV)) continue;

    v_sum.x += in.vx[k];
    v_sum.y += in.vy[k];
    rule3Neighbours++;
  }

  Vector2 v3 = {0, 0};
  if(rule3Neighbours != 0){
    v3 = {v_sum.x / rule3Neighbours - in.vx[i], v_sum.y / rule3Neighbours - in.vy[i]};
    v3.x *= ALIGNMENT_STR;
    v3.y *= ALIGNMENT_STR;
  }

  // Rule 4: Avoidance
  Vector2 v4 = {0, 0};
  if(in.x[i] < INTERACTION4_RADIUS){
    v4.x = (INTERACTION4_RADIUS - in.x[i]) * REPULSION_STR;
  }
  else if(in.x[i] > SCENE_SIZE - INTERACTION4_RADIUS){
    v4.x = - (in.x[i] - SCENE_SIZE + INTERACTION4_RADIUS) * REPULSION_STR;
  }

  if(in.y[i] < INTERACTION4_RADIUS){
    v4.y = (INTERACTION4_RADIUS - in.y[i]) * REPULSION_STR;
  }
  else if(in.y[i] > SCENE_SIZE - INTERACTION4_RADIUS){
    v4.y = - (in.y[i] - SCENE_SIZE + INTERACTION4_RADIUS) * REPULSION_STR;
  }

  Vector2 dv = {v1.x + v2.x + v3.x + v4.x, v1.y + v2.y + v3.y + v4.y};
  out.vx[i] = in.vx[i] + dv.x;
  out.vy[i] = in.vy[i] + dv.y;

  clip_speed(MIN_V, MAX_V, &(out.vx[i]), &(out.vy[i]));

  out.x[i] = in.x[i] + out.vx[i] * dt;
  out.y[i] = in.y[i] + out.vy[i] * dt;
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

    do{
      fish->vx[i] = (rand() % (2 * MAX_V * 100)) / 100.0 - MAX_V;
      fish->vy[i] = (rand() % (2 * MAX_V * 100)) / 100.0 - MAX_V;
      clip_speed(MIN_V, MAX_V, &(fish->vx[i]), &(fish->vy[i]));
    }while(fish->vx[i] * fish->vy[i] == 0);
  }

  return fish;
}

void initSimulation(Fish** device_in_fishes, Fish** device_out_fishes, int num_of_fishes){
  Fish* fish = initFish(num_of_fishes);
  size_t vector_size = num_of_fishes*sizeof(double);

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

  *device_in_fishes = d_in_fish;
  *device_out_fishes = d_out_fish;

  free(fish->x);
  free(fish->y);
  free(fish->vx);
  free(fish->vy);
  free(fish);
}

void copyFishes(const Fish* source, Fish* destination, int num_of_fishes){
  size_t vector_size = num_of_fishes*sizeof(double);
  deviceCopy(source->x, destination->x, vector_size, cudaMemcpyDeviceToDevice);
  deviceCopy(source->y, destination->y, vector_size, cudaMemcpyDeviceToDevice);
  deviceCopy(source->vx, destination->vx, vector_size, cudaMemcpyDeviceToDevice);
  deviceCopy(source->vy, destination->vy, vector_size, cudaMemcpyDeviceToDevice);
}

void freeFishes(Fish* fishes){
  deviceFree(fishes->x);
  deviceFree(fishes->y);
  deviceFree(fishes->vx);
  deviceFree(fishes->vy);
}

void advance(Fish* in_fishes, Fish* out_fishes, int num_of_fishes, double dt){
  // Launch CUDA Kernel
  updateFish<<<1, num_of_fishes>>>(*in_fishes, *out_fishes, dt, num_of_fishes);

  // Check errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  copyFishes(in_fishes, out_fishes, num_of_fishes);
}

#ifndef AS_INCLUDE
int main(void) {
  Fish* d_in_fish;
  Fish* d_out_fish;
  initSimulation(&d_in_fish, &d_out_fish, NUM_OF_FISHES);

  double dt = 0.01;
  double time = 0;
  double max_time = 100;
  
  printFrame(d_in_fish, NUM_OF_FISHES, time);

  while(time < max_time){
    advance(d_in_fish, d_out_fish, NUM_OF_FISHES, dt);
    time += dt;
    printFrame(d_in_fish, NUM_OF_FISHES, time);
  }

  // Free device memory
  freeFishes(d_in_fish);
  freeFishes(d_out_fish);

  return 0;
}
#endif