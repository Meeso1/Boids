#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include "cudaUtils.cu"
#include "utils.cu"
#include "grid.cu"

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
#define CELL_SIZE MAX(MAX(INTERACTION1_RADIUS, INTERACTION2_RADIUS), INTERACTION3_RADIUS)

// Used to store data used in boid rules
struct FishUpdateData{
  Vector2 center;
  int attractionNeighbours;
  Vector2 separation;
  Vector2 vSum;
  int alignmentNeighbours;
};

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

__device__ void updateAttraction(const Fish fish, int current, int other, double distance, FishUpdateData* data){
  if(distance > INTERACTION1_RADIUS) return;
  if(!is_in_front(
    {fish.x[current], fish.y[current]}, 
    {fish.x[other], fish.y[other]}, 
    {fish.vx[current], fish.vy[current]}, 
    COS_FOV)
  ) return;

  data->center.x += fish.x[other];
  data->center.y += fish.y[other];
  data->attractionNeighbours++;
}

__device__ void updateSeparation(const Fish fish, int current, int other, double distance, FishUpdateData* data){
  if(distance > INTERACTION2_RADIUS) return;
  if(!is_in_front(
    {fish.x[current], fish.y[current]}, 
    {fish.x[other], fish.y[other]}, 
    {fish.vx[current], fish.vy[current]}, 
    COS_FOV)
  ) return;

  data->separation.x -= fish.x[other] - fish.x[current];
  data->separation.y -= fish.y[other] - fish.y[current];
}

__device__ void updateAlignment(const Fish fish, int current, int other, double distance, FishUpdateData* data){
  if(distance > INTERACTION3_RADIUS) return;
  if(!is_in_front(
    {fish.x[current], fish.y[current]}, 
    {fish.x[other], fish.y[other]}, 
    {fish.vx[current], fish.vy[current]}, 
    COS_FOV)
  ) return;

  data->vSum.x += fish.vx[other];
  data->vSum.y += fish.vy[other];
  data->alignmentNeighbours++;
}

__device__ Vector2 getAvoidanceDv(const Fish fish, int current){
  Vector2 dv = {0, 0};
  if(fish.x[current] < INTERACTION4_RADIUS){
    dv.x = (INTERACTION4_RADIUS - fish.x[current]) * REPULSION_STR;
  }
  else if(fish.x[current] > SCENE_SIZE - INTERACTION4_RADIUS){
    dv.x = - (fish.x[current] - SCENE_SIZE + INTERACTION4_RADIUS) * REPULSION_STR;
  }

  if(fish.y[current] < INTERACTION4_RADIUS){
    dv.y = (INTERACTION4_RADIUS - fish.y[current]) * REPULSION_STR;
  }
  else if(fish.y[current] > SCENE_SIZE - INTERACTION4_RADIUS){
    dv.y = - (fish.y[current] - SCENE_SIZE + INTERACTION4_RADIUS) * REPULSION_STR;
  }

  return dv;
}

__device__ Vector2 get_dv(const Fish fish, int current, FishUpdateData* data){
  Vector2 v1 = {0, 0};
  if(data->attractionNeighbours != 0){
    v1.x = (data->center.x / data->attractionNeighbours - fish.x[current]) * ATTRACTION_STR;
    v1.y = (data->center.y / data->attractionNeighbours - fish.y[current]) * ATTRACTION_STR;
  }

  Vector2 v2 = {data->separation.x * SEPARATION_STR, data->separation.y * SEPARATION_STR};

  Vector2 v3 = {0, 0};
  if(data->alignmentNeighbours != 0){
    v3 = {
      data->vSum.x / data->alignmentNeighbours - fish.vx[current], 
      data->vSum.y / data->alignmentNeighbours - fish.vy[current]
    };
    v3.x *= ALIGNMENT_STR;
    v3.y *= ALIGNMENT_STR;
  }

  Vector2 v4 = getAvoidanceDv(fish, current);

  return {v1.x + v2.x + v3.x + v4.x, v1.y + v2.y + v3.y + v4.y};
}

__global__ void updateFish(const Fish in, Fish out, Grid grid, int* neighbour_cells, double dt, int numElements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i >= numElements){
    return;
  }

  FishUpdateData data = {{0, 0}, 0, {0, 0}, {0, 0}, 0};
  int cell_index = getCellIndex(in.x[i], in.y[i], grid.gridSize, grid.cellSize);
  for(int k = 0; k < 9; k++){
    int cell = neighbour_cells[9*cell_index + k];
    if(cell == -1){
      // no such cell
      continue;
    }

    int index = grid.cellStarts[cell];
    if(index == -1){
      // cell is empty
      continue;
    }

    while(index < grid.numOfIndexes && grid.indexes.cells[index] == cell){
      if(index == i) continue;

      // process neighbour
      double distance = length(in.x[i] - in.x[index], in.y[i] - in.y[index]);
      updateAttraction(in, i, index, distance, &data);
      updateSeparation(in, i, index, distance, &data);
      updateAlignment(in, i, index, distance, &data);

      index++;
    }
  }

  Vector2 dv = get_dv(in, i, &data);
  out.vx[i] = in.vx[i] + dv.x;
  out.vy[i] = in.vy[i] + dv.y;

  clip_speed(MIN_V, MAX_V, &(out.vx[i]), &(out.vy[i]));

  out.x[i] = in.x[i] + out.vx[i] * dt;
  out.y[i] = in.y[i] + out.vy[i] * dt;
}

__global__ void fillNeighbourCellBuffer(int* buffer, int num_of_cells, size_t resolution){
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i >= num_of_cells){
    return;
  }

  getNeighbourCellsIndexesDevice(i, resolution, &(buffer[9*i]));
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

void initSimulation(Fish** device_in_fishes, Fish** device_out_fishes, int** device_neighbour_buff, int num_of_fishes){
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

  // Allocate neighbour cell buffer
  int res = getGridResolution(SCENE_SIZE, CELL_SIZE);
  int num_of_cells = numOfCells(res);
  *device_neighbour_buff = NULL;
  deviceMalloc((void**) &device_neighbour_buff, 9*num_of_cells*sizeof(int));
  // Fill it with neighbour indexes, as grid is static and they won't change
  fillNeighbourCellBuffer<<<1, num_of_cells>>>(*device_neighbour_buff, num_of_cells, res);

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

void advance(Fish* in_fishes, Fish* out_fishes, int num_of_fishes, int* neighbour_buff, double dt){
  // Make adjacency grid
  Grid grid = makeGrid(SCENE_SIZE, CELL_SIZE, num_of_fishes, in_fishes->x, in_fishes->y);

  // Launch CUDA Kernel
  updateFish<<<1, num_of_fishes>>>(*in_fishes, *out_fishes, grid, neighbour_buff, dt, num_of_fishes);

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