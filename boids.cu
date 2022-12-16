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
  Vector center;
  int attractionNeighbours;
  Vector separation;
  Vector vSum;
  int alignmentNeighbours;
};

#ifdef USE_3D
__host__ __device__ double length(double x, double y, double z){
  return sqrt(x*x + y*y + z*z);
}
#else
__host__ __device__ double length(double x, double y){
  return sqrt(x*x + y*y);
}
#endif

#ifdef USE_3D
__host__ __device__ void clip_speed(double min, double max, double* vx, double* vy, double* vz){
  double len = length(*vx, *vy, *vz);
  
  if(len <= 0){
    return;
  }

  if(len > max){
    *vx = *vx / len * max;
    *vy = *vy / len * max;
    *vz = *vz / len * max;
  }
  else if(len < min){
    *vx = *vx / len * min;
    *vy = *vy / len * min;
    *vz = *vz / len * min;
  }
}
#else
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
#endif

#ifdef USE_3D
__device__ bool is_in_front(Vector fish, Vector other, Vector v, double cos_fov){
  Vector r = {other.x - fish.x, other.y - fish.y, other.z - fish.z};
  double r_length = length(r.x, r.y, r.z);
  double v_length = length(v.x, v.y, v.z);
  if(r_length == 0 || v_length == 0) return true;
  double cos_r_v = (r.x * v.x + r.y * v.y + r.z * v.z) / (r_length * v_length);
  return cos_r_v >= cos_fov;
}
#else
__device__ bool is_in_front(Vector fish, Vector other, Vector v, double cos_fov){
  Vector r = {other.x - fish.x, other.y - fish.y};
  double r_length = length(r.x, r.y);
  double v_length = length(v.x, v.y);
  if(r_length == 0 || v_length == 0) return true;
  double cos_r_v = (r.x * v.x + r.y * v.y) / (r_length * v_length);
  return cos_r_v >= cos_fov;
}
#endif

__device__ Vector get_position(const Fish fish, int i){
  return {
    fish.x[i],
    fish.y[i],
  #ifdef USE_3D
    fish.z[i]
  #endif
  };
}

__device__ Vector get_v(const Fish fish, int i){
  return {
    fish.vx[i],
    fish.vy[i],
  #ifdef USE_3D
    fish.vz[i]
  #endif
  };
}

__device__ void updateAttraction(const Fish fish, int current, int other, double distance, FishUpdateData* data){
  if(distance > INTERACTION1_RADIUS) return;
  if(!is_in_front(
    get_position(fish, current), 
    get_position(fish, other), 
    get_v(fish, current),
    COS_FOV)
  ) return;
  #ifdef TYPES
  if(fish.type[current] != fish.type[other]) return;
  #endif

  data->center.x += fish.x[other];
  data->center.y += fish.y[other];
  #ifdef USE_3D
  data->center.z += fish.z[other];
  #endif
  data->attractionNeighbours++;
}

__device__ void updateSeparation(const Fish fish, int current, int other, double distance, FishUpdateData* data){
  if(distance > INTERACTION2_RADIUS) return;
  if(!is_in_front(
    get_position(fish, current), 
    get_position(fish, other), 
    get_v(fish, current), 
    COS_FOV)
  ) return;

  data->separation.x -= fish.x[other] - fish.x[current];
  data->separation.y -= fish.y[other] - fish.y[current];
  #ifdef USE_3D
  data->separation.z -= fish.z[other] - fish.z[current];
  #endif
}

__device__ void updateAlignment(const Fish fish, int current, int other, double distance, FishUpdateData* data){
  if(distance > INTERACTION3_RADIUS) return;
  if(!is_in_front(
    get_position(fish, current), 
    get_position(fish, other), 
    get_v(fish, current),
    COS_FOV)
  ) return;
  #ifdef TYPES
  if(fish.type[current] != fish.type[other]) return;
  #endif

  data->vSum.x += fish.vx[other];
  data->vSum.y += fish.vy[other];
  #ifdef USE_3D
  data->vSum.z += fish.vz[other];
  #endif
  data->alignmentNeighbours++;
}

__device__ Vector getAvoidanceDv(const Fish fish, int current){
  Vector dv = VECT_0;
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

  #ifdef USE_3D
  if(fish.z[current] < INTERACTION4_RADIUS){
    dv.z = (INTERACTION4_RADIUS - fish.z[current]) * REPULSION_STR;
  }
  else if(fish.z[current] > SCENE_SIZE - INTERACTION4_RADIUS){
    dv.z = - (fish.z[current] - SCENE_SIZE + INTERACTION4_RADIUS) * REPULSION_STR;
  }
  #endif

  return dv;
}

__device__ Vector get_dv(const Fish fish, int current, FishUpdateData* data){
  Vector v1 = VECT_0;
  if(data->attractionNeighbours != 0){
    v1.x = (data->center.x / data->attractionNeighbours - fish.x[current]) * ATTRACTION_STR;
    v1.y = (data->center.y / data->attractionNeighbours - fish.y[current]) * ATTRACTION_STR;
  #ifdef USE_3D
    v1.z = (data->center.z / data->attractionNeighbours - fish.z[current]) * ATTRACTION_STR;
  #endif
  }

  Vector v2 = {
    data->separation.x * SEPARATION_STR, 
    data->separation.y * SEPARATION_STR,
  #ifdef USE_3D
    data->separation.z * SEPARATION_STR,
  #endif
  };

  Vector v3 = VECT_0;
  if(data->alignmentNeighbours != 0){
    v3 = {
      (data->vSum.x / data->alignmentNeighbours - fish.vx[current]) * ALIGNMENT_STR, 
      (data->vSum.y / data->alignmentNeighbours - fish.vy[current]) * ALIGNMENT_STR,
    #ifdef USE_3D
      (data->vSum.z / data->alignmentNeighbours - fish.vz[current]) * ALIGNMENT_STR
    #endif
    };
  }

  Vector v4 = getAvoidanceDv(fish, current);

  return {
    v1.x + v2.x + v3.x + v4.x, 
    v1.y + v2.y + v3.y + v4.y,
  #ifdef USE_3D
    v1.z + v2.z + v3.z + v4.z
  #endif
  };
}

__global__ void updateFish(const Fish in, Fish out, Grid grid, int* neighbour_cells, double dt, int numElements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i >= numElements){
    return;
  }

  FishUpdateData data = {VECT_0, 0, VECT_0, VECT_0, 0};
  #ifdef USE_3D
  int cell_index = getCellIndex(in.x[i], in.y[i], in.z[i], grid.gridSize, grid.cellSize);
  #else
  int cell_index = getCellIndex(in.x[i], in.y[i], grid.gridSize, grid.cellSize);
  #endif
  
  if(cell_index != -1){
    for(int k = 0; k < NUM_OF_NEIGHBOURS; k++){
      int cell = neighbour_cells[NUM_OF_NEIGHBOURS*cell_index + k];
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
        #ifdef USE_3D
        double distance = length(in.x[i] - in.x[index], in.y[i] - in.y[index], in.z[i] - in.z[index]);
        #else
        double distance = length(in.x[i] - in.x[index], in.y[i] - in.y[index]);
        #endif
        updateAttraction(in, i, index, distance, &data);
        updateSeparation(in, i, index, distance, &data);
        updateAlignment(in, i, index, distance, &data);

        index++;
      }
    }
  }

  Vector dv = get_dv(in, i, &data);
  out.vx[i] = in.vx[i] + dv.x;
  out.vy[i] = in.vy[i] + dv.y;
  #ifdef USE_3D
  out.vz[i] = in.vz[i] + dv.z;
  #endif

  #ifdef USE_3D
  clip_speed(MIN_V, MAX_V, &(out.vx[i]), &(out.vy[i]), &(out.vz[i]));
  #else
  clip_speed(MIN_V, MAX_V, &(out.vx[i]), &(out.vy[i]));
  #endif

  out.x[i] = in.x[i] + out.vx[i] * dt;
  out.y[i] = in.y[i] + out.vy[i] * dt;
  #ifdef USE_3D
  out.z[i] = in.z[i] + out.vz[i] * dt;
  #endif
}

__global__ void fillNeighbourCellBuffer(int* buffer, int num_of_cells, size_t resolution){
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i >= num_of_cells){
    return;
  }

  getNeighbourCellsIndexesDevice(i, resolution, &(buffer[NUM_OF_NEIGHBOURS*i]));
}

Fish* initFish(int number){
  srand(1234567890);

  Fish* fish = (Fish*)malloc(sizeof(Fish));
  
  fish->x = (double*)malloc(number*sizeof(double));
  fish->vx = (double*)malloc(number*sizeof(double));
  fish->y = (double*)malloc(number*sizeof(double));
  fish->vy = (double*)malloc(number*sizeof(double));
  #ifdef USE_3D
  fish->z = (double*)malloc(number*sizeof(double));
  fish->vz = (double*)malloc(number*sizeof(double));
  #endif
  #ifdef TYPES
  fish->type = (int*)malloc(number*sizeof(int));
  #endif

  for(int i = 0; i < number; i++){
    fish->x[i]  = (rand() % (SCENE_SIZE * 100)) / 100.0;
    fish->y[i]  = (rand() % (SCENE_SIZE * 100)) / 100.0;
    #ifdef USE_3D
    fish->z[i]  = (rand() % (SCENE_SIZE * 100)) / 100.0;
    #endif

    #ifdef USE_3D
    do{
      fish->vx[i] = (rand() % (2 * MAX_V * 100)) / 100.0 - MAX_V;
      fish->vy[i] = (rand() % (2 * MAX_V * 100)) / 100.0 - MAX_V;
      fish->vz[i] = (rand() % (2 * MAX_V * 100)) / 100.0 - MAX_V;
      clip_speed(MIN_V, MAX_V, &(fish->vx[i]), &(fish->vy[i]), &(fish->vz[i]));
    }while(fish->vx[i] * fish->vy[i] * fish->vz[i] == 0);
    #else
    do{
      fish->vx[i] = (rand() % (2 * MAX_V * 100)) / 100.0 - MAX_V;
      fish->vy[i] = (rand() % (2 * MAX_V * 100)) / 100.0 - MAX_V;
      clip_speed(MIN_V, MAX_V, &(fish->vx[i]), &(fish->vy[i]));
    }while(fish->vx[i] * fish->vy[i] == 0);
    #endif

    #ifdef TYPES
    fish->type[i] = rand() % TYPES;
    #endif
  }

  return fish;
}

Fish* allocateFish(size_t num){
  size_t vector_size = num*sizeof(double);
  Fish* fish = (Fish*)malloc(sizeof(Fish));
  fish->x = NULL;
  deviceMalloc((void**) &(fish->x),  vector_size);
  fish->vx = NULL;
  deviceMalloc((void**) &(fish->vx), vector_size);
  fish->y = NULL;
  deviceMalloc((void**) &(fish->y),  vector_size);
  fish->vy = NULL;
  deviceMalloc((void**) &(fish->vy), vector_size);
  #ifdef USE_3D
  fish->z = NULL;
  deviceMalloc((void**) &(fish->z),  vector_size);
  fish->vz = NULL;
  deviceMalloc((void**) &(fish->vz), vector_size);
  #endif

  #ifdef TYPES
  fish->type = NULL;
  deviceMalloc((void**) &(fish->type), num*sizeof(int));
  #endif

  return fish;
}

void freeHostFishes(Fish* fishes){
  free(fishes->x);
  free(fishes->vx);
  free(fishes->y);
  free(fishes->vy);
  #ifdef USE_3D
  free(fishes->z);
  free(fishes->vz);
  #endif
  #ifdef TYPES
  free(fish->type);
  #endif
  free(fishes);
}

void initSimulation(Fish** device_in_fishes, Fish** device_out_fishes, int** device_neighbour_buff, int num_of_fishes){
  Fish* fish = initFish(num_of_fishes);
  size_t vector_size = num_of_fishes*sizeof(double);

  // Allocate the device input and output vectors
  Fish* d_in_fish = allocateFish(num_of_fishes);
  Fish* d_out_fish = allocateFish(num_of_fishes);

  T("deviceCopy()");

  // Copy to input vector
  deviceCopy(d_in_fish->x, fish->x, vector_size, cudaMemcpyHostToDevice);
  deviceCopy(d_in_fish->vx, fish->vx, vector_size, cudaMemcpyHostToDevice);
  deviceCopy(d_in_fish->y, fish->y, vector_size, cudaMemcpyHostToDevice);
  deviceCopy(d_in_fish->vy, fish->vy, vector_size, cudaMemcpyHostToDevice);
  #ifdef USE_3D
  deviceCopy(d_in_fish->z, fish->z, vector_size, cudaMemcpyHostToDevice);
  deviceCopy(d_in_fish->vz, fish->vz, vector_size, cudaMemcpyHostToDevice);
  #endif
  #ifdef TYPES
  deviceCopy(d_in_fish->type, fish->type, num_of_fishes*sizeof(int), cudaMemcpyHostToDevice);
  #endif

  // Allocate neighbour cell buffer
  int res = getGridResolution(SCENE_SIZE, CELL_SIZE);
  int num_of_cells = numOfCells(res);
  *device_neighbour_buff = NULL;
  deviceMalloc((void**) device_neighbour_buff, NUM_OF_NEIGHBOURS*num_of_cells*sizeof(int));

  T("fillNeighbourCellBuffer()");
  
  // Fill it with neighbour indexes, as grid is static and they won't change
  fillNeighbourCellBuffer<<<1, num_of_cells>>>(*device_neighbour_buff, num_of_cells, res);

  *device_in_fishes = d_in_fish;
  *device_out_fishes = d_out_fish;

  T("freeHostFishes()");
  freeHostFishes(fish);
}

void copyFishes(const Fish* source, Fish* destination, int num_of_fishes){
  size_t vector_size = num_of_fishes*sizeof(double);
  deviceCopy(source->x, destination->x, vector_size, cudaMemcpyDeviceToDevice);
  deviceCopy(source->vx, destination->vx, vector_size, cudaMemcpyDeviceToDevice);
  deviceCopy(source->y, destination->y, vector_size, cudaMemcpyDeviceToDevice);
  deviceCopy(source->vy, destination->vy, vector_size, cudaMemcpyDeviceToDevice);
  #ifdef USE_3D
  deviceCopy(source->z, destination->z, vector_size, cudaMemcpyDeviceToDevice);
  deviceCopy(source->vz, destination->vz, vector_size, cudaMemcpyDeviceToDevice);
  #endif
}

void freeFishes(Fish* fishes){
  deviceFree(fishes->x);
  deviceFree(fishes->vx);
  deviceFree(fishes->y);
  deviceFree(fishes->vy);
  #ifdef USE_3D
  deviceFree(fishes->z);
  deviceFree(fishes->vz);
  #endif
}

void advance(Fish* in_fishes, Fish* out_fishes, int num_of_fishes, int* neighbour_buff, double dt){
  // Make adjacency grid
  #ifdef USE_3D
  Grid grid = makeGrid(SCENE_SIZE, CELL_SIZE, num_of_fishes, in_fishes->x, in_fishes->y, in_fishes->z);
  #else
  Grid grid = makeGrid(SCENE_SIZE, CELL_SIZE, num_of_fishes, in_fishes->x, in_fishes->y);
  #endif

  T("updateFish()");

  // Launch CUDA Kernel
  updateFish<<<1, num_of_fishes>>>(*in_fishes, *out_fishes, grid, neighbour_buff, dt, num_of_fishes);
  deviceCheckErrors("updateFish");

  freeGrid(grid);

  T("copyFishes()");
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