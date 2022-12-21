#define DEBUG_MSG 10

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// includes, cuda
#include <cuda_runtime.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <vector_types.h>

#define MAX_THREADS_PER_BLOCK 1024
#define AS_INCLUDE
#include "boids.cu"

#define T(s) POINT(1, s)

// vbo variables
float4* cuda_vbo_resource;

float sim_time = 0.0;

#define NUM_OF_BOIDS 10000
#define DT 0.01
#define MAX_T 1
Fish* in_fishes;
Fish* out_fishes;
int* neighbour_cell_buffer;

#define BOID_SIZE 0.02

#define MAX(a,b) ((a > b) ? a : b)

bool runSimulation(int argc, char** argv);
void cleanup();

// GL functionality
void createVBO(float4** vbo_res);
void deleteVBO(float4** vbo_res);

void display();

// Cuda functionality
void copyFishesToVbo(float4** vbo_resource);

__global__ void copy_to_vbo(float4* pos, Fish fishes, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= numElements){
        return;
    }

    // scale to [-1, 1]
    float u = (float)fishes.x[i] / SCENE_SIZE;
    u = u*2.0f - 1.0f;
    float v = (float)fishes.y[i] / SCENE_SIZE;
    v = v*2.0f - 1.0f;
    float w = 0;
    #ifdef USE_3D
    w = (float)fishes.z[i] / SCENE_SIZE;
    w = w*2.0f - 1.0f;
    #endif

    #ifdef USE_3D
    float len_v = (float)length(fishes.vx[i], fishes.vy[i], fishes.vz[i]);
    #else
    float len_v = (float)length(fishes.vx[i], fishes.vy[i]);
    #endif

    float dx = fishes.vx[i] / len_v * BOID_SIZE;
    float dy = fishes.vy[i] / len_v * BOID_SIZE;
    float dz = 0;
    #ifdef USE_3D
    dz = fishes.vz[i] / len_v * BOID_SIZE;
    #endif

    // write output vertices
    if(dx * dy == 0){
        pos[3*i]     = make_float4(u + dx, v + dy, w + dz, 1.0f);
        pos[3*i + 1] = make_float4(u - BOID_SIZE/3, v, w, 1.0f);
        pos[3*i + 2] = make_float4(u + BOID_SIZE/3, v, w, 1.0f);
    }
    else{
        pos[3*i]     = make_float4(u + dx, v + dy, w + dz, 1.0f);
        pos[3*i + 1] = make_float4(u - dy/3, v + dx/3, w, 1.0f);
        pos[3*i + 2] = make_float4(u + dy/3, v - dx/3, w, 1.0f);
    }
}

int main(int argc, char** argv)
{
    printf("starting simulation...\n");

    runSimulation(argc, argv);

    printf("simulation completed\n");
    exit(EXIT_SUCCESS);
}

bool runSimulation(int argc, char** argv)
{
    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    int devID = findCudaDevice(argc, (const char **)argv);

    T("createVBO()");
    createVBO(&cuda_vbo_resource);

    initSimulation(&in_fishes, &out_fishes, &neighbour_cell_buffer, NUM_OF_BOIDS);

    // run the cuda part
    copyFishesToVbo(&cuda_vbo_resource);

    // start rendering mainloop
    while(sim_time < MAX_T) display();

    return true;
}

void copyFishesToVbo(float4** vbo_resource)
{
    // map OpenGL buffer object for writing from CUDA
    float4* dptr = *vbo_resource;
	kernelConfig kernel_size = calculateKernelConfig(NUM_OF_BOIDS, MAX_THREADS_PER_BLOCK);
    copy_to_vbo<<<kernel_size.blocks, kernel_size.threads>>>(dptr, *in_fishes, NUM_OF_BOIDS);
	
    deviceCheckErrors("copy_to_vbo");
}

void createVBO(float4** vbo_res)
{
    // initialize buffer object
    size_t size = NUM_OF_BOIDS * 4 * sizeof(float) * 3; // triangle per boid
    deviceMalloc((void**)vbo_res, size);
}

void deleteVBO(float4** vbo_res)
{
    deviceFree(*vbo_res);
}

// Display callback
void display()
{
    T("display()");

    // run CUDA kernel to generate vertex positions
    advance(in_fishes, out_fishes, NUM_OF_BOIDS, neighbour_cell_buffer, DT);

    T("copyFishesToVbo()");
    copyFishesToVbo(&cuda_vbo_resource);

    // Advance time
    sim_time += DT;

    T("display() finished");
	DEBUG("# t = %f\n\n", sim_time);
}

void cleanup()
{
    deleteVBO(&cuda_vbo_resource);

    freeFishes(in_fishes);
    freeFishes(out_fishes);
    deviceFree(neighbour_cell_buffer);
}
