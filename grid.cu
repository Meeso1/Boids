#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include "cudaUtils.cu"

struct IndexWithCell{
	int index;
	int cell;
};

struct Grid{
	IndexWithCell* indexes;
	int* cellStarts;
	int numOfCells;
};

__device__ __host__ int numOfCells(int scene_size, double cell_size){

}

__device__ __host__ int getCellIndex(double x, double y, int scene_size, double cell_size){

}

__global__ void assignCells(double* x, double* y, int scene_size, double cell_size, IndexWithCell* out){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	out[i] = {i, getCellIndex(x[i], y[i], scene_size, cell_size)};
}

__global__ void findCellStarts(IndexWithCell* sortedCells, int* starts){
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i == 0){
		starts[0] = 0;
		return;
	}

	if(sortedCells[i - 1].cell != sortedCells[i].cell){
		starts[sortedCells[i].cell] = i;
	}
}

Grid makeGrid(int scene_size, double cell_size, int num_of_points, double* x, double* y){
	IndexWithCell* assignedCells = NULL;
	deviceMalloc(&assignedCells, num_of_points);

	assignCells<<<1, num_of_points>>>(x, y, scene_size, cell_size, assignedCells);

	// TODO: sort assignedCells

	int* cellStarts = NULL;
	deviceMalloc(&cellStarts, numOfCells(scene_size, cell_size));

	findCellStarts<<<1, num_of_points>>>(assignedCells, cellStarts);

	Grid grid = {assignedCells, cellStarts, numOfCells(scene_size, cell_size)};
	return grid;
}
