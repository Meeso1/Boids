#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include "cudaUtils.cu"
#include "bitonicSort.cu"

#define GRID_2D

struct IndexesWithCells{
	int* indexes;
	int* cells;
};

struct Grid{
	IndexesWithCells indexes;
	int* cellStarts;
	size_t numOfCells;
	size_t numOfIndexes;
	size_t resolution;
	int gridSize;
	double cellSize;
};

__device__ __host__ size_t getGridResolution(int scene_size, double cell_size){
	double res = scene_size / cell_size;
	return res == (size_t)res ? (size_t)res : (size_t)res + 1;
}

__device__ __host__ size_t numOfCells(size_t res){
	#ifdef GRID_3D
		return res * res * res;
	#else
		return res * res;
	#endif
}

__device__ __host__ int getCellIndexFrom2dIndexes(int x_i, int y_i, size_t res){
	if(x_i < 0 || x_i >= res || y_i < 0 || y_i >= res) return -1;
	return y_i * res + x_i;
}

__device__ __host__ int getCellIndex(double x, double y, int scene_size, double cell_size){
	if(x < 0 || x > scene_size || y < 0 || y > scene_size) return -1;

	int x_i = (int)(x / cell_size);
	int y_i = (int)(y / cell_size);

	int res = getGridResolution(scene_size, cell_size);

	return getCellIndexFrom2dIndexes(x_i, y_i, res);
}

__host__ int* getNeighbourCellsIndexesHost(int cell_index, size_t res){
	int num_of_neighbours = 9;
	int* neighbours = (int*)malloc(num_of_neighbours * sizeof(int));

	int x_id = cell_index % res;
	int y_id = cell_index / res;

	for(int x_off = -1; x_off <= 1; x_off++){
		for(int y_off = -1; y_off <= 1; y_off++){
			neighbours[(x_off + 1) * 3 + (y_off + 1)] = getCellIndexFrom2dIndexes(x_id + x_off, y_id + y_off, res);
		}
	}

	return neighbours;
}

__device__ void getNeighbourCellsIndexesDevice(int cell_index, size_t res, int* out /*must have length = 9*/){
	int x_id = cell_index % res;
	int y_id = cell_index / res;

	for(int x_off = -1; x_off <= 1; x_off++){
		for(int y_off = -1; y_off <= 1; y_off++){
			out[(x_off + 1) * 3 + (y_off + 1)] = getCellIndexFrom2dIndexes(x_id + x_off, y_id + y_off, res);
		}
	}
}

__global__ void assignCells(double* x, double* y, int scene_size, double cell_size, size_t length, int* indexes, int* cells){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i >= length){
		return;
	}

	indexes[i] = i;
	cells[i] = getCellIndex(x[i], y[i], scene_size, cell_size);
}

__global__ void fillStarts(int* array, size_t length){
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if(i >= length){
		return;
	}

	array[i] = -1;
}

__global__ void findCellStarts(int* sortedCells, int* starts, size_t length){
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i >= length){
		return;
	}

	if(i == 0){
		if(sortedCells[0] == 0){
			starts[0] = 0;
		}
		return;
	}

	if(sortedCells[i - 1] != sortedCells[i]){
		starts[sortedCells[i]] = i;
	}
}

// NOTE: Points not in [0, scene_size] x [0, scene_size] will not be included in the grid and will be treated as if they have no neighbours
Grid makeGrid(int scene_size, double cell_size, size_t num_of_points, double* x, double* y){
	int res = getGridResolution(scene_size, cell_size);
	DEBUG("Grid resolution: %d\n", res);

	IndexesWithCells assignedCells;
	deviceMalloc((void**)&assignedCells.indexes, num_of_points * sizeof(int));
	deviceMalloc((void**)&assignedCells.cells, num_of_points * sizeof(int));
	DEBUG("Assigning cells...\n");
	assignCells<<<1, num_of_points>>>(x, y, scene_size, cell_size, num_of_points, assignedCells.indexes, assignedCells.cells);
	DEBUG("Cells assigned. result:\n");
	IN_DEBUG(print_dev_int_array(assignedCells.indexes, num_of_points));
	IN_DEBUG(print_dev_int_array(assignedCells.cells, num_of_points));

	bitonic_sort_pairs(assignedCells.cells, assignedCells.indexes, num_of_points, numOfCells(res), true); // numOfCells is bigger than every cell index
	DEBUG("Cells sorted. result:\n");
	IN_DEBUG(print_dev_int_array(assignedCells.indexes, num_of_points));
	IN_DEBUG(print_dev_int_array(assignedCells.cells, num_of_points));

	int* cellStarts = NULL;
	deviceMalloc((void**)&cellStarts, numOfCells(res) * sizeof(int));
	DEBUG("%u cells created\n", (unsigned int)numOfCells(res));
	fillStarts<<<1, numOfCells(res)>>>(cellStarts, numOfCells(res));
	DEBUG("Cells filled\n");
	findCellStarts<<<1, num_of_points>>>(assignedCells.cells, cellStarts, num_of_points);
	DEBUG("Cell starts found. result:\n");
	IN_DEBUG(print_dev_int_array(cellStarts, numOfCells(res)));

	Grid grid = {
		assignedCells, 
		cellStarts, 
		numOfCells(res), 
		num_of_points,
		res,
		scene_size,
		cell_size
	};
	return grid;
}

Grid copyToHost(Grid src){
	Grid res;

	res.indexes.indexes = (int*)malloc(src.numOfIndexes * sizeof(int));
	deviceCopy(res.indexes.indexes, src.indexes.indexes, src.numOfIndexes * sizeof(int), cudaMemcpyDeviceToHost);

	res.indexes.cells = (int*)malloc(src.numOfIndexes * sizeof(int));
	deviceCopy(res.indexes.cells, src.indexes.cells, src.numOfIndexes * sizeof(int), cudaMemcpyDeviceToHost);

	res.cellStarts = (int*)malloc(src.numOfCells * sizeof(int));
	deviceCopy(res.cellStarts, src.cellStarts, src.numOfCells * sizeof(int), cudaMemcpyDeviceToHost);

	res.numOfCells = src.numOfCells;
	res.numOfIndexes = src.numOfIndexes;
	res.resolution = src.resolution;
	res.gridSize = src.gridSize;
	res.cellSize = src.cellSize;

	return res;
}
