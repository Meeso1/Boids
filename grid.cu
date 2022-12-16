#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include "cudaUtils.cu"
#include "bitonicSort.cu"

#ifdef USE_3D
#define NUM_OF_NEIGHBOURS 3*3*3
#else
#define NUM_OF_NEIGHBOURS 3*3
#endif

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
	#ifdef USE_3D
		return res * res * res;
	#else
		return res * res;
	#endif
}

#ifdef USE_3D
__device__ __host__ int getCellIndexFrom2dIndexes(int x_i, int y_i, int z_i, size_t res){
	if(x_i < 0 || x_i >= res || y_i < 0 || y_i >= res || z_i < 0 || z_i >= res) return -1;
	return z_i*res*res + y_i * res + x_i;
}
#else
__device__ __host__ int getCellIndexFrom2dIndexes(int x_i, int y_i, size_t res){
	if(x_i < 0 || x_i >= res || y_i < 0 || y_i >= res) return -1;
	return y_i * res + x_i;
}
#endif

#ifdef USE_3D
__device__ __host__ int getCellIndex(double x, double y, double z, int scene_size, double cell_size){
	if(x < 0 || x > scene_size || y < 0 || y > scene_size || z < 0 || z > scene_size) return -1;

	int x_i = (int)(x / cell_size);
	int y_i = (int)(y / cell_size);
	int z_i = (int)(z / cell_size);

	int res = getGridResolution(scene_size, cell_size);

	return getCellIndexFrom2dIndexes(x_i, y_i, z_i, res);
}
#else
__device__ __host__ int getCellIndex(double x, double y, int scene_size, double cell_size){
	if(x < 0 || x > scene_size || y < 0 || y > scene_size) return -1;

	int x_i = (int)(x / cell_size);
	int y_i = (int)(y / cell_size);

	int res = getGridResolution(scene_size, cell_size);

	return getCellIndexFrom2dIndexes(x_i, y_i, res);
}
#endif

__host__ int* getNeighbourCellsIndexesHost(int cell_index, size_t res){
	int* neighbours = (int*)malloc(NUM_OF_NEIGHBOURS * sizeof(int));

	int x_id = cell_index % res;
	int y_id = (cell_index % (res*res)) / res;
	#ifdef USE_3D
	int z_id = cell_index / (res*res);
	#endif

	for(int x_off = -1; x_off <= 1; x_off++){
		for(int y_off = -1; y_off <= 1; y_off++){
			#ifdef USE_3D
			for(int z_off = -1; z_off <= 1; z_off++){
				neighbours[(x_off + 1) * 9 + (y_off + 1) * 3 + (z_off + 1)] = 
					getCellIndexFrom2dIndexes(x_id + x_off, y_id + y_off, z_id + z_off, res);
			}
			#else
			neighbours[(x_off + 1) * 3 + (y_off + 1)] = getCellIndexFrom2dIndexes(x_id + x_off, y_id + y_off, res);
			#endif
		}
	}

	return neighbours;
}

__device__ void getNeighbourCellsIndexesDevice(int cell_index, size_t res, int* out /*must have length = NUM_OF_NEIGHBOURS*/){
	int x_id = cell_index % res;
	int y_id = (cell_index % (res*res)) / res;
	#ifdef USE_3D
	int z_id = cell_index / (res*res);
	#endif

	for(int x_off = -1; x_off <= 1; x_off++){
		for(int y_off = -1; y_off <= 1; y_off++){
			#ifdef USE_3D
			for(int z_off = -1; z_off <= 1; z_off++){
				out[(x_off + 1) * 9 + (y_off + 1) * 3 + (z_off + 1)] = 
					getCellIndexFrom2dIndexes(x_id + x_off, y_id + y_off, z_id + z_off, res);
			}
			#else
			out[(x_off + 1) * 3 + (y_off + 1)] = getCellIndexFrom2dIndexes(x_id + x_off, y_id + y_off, res);
			#endif
		}
	}
}

#ifdef USE_3D
__global__ void assignCells(double* x, double* y, double* z, int scene_size, double cell_size, size_t length, int* indexes, int* cells){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i >= length){
		return;
	}

	indexes[i] = i;
	cells[i] = getCellIndex(x[i], y[i], z[i], scene_size, cell_size);
}
#else
__global__ void assignCells(double* x, double* y, int scene_size, double cell_size, size_t length, int* indexes, int* cells){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i >= length){
		return;
	}

	indexes[i] = i;
	cells[i] = getCellIndex(x[i], y[i], scene_size, cell_size);
}
#endif

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
#ifdef USE_3D
Grid makeGrid(int scene_size, double cell_size, size_t num_of_points, double* x, double* y, double* z){
	int res = getGridResolution(scene_size, cell_size);

	IndexesWithCells assignedCells;
	deviceMalloc((void**)&assignedCells.indexes, num_of_points * sizeof(int));
	deviceMalloc((void**)&assignedCells.cells, num_of_points * sizeof(int));

	assignCells<<<1, num_of_points>>>(x, y, z, scene_size, cell_size, num_of_points, assignedCells.indexes, assignedCells.cells);

	bitonic_sort_pairs(assignedCells.cells, assignedCells.indexes, num_of_points, numOfCells(res), true); // numOfCells is bigger than every cell index

	int* cellStarts = NULL;
	deviceMalloc((void**)&cellStarts, numOfCells(res) * sizeof(int));

	fillStarts<<<1, numOfCells(res)>>>(cellStarts, numOfCells(res));
	findCellStarts<<<1, num_of_points>>>(assignedCells.cells, cellStarts, num_of_points);

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
#else
Grid makeGrid(int scene_size, double cell_size, size_t num_of_points, double* x, double* y){
	int res = getGridResolution(scene_size, cell_size);

	T("grid: deviceMalloc()");

	IndexesWithCells assignedCells;
	deviceMalloc((void**)&assignedCells.indexes, num_of_points * sizeof(int));
	deviceMalloc((void**)&assignedCells.cells, num_of_points * sizeof(int));

	T("grid: assignCells()");
	assignCells<<<1, num_of_points>>>(x, y, scene_size, cell_size, num_of_points, assignedCells.indexes, assignedCells.cells);
	deviceCheckErrors("assignCells");

	bitonic_sort_pairs(assignedCells.cells, assignedCells.indexes, num_of_points, numOfCells(res), true); // numOfCells is bigger than every cell index
	T("grid: sort finished");

	int* cellStarts = NULL;
	deviceMalloc((void**)&cellStarts, numOfCells(res) * sizeof(int));

	T("grid: fillStarts()");
	fillStarts<<<1, numOfCells(res)>>>(cellStarts, numOfCells(res));
	deviceCheckErrors("fillStarts");
	findCellStarts<<<1, num_of_points>>>(assignedCells.cells, cellStarts, num_of_points);
	deviceCheckErrors("findCellStarts");

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
#endif

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

void freeGrid(Grid grid){
	deviceFree(grid.indexes.indexes);
	deviceFree(grid.indexes.cells);
	deviceFree(grid.cellStarts);
}
