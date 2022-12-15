#define DEBUG_MSG 1
#define USE_3D
#include "grid.cu"
#include "testUtils.cu"

void test_3d(){
	int grid_size = 100;
	int cell_size = 10;
	size_t length = 9;
	double x[] = {-1, 5,  5,  15, 15, 25, 45, 75, 85};
	double y[] = { 5, 5,  15, 5,  15, 25, 55, 85, 85};
	double z[] = { 4, 13, 4,  4,  24, 17, 55, 22, 23};
				//-1,100, 10, 1, 211,122,554,287,288
	int res = 0;

	int resolution = getGridResolution(grid_size, cell_size);
	res += int_should_equal(resolution, 10);
	int test_index_1 = getCellIndexFrom2dIndexes(2, 3, 4, resolution);
	res += int_should_equal(test_index_1, 432);
	int test_index_2 = getCellIndex(14, 41, 23, grid_size, cell_size);
	res += int_should_equal(test_index_2, 241);

	double* d_x = NULL;
	double* d_y = NULL;
	double* d_z = NULL;
	deviceMalloc((void**)&d_x, length*sizeof(double)); 
	deviceMalloc((void**)&d_y, length*sizeof(double)); 
	deviceMalloc((void**)&d_z, length*sizeof(double)); 
	deviceCopy(d_x, x, length*sizeof(double), cudaMemcpyHostToDevice);
	deviceCopy(d_y, y, length*sizeof(double), cudaMemcpyHostToDevice);
	deviceCopy(d_z, z, length*sizeof(double), cudaMemcpyHostToDevice);

	Grid grid = copyToHost(makeGrid(grid_size, cell_size, length, d_x, d_y, d_z));

	int ids_exp[] = {0, 3, 2, 1, 5, 4, 7, 8, 6};
	int cells_exp[] = {-1, 1, 10, 100, 122, 211, 287, 288, 554};

	int* starts_exp = (int*)malloc(1000*sizeof(int));
	for(int i = 0; i < 1000; i++){
		starts_exp[i] = -1;
	}
	for(int i = 0; i < length; i++){
		if(cells_exp[i] == -1) continue;
		if(starts_exp[cells_exp[i]] == -1) starts_exp[cells_exp[i]] = i;
	}

	res += int_should_equal(grid.numOfIndexes, length);
	res += int_should_equal(grid.numOfCells, 1000);
	res += int_array_should_equal(grid.indexes.indexes, ids_exp, grid.numOfIndexes);
	res += int_array_should_equal(grid.indexes.cells, cells_exp, grid.numOfIndexes);
	res += int_array_should_equal(grid.cellStarts, starts_exp, grid.numOfCells);

	int index = grid.cellStarts[10];
	while(index < grid.numOfIndexes && grid.indexes.cells[index] == 10){
		index++;
	}
	res += int_should_equal(index, grid.cellStarts[10] + 1);
	
	int* neighbour_cells = getNeighbourCellsIndexesHost(getCellIndex(x[1], y[1], z[1], grid_size, cell_size), resolution);
	int count = 0;
	for(int k = 0; k < NUM_OF_NEIGHBOURS; k++){
		int cell = neighbour_cells[k];
		if(cell == -1) {
			continue; // No such cell
		}
		int index = grid.cellStarts[cell];
		if(index == -1){
			continue; // Cell is empty
		} 
		while(index < grid.numOfIndexes && grid.indexes.cells[index] == cell){
			count++;
			index++;
		}
	}
	res += int_should_equal(count, 4);

	printf(res == 0 ? "PASSED\n" : "FAILED\n");
}

int main(){
	test_3d();
	return 0;
}