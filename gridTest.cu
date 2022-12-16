#define DEBUG_MSG 1
#include "grid.cu"
#include "testUtils.cu"

void manual_test(){
	int grid_size = 50;
	int cell_size = 5;
	size_t length = 34;
	double x[] = {0,  0,  1,  1,  2,  2,  3,  3, 5,  5,  7,  7, 11, 11, 13, 13, 17, 17, 19, 19, 23, 23, 29, 29, 31, 31, 37, 37, 41, 41, 43, 43, 47, 47};
	double y[] = {50, 37, 24, 11, 48, 35, 22, 9, 46, 33, 20, 7, 44, 31, 18, 5,  42, 29, 16, 3,  40, 27, 14, 1,  38, 25, 12, 49, 36, 23, 10, 47, 34, 21};

	printf("index from [2, 3]: %d\n", getCellIndexFrom2dIndexes(2, 3, getGridResolution(grid_size, cell_size)));
	printf("index of (49, 40): %d\n\n", getCellIndex(49, 40, grid_size, cell_size));

	double* d_x = NULL;
	double* d_y = NULL;
	deviceMalloc((void**)&d_x, length*sizeof(double)); 
	deviceMalloc((void**)&d_y, length*sizeof(double)); 
	deviceCopy(d_x, x, length*sizeof(double), cudaMemcpyHostToDevice);
	deviceCopy(d_y, y, length*sizeof(double), cudaMemcpyHostToDevice);

	printf("Creating grid...\n");
	Grid d_grid = makeGrid(grid_size, cell_size, length, d_x, d_y);
	printf("Grid created\n");
	Grid grid = copyToHost(d_grid);
	printf("Grid copied\n");

	printf("RAW GRID DATA:\n");
	printf("%zu indexes, %zu cells\n", grid.numOfIndexes, grid.numOfCells);
	printf("IDS:    ");
	print_int_array(grid.indexes.indexes, grid.numOfIndexes);
	printf("CELLS:  ");
	print_int_array(grid.indexes.cells, grid.numOfIndexes);
	printf("STARTS: ");
	print_int_array(grid.cellStarts, grid.numOfCells);

	printf("CELLS:\n");
	for(int cell = 0; cell < grid.numOfCells; cell++){
		int index = grid.cellStarts[cell];
		if(index == -1){
			// Cell is empty
			continue;
		}
		
		printf("{%3d}: ", cell);
		while(index < grid.numOfIndexes && grid.indexes.cells[index] == cell){
			int p = grid.indexes.indexes[index];
			printf("%3d(%4.1f, %4.1f) ", p, x[p], y[p]);
			index++;
		}
		printf("\n");
	}

	printf("NEIGHBOURS:\n");
	for(int i = 0; i < length; i++){
		int* neighbour_cells = getNeighbourCellsIndexesHost(getCellIndex(x[i], y[i], grid_size, cell_size), getGridResolution(grid_size, cell_size));
		printf("[%3d](%4.1f, %4.1f): ", i, x[i], y[i]);
		for(int k = 0; k < 9; k++){
			int cell = neighbour_cells[k];
			if(cell == -1) {
				continue; // No such cell
			}
			if(grid.cellStarts[cell] == -1){
				continue; // Cell is empty
			} 
			int index = grid.cellStarts[cell];
			while(index < grid.numOfIndexes && grid.indexes.cells[index] == cell){
				printf("%3d " , grid.indexes.indexes[index]);
				index++;
			}
		}
		printf("\n");
	}
}

void test_2d(){
	int grid_size = 100;
	int cell_size = 10;
	size_t length = 9;
	double x[] = {-1, 5, 5,  15, 15, 25, 45, 75, 85};
	double y[] = { 5, 5, 15, 5,  15, 25, 55, 85, 85};
	int res = 0;

	int resolution = getGridResolution(grid_size, cell_size);
	res += int_should_equal(resolution, 10);
	int test_index_1 = getCellIndexFrom2dIndexes(2, 3, resolution);
	res += int_should_equal(test_index_1, 32);
	int test_index_2 = getCellIndex(14, 41, grid_size, cell_size);
	res += int_should_equal(test_index_2, 41);

	double* d_x = NULL;
	double* d_y = NULL;
	deviceMalloc((void**)&d_x, length*sizeof(double)); 
	deviceMalloc((void**)&d_y, length*sizeof(double)); 
	deviceCopy(d_x, x, length*sizeof(double), cudaMemcpyHostToDevice);
	deviceCopy(d_y, y, length*sizeof(double), cudaMemcpyHostToDevice);

	Grid d_grid = makeGrid(grid_size, cell_size, length, d_x, d_y);
	Grid grid = copyToHost(d_grid);

	int ids_exp[] = {0, 1, 3, 2, 4, 5, 6, 7, 8};
	int cells_exp[] = {-1, 0, 1, 10, 11, 22, 54, 87, 88};

	int* starts_exp = (int*)malloc(100*sizeof(int));
	for(int i = 0; i < 100; i++){
		starts_exp[i] = -1;
	}
	for(int i = 0; i < length; i++){
		if(cells_exp[i] == -1) continue;
		if(starts_exp[cells_exp[i]] == -1) starts_exp[cells_exp[i]] = i;
	}

	res += int_should_equal(grid.numOfIndexes, length);
	res += int_should_equal(grid.numOfCells, 100);
	res += int_array_should_equal(grid.indexes.indexes, ids_exp, grid.numOfIndexes);
	res += int_array_should_equal(grid.indexes.cells, cells_exp, grid.numOfIndexes);
	res += int_array_should_equal(grid.cellStarts, starts_exp, grid.numOfCells);

	int index = grid.cellStarts[10];
	while(index < grid.numOfIndexes && grid.indexes.cells[index] == 10){
		index++;
	}
	res += int_should_equal(index, grid.cellStarts[10] + 1);

	int* neighbour_cells = getNeighbourCellsIndexesHost(getCellIndex(x[1], y[1], grid_size, cell_size), resolution);
	int count = 0;
	for(int k = 0; k < 9; k++){
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
	freeGrid(d_grid);

	printf(res == 0 ? "PASSED\n" : "FAILED\n");
}

int main(){
	test_2d();
	return 0;
}