#include "grid.cu"

int main(){
	size_t length = 34;
	double x[] = {0,  0,  1,  1,  2,  2,  3,  3, 5,  5,  7,  7, 11, 11, 13, 13, 17, 17, 19, 19, 23, 23, 29, 29, 31, 31, 37, 37, 41, 41, 43, 43, 47, 47};
	double y[] = {50, 37, 24, 11, 48, 35, 22, 9, 46, 33, 20, 7, 44, 31, 18, 5,  42, 29, 16, 3,  40, 27, 14, 1,  38, 25, 12, 49, 36, 23, 10, 47, 34, 21};

	printf("index from [2, 3]: %d\n", getCellIndexFrom2dIndexes(2, 3, getGridResolution(50, 5)));
	printf("index of (49, 40): %d\n\n", getCellIndex(49, 40, 50, 5));

	double* d_x = NULL;
	double* d_y = NULL;
	deviceMalloc((void**)&d_x, length*sizeof(double)); 
	deviceMalloc((void**)&d_y, length*sizeof(double)); 
	deviceCopy(d_x, x, length*sizeof(double), cudaMemcpyHostToDevice);
	deviceCopy(d_y, y, length*sizeof(double), cudaMemcpyHostToDevice);

	printf("Creating grid...\n");
	Grid d_grid = makeGrid(50, 5, length, d_x, d_y);
	printf("Grid created\n");
	Grid grid = copyToHost(d_grid);
	printf("Grid copied\n");
	for(int i = 0; i < length; i++){
		int* neighbour_cells = getNeighbourCellsIndexesHost(getCellIndex(x[i], y[i], 50, 10), getGridResolution(50, 5));
		printf("[%3d](%4.1f, %4.1f): ", i, x[i], y[i]);
		for(int k = 0; k < 8; k++){
			if(neighbour_cells[k] == -1) {
				continue; // No such cell
			}
			if(grid.cellStarts[neighbour_cells[k]] == -1){
				continue; // Cell is empty
			} 
			int index = grid.cellStarts[neighbour_cells[k]];
			int end = neighbour_cells[k] == grid.numOfCells-1 ? length : grid.cellStarts[neighbour_cells[k] + 1];
			while(index < end){
				printf("%3d " , grid.indexes.indexes[index]);
				index++;
			}
		}
		printf("\n");
	}
}