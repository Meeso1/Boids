#include "bitonicSort.cu"
#include "utils.cu"

int main(){
	size_t length = 11;
	int keys[]   = {10, 23, 5,  43, 7,  90, 2,  5,  6,  6,  10};
	int values[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11};

	printf("INPUT:\n");
	print_int_array(keys, length);
	print_int_array(values, length);

	bitonic_sort_pairs(keys, values, length, 1<<20, false);

	printf("OUTPUT:\n");
	print_int_array(keys, length);
	print_int_array(values, length);

	return 0;
}