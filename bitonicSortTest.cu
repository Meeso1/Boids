#define DEBUG_MSG 1

#include "bitonicSort.cu"
#include "utils.cu"
#include "testUtils.cu"

int main(){
	size_t length = 11;
	int keys[]   = {11, 23, 4,  43, 8,  90, 2,  5,  6,  7,  10};
	int values[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11};

	bitonic_sort_pairs(keys, values, length, 1<<20, false);

	int res = 0;
	int keys_exp[]   = {2, 4, 5, 6, 7,  8, 10, 11, 23, 43, 90};
	int values_exp[] = {7, 3, 8, 9, 10, 5, 11, 1,  2,  4,  6,};
	res += int_array_should_equal(keys, keys_exp, length);
	res += int_array_should_equal(values, values_exp, length);

	printf(res == 0 ? "PASSED\n" : "FAILED\n");
	return 0;
}