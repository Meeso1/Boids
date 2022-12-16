#define DEBUG_MSG 1

#include "bitonicSort.cu"
#include "utils.cu"
#include "testUtils.cu"

void small_test(){
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
}

void big_test(){
	size_t length = 10000;
	int* keys   = (int*)malloc(length*sizeof(int));
	int* values = (int*)malloc(length*sizeof(int));
	for(int i = 0; i < length; i++){
		int new_val = -1;
		do{
			new_val = rand() % (length*length);
			for(int j = 0; j < i; j++){
				if(keys[j] == new_val){
					new_val = -1;
					break;
				}
			}
		}while(new_val == -1);
		keys[i] = new_val;
		values[i] = i;
	}

	bitonic_sort_pairs(keys, values, length, 1<<20, false);

	int res = 0;
	int* keys_exp   = (int*)malloc(length*sizeof(int));
	int* values_exp = (int*)malloc(length*sizeof(int));
	for(int i = 0; i < length; i++){
		keys_exp[i] = keys[i];
		values_exp[i] = values[i];
	}
	for(int i = 0; i < length; i++){
		int min_key = length*length;
		int min_id = -1;
		for(int j = i; j < length; j++){
			if(keys_exp[j] < min_key){
				min_key = keys_exp[j];
				min_id = j;
			}
		}
		int tmp_key = keys_exp[i];
		int tmp_val = values_exp[i];
		keys_exp[i] = keys_exp[min_id];
		values_exp[i] = values_exp[min_id];
		keys_exp[min_id] = tmp_key;
		values_exp[min_id] = tmp_val;
	}

	res += int_array_should_equal(keys, keys_exp, length);
	res += int_array_should_equal(values, values_exp, length);

	printf(res == 0 ? "PASSED\n" : "FAILED\n");
}

int main(){
	small_test();
	big_test();
	return 0;
}