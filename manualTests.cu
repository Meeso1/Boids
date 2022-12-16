#include "bitonicSort.cu"

int main(){
	size_t length = 100;
	int* keys   = (int*)malloc(length*sizeof(int));
	int* values = (int*)malloc(length*sizeof(int));
	for(int i = 0; i < length; i++){
		keys[i] = i;
		values[i] = i;
	}

	int* d_keys = NULL;
	deviceMalloc((void**)&d_keys, length*sizeof(int));
	deviceCopy(d_keys, keys, length*sizeof(int), cudaMemcpyHostToDevice);
	int* d_vals = NULL;
	deviceMalloc((void**)&d_vals, length*sizeof(int));
	deviceCopy(d_vals, values, length*sizeof(int), cudaMemcpyHostToDevice);

	for(int i = 0; i < 20; i++){
		key_val_buffer buf1 = create_pairs_buffer(d_keys, d_vals, length, next_pow_2(length), -1, -1, true);
		deviceCopy(keys, d_keys, length*sizeof(int), cudaMemcpyDeviceToHost);
		deviceCopy(values, d_vals, length*sizeof(int), cudaMemcpyDeviceToHost);
		deviceFree(buf1.keys);
		deviceFree(buf1.values);
	}
	printf("DONE\n");

	bitonic_sort_pairs(d_keys, d_vals, length, length, true);
	bitonic_sort_pairs(d_keys, d_vals, length, length, true);
	bitonic_sort_pairs(d_keys, d_vals, length, length, true);
	printf("Done 2\n");

	bitonic_sort_pairs(keys, values, length, length, false);
	bitonic_sort_pairs(keys, values, length, length, false);
	bitonic_sort_pairs(keys, values, length, length, false);
	printf("Done 3\n");
	return 0;
}