#include "cudaUtils.cu"
#include "utils.cu"

int int_array_should_equal(int* result, int* expectation, size_t length){
	int i;
	for(i = 0; i < length; i++){
		if(result[i] != expectation[i]){
			break;
		}
	}

	if(i < length){
		DEBUG("Arrays are not equal (difference at index %d)\n", i);
		DEBUG("Expected: ");
		IN_DEBUG(print_int_array(expectation, length));
		DEBUG("Actual:   ");
		IN_DEBUG(print_int_array(result, length));
		return 1;
	}

	return 0;
}

int dev_int_array_should_equal(int* d_result, int* expectation, size_t length){
	int* tmp = (int*)malloc(length*sizeof(int));
	deviceCopy(tmp, d_result, length*sizeof(int), cudaMemcpyDeviceToHost);
	return int_array_should_equal(tmp, expectation, length);
}

int int_should_equal(int value, int expected){
	if(value != expected){
		DEBUG("Expected %d to be equal %d\n", value, expected);
		return 1;
	}

	return 0;
}