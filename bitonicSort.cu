/*
 * Source: https://gist.github.com/mre/1392067
 *
 * Parallel bitonic sort using CUDA.
 * Compile with
 * nvcc -arch=sm_11 bitonic_sort.cu
 * Based on http://www.tools-of-computing.com/tc/CS/Sorts/bitonic_sort.htm
 * License: BSD 3
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cudaUtils.cu"

/* Every thread gets exactly one value in the unsorted array. */
#define NUM_VALS 23

__global__ void bitonic_sort_step(float* dev_values, int j, int k)
{
  unsigned int i, ixj; /* Sorting partners: i and ixj */
  i = threadIdx.x + blockDim.x * blockIdx.x;
  ixj = i^j;

  /* The threads with the lowest ids sort the array. */
  if ((ixj)>i) {
    if ((i&k)==0) {
      /* Sort ascending */
      if (dev_values[i]>dev_values[ixj]) {
      /* exchange(i,ixj); */
      float temp = dev_values[i];
      dev_values[i] = dev_values[ixj];
      dev_values[ixj] = temp;
      }
    }
    if ((i&k)!=0) {
      /* Sort descending */
      if (dev_values[i]<dev_values[ixj]) {
        /* exchange(i,ixj); */
        float temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
  }
}

__global__ void fill(float* destination, size_t start, size_t destination_length, float fill_value){
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if(i + start >= destination_length){
    return;
  }

  destination[i + start] = fill_value;
}

/**
* Inplace bitonic sort using CUDA.
*/
void bitonic_sort_pow2(float* dev_values, size_t length)
{
  dim3 blocks(length > 64 ? 64 : 1, 1);
  dim3 threads(length > 64 ? length / 64 : length, 1);

  int j, k;
  /* Major step */
  for (k = 2; k <= length; k <<= 1) {
    /* Minor step */
    for (j=k>>1; j>0; j=j>>1) {
      bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
    }
  }
}

size_t next_pow_2(size_t num){
  float log2 = log(length) / log(2);
  int exp = log2 == (int)log2 ? (int)log2 : (int)log2 + 1;
  return (size_t)pow(2, exp);
}

void bitonic_sort(float* values, size_t length, float max_val){
  size_t valid_size = next_pow_2(length);

  float* buffer = NULL;
  deviceMalloc((void**)&buffer, valid_size * sizeof(float));
  deviceCopy(buffer, values, length * sizeof(float), cudaMemcpyHostToDevice);

  size_t len_to_fill = valid_size - length;
  if(len_to_fill > 0){
    dim3 blocks(len_to_fill > 64 ? 64 : 1, 1);
    dim3 threads(len_to_fill > 64 ? len_to_fill / 64 : len_to_fill, 1);
    fill<<<blocks, threads>>>(buffer, length, valid_size, max_val);
  }

  bitonic_sort_pow2(buffer, valid_size);

  deviceCopy(values, buffer, length * sizeof(float), cudaMemcpyDeviceToHost);
  deviceFree(buffer);
}