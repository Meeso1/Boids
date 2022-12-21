/*
 * Source: https://gist.github.com/mre/1392067
 *
 * Parallel bitonic sort using CUDA.
 * Based on http://www.tools-of-computing.com/tc/CS/Sorts/bitonic_sort.htm
 * License: BSD 3
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cudaUtils.cu"
#include "utils.cu"

#define T(s) POINT(4, s)

///
/// FLOAT
///
__global__ void bitonic_sort_step(float* dev_values, int j, int k)
{
  unsigned int i, ixj; /* Sorting partners: i and ixj */
  i = threadIdx.x + blockDim.x * blockIdx.x;
  ixj = i^j;

  /* The threads with the lowest ids sort the array. */
  if ((ixj)>i) {
    if ((i&k)==0) {
      /* Sort ascending */
      if (dev_values[i] - dev_values[ixj] > 0) {
        /* exchange(i,ixj); */
        float temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
    if ((i&k)!=0) {
      /* Sort descending */
      if (dev_values[i] - dev_values[ixj] < 0) {
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
  float log2 = log(num) / log(2);
  int exp = log2 == (int)log2 ? (int)log2 : (int)log2 + 1;
  return (size_t)pow(2, exp);
}

float* create_buffer(float* values, size_t values_length, size_t buffer_length, float fill_value){
  float* buffer = NULL;
  deviceMalloc((void**)&buffer, buffer_length * sizeof(float));
  deviceCopy(buffer, values, values_length * sizeof(float), cudaMemcpyHostToDevice);

  size_t len_to_fill = buffer_length - values_length;
  if(len_to_fill > 0){
    dim3 blocks(len_to_fill > 64 ? 64 : 1, 1);
    dim3 threads(len_to_fill > 64 ? len_to_fill / 64 : len_to_fill, 1);
    fill<<<blocks, threads>>>(buffer, values_length, buffer_length, fill_value);
  }

  return buffer;
}

void bitonic_sort(float* values, size_t length, float max_val){
  size_t valid_size = next_pow_2(length);
  float* buffer = create_buffer(values, length, valid_size, max_val);

  bitonic_sort_pow2(buffer, valid_size);

  deviceCopy(values, buffer, length * sizeof(float), cudaMemcpyDeviceToHost);
  deviceFree(buffer);
}

///
/// INT, INT
///
struct key_val_buffer{
  int* keys;
  int* values;
};

__global__ void bitonic_sort_pairs_step(int* keys, int* values, int j, int k, size_t length)
{
  unsigned int i, ixj; /* Sorting partners: i and ixj */
  i = threadIdx.x + blockDim.x * blockIdx.x;
  ixj = i^j;

  /* The threads with the lowest ids sort the array. */
  if ((ixj)>i) {
    if ((i&k)==0) {
      /* Sort ascending */
      if (keys[i] - keys[ixj] > 0) {
        /* exchange(i,ixj); */
        int temp_key = keys[i];
        int temp_val = values[i];
        keys[i] = keys[ixj];
        values[i] = values[ixj];
        keys[ixj] = temp_key;
        values[ixj] = temp_val;
      }
    }
    if ((i&k)!=0) {
      /* Sort descending */
      if (keys[i] - keys[ixj] < 0) {
        /* exchange(i,ixj); */
        int temp_key = keys[i];
        int temp_val = values[i];
        keys[i] = keys[ixj];
        values[i] = values[ixj];
        keys[ixj] = temp_key;
        values[ixj] = temp_val;
      }
    }
  }
}

__global__ void fill_pairs(int* destination_a, int* destination_b, size_t start, size_t destination_length, int fill_a, int fill_b){
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if(i + start >= destination_length){
    return;
  }

  destination_a[i + start] = fill_a;
  destination_b[i + start] = fill_b;
}

void bitonic_sort_pairs_pow2(int* keys, int* values, size_t length)
{
  kernelConfig kernel_size = calculateKernelConfig(length, MAX_THREADS_PER_BLOCK);

  int j, k;
  /* Major step */
  for (k = 2; k <= length; k <<= 1) {
    /* Minor step */
    for (j=k>>1; j>0; j=j>>1) {
      bitonic_sort_pairs_step<<<kernel_size.blocks, kernel_size.threads>>>(keys, values, j, k, length);
      deviceCheckErrors("bitonic_sort_pairs_step");
    }
  }
}

key_val_buffer create_pairs_buffer(int* keys, int* values, size_t pairs_length, size_t buffer_length, int fill_key, int fill_value, bool src_is_device){
  key_val_buffer buffer;

  int* buf_keys = NULL;
  deviceMalloc((void**)&buf_keys, buffer_length * sizeof(int));
  deviceCopy(buf_keys, keys, pairs_length * sizeof(int), src_is_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice);

  int* buf_vals = NULL;
  deviceMalloc((void**)&buf_vals, buffer_length * sizeof(int));
  deviceCopy(buf_vals, values, pairs_length * sizeof(int), src_is_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice);

  size_t len_to_fill = buffer_length - pairs_length;
  if(len_to_fill > 0){
    dim3 blocks(len_to_fill > 64 ? 64 : 1, 1);
    dim3 threads(len_to_fill > 64 ? len_to_fill / 64 : len_to_fill, 1);
    fill_pairs<<<blocks, threads>>>(buf_keys, buf_vals, pairs_length, buffer_length, fill_key, fill_value);
    deviceCheckErrors("fill_pairs");
  }

  buffer.keys = buf_keys;
  buffer.values = buf_vals;

  DEBUG("Allocated buffer at %p and %p\n", buffer.keys, buffer.values);
  return buffer;
}

void bitonic_sort_pairs(int* keys, int* values, size_t length, int max_key, bool src_is_device){
  size_t valid_size = next_pow_2(length);
  T("sort: fill_pairs_buffer()");
  key_val_buffer buffer = create_pairs_buffer(keys, values, length, valid_size, max_key, -1, src_is_device);

  T("sort: bitonic_sort_pairs_pow2()");
  bitonic_sort_pairs_pow2(buffer.keys, buffer.values, valid_size);

  T("sort: deviceCopy()");
  deviceCopy(keys, buffer.keys, length * sizeof(int), src_is_device ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost);
  deviceCopy(values, buffer.values, length * sizeof(int), src_is_device ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost);
  deviceFree(buffer.keys);
  deviceFree(buffer.values);
}

#undef T
