/*
This file is part of CUDAnuSQuIDS.

CUDAnuSQuIDS is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CUDAnuSQuIDS is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with CUDAnuSQuIDS.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef CUDANUSQUIDS_CUDA_UTILS_CUH
#define CUDANUSQUIDS_CUDA_UTILS_CUH

#include <vector>
#include <stdio.h>

#include <cudanuSQuIDS/cudahelpers.cuh>

#define WARP_SIZE 32


template <unsigned int MAX_BLOCK_DIM_X_, class S, class Func>
__device__ void blockreduce(S *result, S localvalue, Func func){

    __shared__ S sdata[MAX_BLOCK_DIM_X_];

    unsigned int tid = threadIdx.x;
    S myValue = localvalue;

    // each thread puts its local sum into shared memory
    sdata[tid] = localvalue;
    __syncthreads();

    // do reduction in shared mem
    if ((blockDim.x >= 1024) && (tid < 512)){
        sdata[tid] = myValue = func(myValue, sdata[tid + 512]);
    }
    __syncthreads();

    if ((blockDim.x >= 512) && (tid < 256)){
        sdata[tid] = myValue = func(myValue, sdata[tid + 256]);
    }
    __syncthreads();

    if ((blockDim.x >= 256) && (tid < 128)){
            sdata[tid] = myValue = func(myValue, sdata[tid + 128]);
    }
     __syncthreads();

    if ((blockDim.x >= 128) && (tid <  64)){
       sdata[tid] = myValue = func(myValue, sdata[tid +  64]);
    }
    __syncthreads();

    if ( tid < 32 ){
        // Fetch final intermediate sum from 2nd warp
        if (blockDim.x >=  64) myValue = func(myValue, sdata[tid + 32]);
        // Reduce final warp using shuffle
        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2){
            myValue = func(myValue, __shfl_down_sync(__activemask(), myValue, offset));
        }
    }

    if (tid == 0) *result = myValue;
}

/*
    Broadcast value from thread with threadIdx.x == tid to all other threads in the thread block.
*/
template <class S>
__device__
void blockbroadcast(S* result, S value, unsigned int tid){
	__shared__ S smem;
	if(threadIdx.x == tid)
		smem = value;
	__syncthreads();
	*result = smem;
}

// get entry in array allocated with cudamalloc2d
template<typename T>
__host__ __device__
T* getPitchedElement(T* base, size_t row, size_t col, size_t pitch){
	return (T*)((char*)base + row * pitch) + col;
}


#endif
