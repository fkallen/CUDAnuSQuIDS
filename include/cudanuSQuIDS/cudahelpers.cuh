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

#ifndef CUDANUSQUIDS_CUDA_HELPERS_CUH
#define CUDANUSQUIDS_CUDA_HELPERS_CUH

#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define CUERR {                                                            \
    cudaError_t err;                                                       \
    if ((err = cudaGetLastError()) != cudaSuccess) {                       \
        std::cout << "CUDA error: " << cudaGetErrorString(err) << " : "    \
                  << __FILE__ << ", line " << __LINE__ << std::endl;       \
        exit(1);                                                           \
    }                                                                      \
}

#define H2D (cudaMemcpyHostToDevice)
#define D2H (cudaMemcpyDeviceToHost)
#define H2H (cudaMemcpyHostToHost)
#define D2D (cudaMemcpyDeviceToDevice)

// safe division
#define SDIV(x,y)(((x)+(y)-1)/(y))

#ifdef __CUDA_ARCH__
    #define UNROLLQUALIFIER #pragma unroll
#else
    #define UNROLLQUALIFIER
#endif

#ifdef __CUDACC__
    #define HOSTQUALIFIER __host__
    #define DEVICEQUALIFIER __device__
    #define HOSTDEVICEQUALIFIER __host__ __device__
    #define KERNEL __global__
#else
    #define HOSTQUALIFIER
    #define DEVICEQUALIFIER
    #define HOSTDEVICEQUALIFIER
    #define KERNEL
#endif




#endif
