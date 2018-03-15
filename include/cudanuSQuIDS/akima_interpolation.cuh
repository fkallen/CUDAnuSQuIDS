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

#ifndef CUDANUSQUIDS_AKIMA_INTERPOLATION_CUDA
#define CUDANUSQUIDS_AKIMA_INTERPOLATION_CUDA

#include <cudanuSQuIDS/cudahelpers.cuh>

#include <assert.h>
#include <vector>

namespace cudanusquids{

	namespace Akima{

		struct AkimaInterpolationData{
			size_t size;

			const double* x_array; //not owned
			const double* y_array; //not owned

			double* m;
			double* b;
			double* c;
			double* d;

			HOSTDEVICEQUALIFIER
			AkimaInterpolationData(){}

			HOSTDEVICEQUALIFIER
			double interpolation(const double x, size_t* cache) const{

				auto findindexwithcache = [](const double* x_array, const double x, const size_t size, size_t* cache) -> size_t{

					auto binarysearch = [](const double* x_array, double x, size_t index_lo, size_t index_hi) -> size_t{
						size_t ilo = index_lo;
						size_t ihi = index_hi;
						while(ihi > ilo + 1) {
							size_t i = (ihi + ilo)/2;
							if(x_array[i] > x)
								ihi = i;
							else
								ilo = i;
						}

						return ilo;
					};

					size_t index = *cache;
					if(x < x_array[index]){
						*cache = binarysearch(x_array, x, 0, index);
					}else{
						*cache = binarysearch(x_array, x, index, size-1);
					}
					return *cache;
				};

				size_t index = findindexwithcache(x_array, x, size, cache);
				const double x_lo = x_array[index];
				const double delx = x - x_lo;
				const double result = y_array[index] + delx * (b[index] + delx * (c[index] + d[index] * delx));
				return result;
			}
		};

		void init_akima_interpolation_gpu_data_from_gpu_arrays(AkimaInterpolationData& ai, const double* x_arr_gpu, const double* y_arr_gpu, size_t size);

		void destroy_akima_interpolation_gpu_data(AkimaInterpolationData& ai);

	}
}


#endif
