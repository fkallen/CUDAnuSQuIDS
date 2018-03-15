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

#include <cudanuSQuIDS/cudahelpers.cuh>
#include <cudanuSQuIDS/akima_interpolation.cuh>

#include <vector>
#include <cmath>

namespace cudanusquids{

	namespace Akima{

        void init_akima_interpolation_gpu_data_from_gpu_arrays(AkimaInterpolationData& ai, const double* x_arr_gpu, const double* y_arr_gpu, size_t size){
			std::vector<double> m(size+4);
			std::vector<double> b(size);
			std::vector<double> c(size);
			std::vector<double> d(size);

			std::vector<double> x_arr(size);
			std::vector<double> y_arr(size);

			cudaMemcpy(x_arr.data(), x_arr_gpu, sizeof(double) * size, D2H); CUERR;
			cudaMemcpy(y_arr.data(), y_arr_gpu, sizeof(double) * size, D2H); CUERR;

			double* shiftedM = m.data() + 2;

			for (size_t i = 0; i <= size - 2; i++){
				shiftedM[i] = (y_arr[i + 1] - y_arr[i]) / (x_arr[i + 1] - x_arr[i]);
			}

			/* non-periodic boundary conditions */
			shiftedM[-2] = 3.0 * shiftedM[0] - 2.0 * shiftedM[1];
			shiftedM[-1] = 2.0 * shiftedM[0] - shiftedM[1];
			shiftedM[size - 1] = 2.0 * shiftedM[size - 2] - shiftedM[size - 3];
			shiftedM[size] = 3.0 * shiftedM[size - 2] - 2.0 * shiftedM[size - 3];

			for (size_t i = 0; i < (size - 1); i++){
				const double NE = std::fabs(shiftedM[i + 1] - shiftedM[i]) + std::fabs(shiftedM[i - 1] - shiftedM[i - 2]);
				if (NE == 0.0){
					b[i] = shiftedM[i];
					c[i] = 0.0;
					d[i] = 0.0;
				}else{
					const double h_i = x_arr[i + 1] - x_arr[i];
					const double NE_next = std::fabs(shiftedM[i + 2] - shiftedM[i + 1]) + std::fabs(shiftedM[i] - shiftedM[i - 1]);
					const double alpha_i = std::fabs(shiftedM[i - 1] - shiftedM[i - 2]) / NE;

					double tL_ip1;
					if (NE_next == 0.0){
						tL_ip1 = shiftedM[i];
					}else{
						const double alpha_ip1 = std::fabs(shiftedM[i] - shiftedM[i - 1]) / NE_next;
						tL_ip1 = (1.0 - alpha_ip1) * shiftedM[i] + alpha_ip1 * shiftedM[i + 1];
					}
					b[i] = (1.0 - alpha_i) * shiftedM[i - 1] + alpha_i * shiftedM[i];
					c[i] = (3.0 * shiftedM[i] - 2.0 * b[i] - tL_ip1) / h_i;
					d[i] = (b[i] + tL_ip1 - 2.0 * shiftedM[i]) / (h_i * h_i);
				}
			}

			ai.size = size;
			cudaMalloc((void**)&ai.m, sizeof(double) * (size+4)); CUERR;
			cudaMalloc((void**)&ai.b, sizeof(double) * (size)); CUERR;
			cudaMalloc((void**)&ai.c, sizeof(double) * (size)); CUERR;
			cudaMalloc((void**)&ai.d, sizeof(double) * (size)); CUERR;

			cudaMemcpy(ai.m, m.data(), sizeof(double) * (size+4), H2D); CUERR;
			cudaMemcpy(ai.b, b.data(), sizeof(double) * (size), H2D); CUERR;
			cudaMemcpy(ai.c, c.data(), sizeof(double) * (size), H2D); CUERR;
			cudaMemcpy(ai.d, d.data(), sizeof(double) * (size), H2D); CUERR;

			ai.x_array = x_arr_gpu;
			ai.y_array = y_arr_gpu;
		}

		void destroy_akima_interpolation_gpu_data(AkimaInterpolationData& ai){
			cudaFree(ai.m); CUERR;
			cudaFree(ai.b); CUERR;
			cudaFree(ai.c); CUERR;
			cudaFree(ai.d); CUERR;
		}

    }

}
