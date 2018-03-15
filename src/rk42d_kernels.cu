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

#include <cudanuSQuIDS/rk42d_kernels.hpp>
#include <cudanuSQuIDS/cudahelpers.cuh>

namespace cudanusquids{

    namespace ode{

        namespace stepper{

            KERNEL
            void rk42d_step_updateTtmpstep(double* const __restrict__ t_tmp_step, const double* const __restrict__ t, const double* const __restrict__ h,
                        const size_t* const __restrict__ activeIndices, size_t nIndices,
                        const double hfac){

                for(size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < nIndices; i += blockDim.x * gridDim.x){
                    const size_t si = activeIndices[i];

                    t_tmp_step[si] = t[si] + hfac * h[si];
                }
            }

            KERNEL
            void rk42d_step_updateYandTmp(double* const __restrict__ y, double* const __restrict__ tmp, const double* const __restrict__ h, const double* const __restrict__ y0, const double* const __restrict__ k,
                        const size_t* const __restrict__ activeIndices, size_t nIndices, size_t systemsize,
                        const double ydiv, const double tmpfac){

                for(size_t i = blockIdx.y; i < nIndices; i += gridDim.y){
                    const size_t si = activeIndices[i];

                    for(size_t j = threadIdx.x + blockIdx.x * blockDim.x; j < systemsize; j += blockDim.x * gridDim.x){
                        const size_t index = si * systemsize + j;

                        y[index] += h[si] / ydiv * k[index];
                        tmp[index] = y0[index] + tmpfac * h[si] * k[index];
                    }
                }
            }


            KERNEL
            void rk42d_step_updateY(double* const __restrict__ y, const double* const __restrict__ h, const double* const __restrict__ k,
                        const size_t* const __restrict__ activeIndices, size_t nIndices, size_t systemsize,
                        const double ydiv){

                for(size_t i = blockIdx.y; i < nIndices; i += gridDim.y){
                    const size_t si = activeIndices[i];

                    for(size_t j = threadIdx.x + blockIdx.x * blockDim.x; j < systemsize; j += blockDim.x * gridDim.x){
                        const size_t index = si * systemsize + j;

                        y[index] += h[si] / ydiv * k[index];
                    }
                }
            }

            KERNEL
            void rk42d_calculateError(const size_t* const __restrict__ activeIndices, size_t nIndices, double* const __restrict__ yerr,
                                const double* const __restrict__ y, const double* const __restrict__ y_onestep, size_t systemsize)
            {
                constexpr double ODEIV_ERR_SAFETY = 8.0;

                for(size_t i = blockIdx.y; i < nIndices; i += gridDim.y){
                    const size_t si = activeIndices[i];

                    for(size_t j = threadIdx.x + blockIdx.x * blockDim.x; j < systemsize; j += blockDim.x * gridDim.x){
                        const size_t index = si * systemsize + j;

                        yerr[index] = ODEIV_ERR_SAFETY * 0.5 * (y[index] - y_onestep[index]) / 15.0;
                    }
                }
            }

            KERNEL
            void rk42d_step_apply_updateHtmpAndTtmp(double* const __restrict__ h_tmp, double* const __restrict__ t_tmp, const double* const __restrict__ h, const double* const __restrict__ t,
                        const size_t* const __restrict__ activeIndices, size_t nIndices,
                        const double htmpdiv, const double ttmpdiv){

                for(size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < nIndices; i += blockDim.x * gridDim.x){
                    const size_t si = activeIndices[i];

                    h_tmp[si] = h[si] / htmpdiv;
                    t_tmp[si] = t[si] + h[si] / ttmpdiv;
                }
            }

            void call_rk42d_step_updateTtmpstep(double* t_tmp_step, const double* t, const double* h,
                        const size_t* activeIndices, size_t nIndices,
                        const double hfac, dim3 grid, dim3 block, cudaStream_t stream){

                    rk42d_step_updateTtmpstep<<<grid, block, 0, stream>>>(t_tmp_step, t, h, activeIndices, nIndices, hfac); CUERR;
            }

            void call_rk42d_step_updateYandTmp(double* y, double* tmp, const double* h, const double* y0, const double* k,
                        const size_t* activeIndices, size_t nIndices, size_t systemsize,
                        double ydiv, double tmpfac, dim3 grid, dim3 block, cudaStream_t stream){

                    rk42d_step_updateYandTmp<<<grid, block, 0, stream>>>(y, tmp, h, y0, k, activeIndices, nIndices, systemsize, ydiv, tmpfac); CUERR;
            }

            void call_rk42d_step_updateY(double* y, const double* h, const double* k,
                        const size_t* activeIndices, size_t nIndices, size_t systemsize,
                        double ydiv, dim3 grid, dim3 block, cudaStream_t stream){

                    rk42d_step_updateY<<<grid, block, 0, stream>>>(y, h, k, activeIndices, nIndices, systemsize, ydiv); CUERR;
            }

            void call_rk42d_calculateError(const size_t* activeIndices, size_t nIndices, double* yerr,
                                const double* y, const double* y_onestep, size_t systemsize, dim3 grid, dim3 block, cudaStream_t stream){

                    rk42d_calculateError<<<grid, block, 0, stream>>>(activeIndices, nIndices, yerr, y, y_onestep, systemsize); CUERR;
            }

            void call_rk42d_step_apply_updateHtmpAndTtmp(double* h_tmp, double* t_tmp, const double* h, const double* t,
                        const size_t* activeIndices, size_t nIndices,
                        double htmpdiv, double ttmpdiv, dim3 grid, dim3 block, cudaStream_t stream){

                    rk42d_step_apply_updateHtmpAndTtmp<<<grid, block, 0, stream>>>(h_tmp, t_tmp, h, t, activeIndices, nIndices, htmpdiv, ttmpdiv); CUERR;
            }

        }
    }
}
