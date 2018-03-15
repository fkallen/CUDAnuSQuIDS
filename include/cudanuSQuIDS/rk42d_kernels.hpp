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

#ifndef CUDANUSQUIDS_RK42D_KERNELS_HPP
#define CUDANUSQUIDS_RK42D_KERNELS_HPP

#include <cudanuSQuIDS/cudahelpers.cuh>
#include <cudanuSQuIDS/cudautils.cuh>

namespace cudanusquids{

    namespace ode{

        namespace stepper{

            void call_rk42d_step_updateTtmpstep(double* t_tmp_step, const double* t, const double* h,
                        const size_t* activeIndices, size_t nIndices,
                        const double hfac, dim3 grid, dim3 block, cudaStream_t stream);

            void call_rk42d_step_updateYandTmp(double* y, double* tmp, const double* h, const double* y0, const double* k,
                        const size_t* activeIndices, size_t nIndices, size_t systemsize,
                        double ydiv, double tmpfac, dim3 grid, dim3 block, cudaStream_t stream);

            void call_rk42d_step_updateY(double* y, const double* h, const double* k,
                        const size_t* activeIndices, size_t nIndices, size_t systemsize,
                        double ydiv, dim3 grid, dim3 block, cudaStream_t stream);

            void call_rk42d_calculateError(const size_t* activeIndices, size_t nIndices, double* yerr,
                                const double* y, const double* y_onestep, size_t systemsize, dim3 grid, dim3 block, cudaStream_t stream);

            void call_rk42d_step_apply_updateHtmpAndTtmp(double* h_tmp, double* t_tmp, const double* h, const double* t,
                        const size_t* activeIndices, size_t nIndices,
                        double htmpdiv, double ttmpdiv, dim3 grid, dim3 block, cudaStream_t stream);
        }
    }
}

#endif
