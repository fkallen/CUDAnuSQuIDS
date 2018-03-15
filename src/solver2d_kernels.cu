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

#include <cudanuSQuIDS/solver2d_kernels.hpp>
#include <cudanuSQuIDS/cudahelpers.cuh>

#include <cudanuSQuIDS/cudautils.cuh>

#include <limits>

namespace cudanusquids{

    namespace ode{


        /* one block per path */
        KERNEL
        void solver2dHadjustKernel(int* const __restrict__ retVals, unsigned int ord,
                    const size_t* const __restrict__ activeIndices, size_t nIndices,
                    const double* const __restrict__ y, const double* const __restrict__ yerr, const double* const __restrict__ yp,
                    double* const __restrict__ h, size_t systemsize, double eps_abs, double eps_rel, size_t dimx, size_t dimy, size_t pitchx)
        {

            constexpr double a_y = 1;
            constexpr double a_dydt = 0;
            constexpr double S = 0.9;

            for(size_t i = blockIdx.y; i < nIndices; i += gridDim.y){
                const size_t si = activeIndices[i];

                const double h_old = h[si];

                double myrmax = std::numeric_limits<double>::min();

                for(size_t j = threadIdx.x; j < systemsize; j += blockDim.x){
                    const size_t index = si * systemsize + j;
                    const double yerr_ = yerr[index];
                    const double D0 =
                        eps_rel * (a_y * fabs (y[index]) + a_dydt * fabs (h_old * yp[index])) +
                        eps_abs;
                    const double r = fabs (yerr_) / fabs (D0);
                    myrmax = (myrmax > r ? myrmax : r);
                }
                __syncthreads();

                double reducedRMax;

                blockreduce<256>(&reducedRMax, myrmax, [](double l, double r){
                    return max(l, r);
                });


                if(threadIdx.x == 0){
                    if (reducedRMax > 1.1){
                        double r = S / pow (reducedRMax, 1.0 / ord);

                        if (r < 0.2)
                            r = 0.2;

                        h[si] = r * h_old;
                        retVals[i] = -1;
                    }else if (reducedRMax < 0.5){
                        double r = S / pow (reducedRMax, 1.0 / (ord + 1.0));

                        if (r > 5.0)
                            r = 5.0;

                        if (r < 1.0)
                            r = 1.0;

                        h[si] = r * h_old;
                        retVals[i] = 1;
                    }else{
                        retVals[i] = 0;
                    }
                }
            }
        }

        void callSolver2dHadjustKernelAsync(int* retVals, unsigned int ord,
                    const size_t* activeIndices, size_t nIndices,
                    const double* y, const double* yerr, const double* yp,
                    double* h, size_t systemsize, double eps_abs, double eps_rel, size_t dimx, size_t dimy, size_t pitchx,
                    dim3 grid, dim3 block, cudaStream_t stream){


            solver2dHadjustKernel<<<grid, block, 0, stream>>>(retVals, ord,
                    activeIndices, nIndices,
                    y, yerr,  yp,
                    h, systemsize, eps_abs, eps_rel, dimx, dimy, pitchx);
            CUERR;
        }

    }

}
