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

#ifndef CUDANUSQUIDS_RK4_CUH
#define CUDANUSQUIDS_RK4_CUH

#include <cudanuSQuIDS/cudahelpers.cuh>
#include <cudanuSQuIDS/cudautils.cuh>

#include <cooperative_groups.h>

using namespace cooperative_groups;


namespace cudanusquids{

    namespace ode{

            namespace stepper{

            // Runge-Kutta 4 for a single system. To be used on the GPU as a stepper for SolverGPU in one threadblock.
            struct RK4{
                size_t dimx;

                void (*stepfunc)(double t, double* y, double* y_derived, void* userdata);
                void* userdata;

                double* y0;
                double* k;
                double* tmp;
                double* y_onestep;

                // returns number of elements (doubles), not bytes
                HOSTDEVICEQUALIFIER
                static size_t getMinimumMemorySize(size_t dimx){
                    size_t paddedDimx = dimx;
                    return paddedDimx * 4;
                }

                DEVICEQUALIFIER
                RK4() {};

                DEVICEQUALIFIER
                ~RK4(){};

                DEVICEQUALIFIER
                void init(double* buffer, size_t dimx_, void* userdata_,
                            void (*stepfunc_)(double t, double* const y, double* const y_derived, void* userdata)){

                    dimx = dimx_;
                    userdata = userdata_;
                    stepfunc = stepfunc_;

                    y0 = buffer;
                    k = y0 + dimx;
                    tmp = k + dimx;
                    y_onestep = tmp + dimx;

                    for(size_t j = threadIdx.x + blockDim.x * blockIdx.x; j < dimx; j += blockDim.x * gridDim.x){
                        y0[j] = 0;
                        k[j] = 0;
                        tmp[j] = 0;
                        y_onestep[j] = 0;
                    }
                    systembarrier();
                }

                DEVICEQUALIFIER
                void systembarrier() const{
    #if 0
    //#ifdef USE_GRID_SYNC
                    this_grid().sync();
                    //__syncthreads();
    #else
                    __syncthreads();
    #endif
                }

                DEVICEQUALIFIER
                void step(double* y, double t, double h) {
                    const size_t tidx = threadIdx.x + blockDim.x * blockIdx.x;

                    for(size_t j = tidx; j < dimx / 2; j += blockDim.x * gridDim.x){
                        const double2 k2 = reinterpret_cast<const double2*>(k)[j];
                        const double2 y02 = reinterpret_cast<const double2*>(y0)[j];
                        double2 y2 = reinterpret_cast<const double2*>(y)[j];

                        const double a = y02.x + h / 2.0 * k2.x;
                        const double b = y02.y + h / 2.0 * k2.y;
                        const double2 tmp2 =  make_double2(a,b);
                        reinterpret_cast<double2*>(tmp)[j] = tmp2;

                        y2.x += h / 6.0 * k2.x;
                        y2.y += h / 6.0 * k2.y;
                        reinterpret_cast<double2*>(y)[j] = y2;
                    }

                    for(size_t j = tidx + dimx/2 * 2; j < dimx; j += tidx){
                        y[j] += h / 6.0 * k[j];
                        tmp[j] = y0[j] + h / 2.0 * k[j];
                    }

                    systembarrier();

                    const double time2 = t + h / 2.0;

                    stepfunc(time2, tmp, k, userdata);

                    for(size_t j = tidx; j < dimx / 2; j += blockDim.x * gridDim.x){
                        const double2 k2 = reinterpret_cast<const double2*>(k)[j];
                        const double2 y02 = reinterpret_cast<const double2*>(y0)[j];
                        double2 y2 = reinterpret_cast<const double2*>(y)[j];

                        const double a = y02.x + h / 2.0 * k2.x;
                        const double b = y02.y + h / 2.0 * k2.y;
                        const double2 tmp2 = make_double2(a,b);
                        reinterpret_cast<double2*>(tmp)[j] = tmp2;

                        y2.x += h / 3.0 * k2.x;
                        y2.y += h / 3.0 * k2.y;
                        reinterpret_cast<double2*>(y)[j] = y2;
                    }

                    for(size_t j = tidx + dimx/2 * 2; j < dimx; j += tidx){
                        y[j] += h / 3.0 * k[j];
                        tmp[j] = y0[j] + h / 2.0 * k[j];
                    }

                    systembarrier();

                    const double time3 = t + h / 2.0;

                    stepfunc(time3, tmp, k, userdata);

                    for(size_t j = tidx; j < dimx / 2; j += blockDim.x * gridDim.x){
                        const double2 k2 = reinterpret_cast<const double2*>(k)[j];
                        const double2 y02 = reinterpret_cast<const double2*>(y0)[j];
                        double2 y2 = reinterpret_cast<const double2*>(y)[j];

                        const double a = y02.x + h * k2.x;
                        const double b = y02.y + h * k2.y;
                        const double2 tmp2 = make_double2(a,b);
                        reinterpret_cast<double2*>(tmp)[j] = tmp2;

                        y2.x += h / 3.0 * k2.x;
                        y2.y += h / 3.0 * k2.y;
                        reinterpret_cast<double2*>(y)[j] = y2;
                    }

                    for(size_t j = tidx + dimx/2 * 2; j < dimx; j += tidx){
                        y[j] += h / 3.0 * k[j];
                        tmp[j] = y0[j] + h * k[j];
                    }

                    systembarrier();

                    const double time4 = t + h;

                    stepfunc(time4, tmp, k, userdata);

                    for(size_t j = tidx; j < dimx / 2; j += blockDim.x * gridDim.x){
                        const double2 k2 = reinterpret_cast<const double2*>(k)[j];
                        double2 y2 = reinterpret_cast<const double2*>(y)[j];

                        y2.x += h / 6.0 * k2.x;
                        y2.y += h / 6.0 * k2.y;
                        reinterpret_cast<double2*>(y)[j] = y2;
                    }

                    for(size_t j = tidx + dimx/2 * 2; j < dimx; j += tidx){
                        y[j] += h / 6.0 * k[j];
                    }

                    systembarrier();
                }


                DEVICEQUALIFIER
                void step_apply(double t, double h, double* const __restrict__ y, double* const __restrict__ yerr, const double* const __restrict__ dydt_in, double* const __restrict__ dydt_out) {
                    constexpr double ODEIV_ERR_SAFETY = 8.0;

                    /*
                        uses step doubling to estimate the error
                    */

                    const size_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
                    for(size_t j = tidx; j < dimx / 2; j += blockDim.x * gridDim.x){
                        reinterpret_cast<double2*>(y0)[j] = reinterpret_cast<const double2*>(y)[j];
                        reinterpret_cast<double2*>(k)[j] = reinterpret_cast<const double2*>(dydt_in)[j];
                        reinterpret_cast<double2*>(y_onestep)[j] = reinterpret_cast<const double2*>(y)[j];
                    }

                    for(size_t j = tidx + dimx/2 * 2; j < dimx; j += tidx){
                        y0[j] = y[j];
                        k[j] = dydt_in[j];
                        y_onestep[j] = y[j];
                    }

                    systembarrier();

                    step(y_onestep, t, h); // one step with full step size

                    /* two half steps */
                    for(size_t j = tidx; j < dimx / 2; j += blockDim.x * gridDim.x){
                        reinterpret_cast<double2*>(k)[j] = reinterpret_cast<const double2*>(dydt_in)[j];
                    }

                    for(size_t j = tidx + dimx/2 * 2; j < dimx; j += tidx){
                        k[j] = dydt_in[j];
                    }

                    systembarrier();

                    step(y, t, h / 2.0); // first step with half the step size

                    const double tt = t + h / 2.0;

                    stepfunc(tt, y, k, userdata);

                    for(size_t j = tidx; j < dimx / 2; j += blockDim.x * gridDim.x){
                        reinterpret_cast<double2*>(y0)[j] = reinterpret_cast<const double2*>(y)[j];
                    }

                    for(size_t j = tidx + dimx/2 * 2; j < dimx; j += tidx){
                        y0[j] = y[j];
                    }

                    systembarrier();

                    step(y, tt, h / 2.0); // second step with half the step size

                    const double ttt = t + h;

                    systembarrier();

                    stepfunc(ttt, y, dydt_out, userdata);

                    systembarrier();

                    for(size_t j = tidx; j < dimx / 2; j += blockDim.x * gridDim.x){
                        const double2 y2 = reinterpret_cast<const double2*>(y)[j];
                        const double2 y_onestep2 = reinterpret_cast<const double2*>(y_onestep)[j];
                        const double a = ODEIV_ERR_SAFETY * 0.5 * (y2.x - y_onestep2.x) / 15.0;
                        const double b = ODEIV_ERR_SAFETY * 0.5 * (y2.y - y_onestep2.y) / 15.0;
                        const double2 yerr2 = make_double2(a,b);
                        reinterpret_cast<double2*>(yerr)[j] = yerr2;
                    }

                    for(size_t j = tidx + dimx/2 * 2; j < dimx; j += tidx){
                        const double y2 = y[j];
                        const double y_onestep2 = y_onestep[j];
                        const double a = ODEIV_ERR_SAFETY * 0.5 * (y2 - y_onestep2) / 15.0;
                        yerr[j] = a;
                    }

                    systembarrier();
                }

                DEVICEQUALIFIER
                static constexpr unsigned int getOrder(){ return 4; }

            };

        } //namespace stepper

    } // namespace ode

}

#endif
