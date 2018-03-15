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

#include <cudanuSQuIDS/rk42d.hpp>


#include <cudanuSQuIDS/cuda_unique.cuh>
#include <cudanuSQuIDS/cudahelpers.cuh>
#include <cudanuSQuIDS/rk42d_kernels.hpp>

#include <stdexcept>

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace cudanusquids{

    namespace ode{

        namespace stepper{

                RK42D::RK42D(RK42D&& rhs){
                    *this = std::move(rhs);
                    cudaStreamCreate(&updateStream1); CUERR;
                    cudaStreamCreate(&updateStream2); CUERR;

                    for(size_t i = 0; i < nCopyStreams; i++)
                        cudaStreamCreate(&copyStreams[i]); CUERR;
                }


                RK42D::~RK42D(){
                    cudaSetDevice(deviceId);
                    cudaStreamDestroy(updateStream1); CUERR;
                    cudaStreamDestroy(updateStream2); CUERR;

                    for(size_t i = 0; i < nCopyStreams; i++)
                        cudaStreamDestroy(copyStreams[i]); CUERR;
                }

                RK42D& RK42D::operator=(RK42D&& rhs){
                        deviceId = rhs.deviceId;
                        pitchx = rhs.pitchx;
                        dimx = rhs.dimx;
                        dimy = rhs.dimy;
                        systemsize = rhs.systemsize;
                        ode_system_size = rhs.ode_system_size;
                        nSystems = rhs.nSystems;
                        stepfunc = rhs.stepfunc;
                        userdata = rhs.userdata;
                        tmp = std::move(rhs.tmp);
                        y0 = std::move(rhs.y0);
                        k = std::move(rhs.k);
                        t_tmp_step = std::move(rhs.t_tmp_step);
                        y_onestep = std::move(rhs.y_onestep);
                        h_tmp = std::move(rhs.h_tmp);
                        t_tmp = std::move(rhs.t_tmp);

                        return *this;
                }

                void RK42D::init(int deviceId_, size_t pitchx_, size_t dimx_, size_t dimy_, size_t nSystems_, size_t systemsize_, size_t ode_system_size_, void* userdata_,
                            void (*stepfunc_)(const size_t* const activeIndices, size_t nIndices,
                                                const double* const t, double* const y, double* const y_derived, void* userdata)){
                    deviceId = deviceId_;

                    pitchx = pitchx_;
                    dimx = dimx_;
                    dimy = dimy_;
                    systemsize = systemsize_;
                    nSystems = nSystems_;
                    userdata = userdata_;
                    stepfunc = stepfunc_;

                    ode_system_size = ode_system_size_;

                    int nGpus;
                    cudaGetDeviceCount(&nGpus); CUERR;

                    if(deviceId < 0 || deviceId >= nGpus)
                        throw std::runtime_error("RK42D::init : invalid device id");

                    cudaSetDevice(deviceId); CUERR;

                    tmp = make_unique_dev<double>(deviceId, ode_system_size);
                    y0 = make_unique_dev<double>(deviceId, ode_system_size);
                    k = make_unique_dev<double>(deviceId,ode_system_size);
                    y_onestep = make_unique_dev<double>(deviceId, ode_system_size);
                    t_tmp_step = make_unique_dev<double>(deviceId, nSystems);
                    h_tmp = make_unique_dev<double>(deviceId, nSystems);
                    t_tmp = make_unique_dev<double>(deviceId, nSystems);

                    cudaMemset(tmp.get(), 0, sizeof(double) * ode_system_size);
                    cudaMemset(y0.get(), 0, sizeof(double) * ode_system_size);
                    cudaMemset(k.get(), 0, sizeof(double) * ode_system_size);
                    cudaMemset(y_onestep.get(), 0, sizeof(double) * ode_system_size);
                    cudaMemset(t_tmp_step.get(), 0, sizeof(double) * nSystems);
                    cudaMemset(h_tmp.get(), 0, sizeof(double) * nSystems);
                    cudaMemset(t_tmp.get(), 0, sizeof(double) * nSystems);

                    cudaStreamCreate(&updateStream1);
                    cudaStreamCreate(&updateStream2);

                    for(size_t i = 0; i < nCopyStreams; i++)
                        cudaStreamCreate(&copyStreams[i]); CUERR;
                }

                void RK42D::step(const size_t* activeIndicesH, const size_t* activeIndicesD, size_t nIndices, double* y, const double* t, const double* h){

                    cudaSetDevice(deviceId); CUERR;

                    const size_t* activeIndices = activeIndicesD;

                    dim3 block(256, 1, 1);
                    dim3 grid(SDIV(systemsize, block.x), SDIV(nIndices, 1), 1);

                    dim3 blocksmall(128, 1, 1);
                    dim3 gridsmall(SDIV(nIndices, blocksmall.x), 1, 1);

                    call_rk42d_step_updateYandTmp(y, tmp.get(), h, y0.get(), k.get(), activeIndices, nIndices, systemsize, 6.0, 0.5, grid, block, updateStream1);
                    call_rk42d_step_updateTtmpstep(t_tmp_step.get(), t, h, activeIndices, nIndices, 0.5, gridsmall, blocksmall, updateStream2);

                    cudaStreamSynchronize(updateStream1); CUERR;
                    cudaStreamSynchronize(updateStream2); CUERR;

                    stepfunc(activeIndices, nIndices, t_tmp_step.get(), tmp.get(), k.get(), userdata);

                    call_rk42d_step_updateYandTmp(y, tmp.get(), h, y0.get(), k.get(), activeIndices, nIndices, systemsize, 3.0, 0.5, grid, block, updateStream1);
                    // no need to update time since k3 uses same time as k2.
                    cudaStreamSynchronize(updateStream1); CUERR;

                    stepfunc(activeIndices, nIndices, t_tmp_step.get(), tmp.get(), k.get(), userdata);

                    call_rk42d_step_updateYandTmp(y, tmp.get(), h, y0.get(), k.get(), activeIndices, nIndices, systemsize, 3.0, 1.0, grid, block, updateStream1);
                    call_rk42d_step_updateTtmpstep(t_tmp_step.get(), t, h, activeIndices, nIndices, 1.0, gridsmall, blocksmall, updateStream2);

                    cudaStreamSynchronize(updateStream1); CUERR;
                    cudaStreamSynchronize(updateStream2); CUERR;

                    stepfunc(activeIndices, nIndices, t_tmp_step.get(), tmp.get(), k.get(), userdata);

                    call_rk42d_step_updateY(y, h, k.get(), activeIndices, nIndices, systemsize, 6.0, grid, block, updateStream1);

                    cudaStreamSynchronize(updateStream1); CUERR;
                }

                void RK42D::step_apply(const size_t* activeIndicesH, const size_t* activeIndicesD, size_t nIndices, const double* t, const double* h,
                                    double* y, double* yerr, const double* dydt_in, double* dydt_out){

                    cudaSetDevice(deviceId); CUERR;

                    const size_t* activeIndices = activeIndicesD;

                    dim3 blocksmall(128, 1, 1);
                    dim3 gridsmall(SDIV(nIndices, blocksmall.y), 1, 1);

                    //copy y to y0, dydt_in to k, y + y_onestep
                    for(size_t i = 0; i < nIndices; i++){
                        //group copies of consecutive systems into single large copy
                        size_t begin = activeIndicesH[i];
                        size_t copyBatch = 1;
                        while(i+1 < nIndices && activeIndicesH[i+1] == activeIndicesH[i] + 1){
                            copyBatch++;
                            i++;
                        }

                        cudaMemcpyAsync(y0.get() + begin * systemsize,
                                        y + begin * systemsize,
                                        sizeof(double) * systemsize * copyBatch,
                                        D2D,
                                        copyStreams[i % nCopyStreams]); CUERR;
                        cudaMemcpyAsync(k.get() + begin * systemsize,
                                        dydt_in + begin * systemsize,
                                        sizeof(double) * systemsize * copyBatch,
                                        D2D,
                                        copyStreams[i % nCopyStreams]); CUERR;
                        cudaMemcpyAsync(y_onestep.get() + begin * systemsize,
                                        y + begin * systemsize,
                                        sizeof(double) * systemsize * copyBatch,
                                        D2D,
                                        copyStreams[i % nCopyStreams]); CUERR;
                    }

                    for(size_t i = 0; i < nCopyStreams; i++)
                        cudaStreamSynchronize(copyStreams[i]); CUERR;

                    step(activeIndicesH, activeIndicesD, nIndices, y_onestep.get(), t, h);

                    // calculate the half step sizes and the integration times for the second half step
                    call_rk42d_step_apply_updateHtmpAndTtmp(h_tmp.get(), t_tmp.get(), h, t, activeIndices, nIndices, 2.0, 2.0, gridsmall, blocksmall, updateStream1);

                    cudaStreamSynchronize(updateStream1); CUERR;

                    // first half step

                    // copy dydt_in to k
                    for(size_t i = 0; i < nIndices; i++){
                        size_t begin = activeIndicesH[i];
                        size_t copyBatch = 1;
                        while(i+1 < nIndices && activeIndicesH[i+1] == activeIndicesH[i] + 1){
                            copyBatch++;
                            i++;
                        }
                        cudaMemcpyAsync(k.get() + begin * systemsize,
                                        dydt_in + begin * systemsize,
                                        sizeof(double) * systemsize * copyBatch,
                                        D2D,
                                        copyStreams[i % nCopyStreams]); CUERR;
                    }

                    for(size_t i = 0; i < nCopyStreams; i++)
                        cudaStreamSynchronize(copyStreams[i]); CUERR;

                    step(activeIndicesH, activeIndicesD, nIndices, y, t, h_tmp.get());

                    // calculate k for second half step
                    stepfunc(activeIndices, nIndices, t_tmp.get(), y, k.get(), userdata);

                    // copy y to y0 for second half step
                    for(size_t i = 0; i < nIndices; i++){
                        size_t begin = activeIndicesH[i];
                        size_t copyBatch = 1;
                        while(i+1 < nIndices && activeIndicesH[i+1] == activeIndicesH[i] + 1){
                            copyBatch++;
                            i++;
                        }

                        cudaMemcpyAsync(y0.get() + begin * systemsize,
                        y + begin * systemsize,
                        sizeof(double) * systemsize * copyBatch,
                        D2D,
                        copyStreams[i % nCopyStreams]); CUERR;
                    }

                    for(size_t i = 0; i < nCopyStreams; i++)
                        cudaStreamSynchronize(copyStreams[i]); CUERR;

                    // second half step
                    step(activeIndicesH, activeIndicesD, nIndices, y, t_tmp.get(), h_tmp.get());

                    call_rk42d_step_updateTtmpstep(t_tmp.get(), t, h, activeIndices, nIndices, 1.0, gridsmall, blocksmall, updateStream1);

                    cudaStreamSynchronize(updateStream1); CUERR;

                    // evaluate function at t + h and save to dydt_out
                    stepfunc(activeIndices, nIndices, t_tmp.get(), y, dydt_out, userdata);

                    dim3 block(256, 1, 1);
                    dim3 grid(SDIV(systemsize, block.x), SDIV(nIndices, 1), 1);

                    call_rk42d_calculateError(activeIndices, nIndices, yerr,
                                    y, y_onestep.get(), systemsize, grid, block, updateStream1);
                    cudaStreamSynchronize(updateStream1); CUERR;
                }

        } //namespace stepper

    } // namespace ode

}
