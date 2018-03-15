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

#ifndef CUDANUSQUIDS_RK42D_HPP
#define CUDANUSQUIDS_RK42D_HPP

#include <cudanuSQuIDS/cudahelpers.cuh>
#include <cudanuSQuIDS/cudautils.cuh>
#include <cudanuSQuIDS/cuda_unique.cuh>

namespace cudanusquids{

    namespace ode{

        namespace stepper{

            // Runge-Kutta 4 Stepper to use with Solver2D
            struct RK42D{
                static constexpr size_t nCopyStreams = 4;

                int deviceId;

                size_t ode_system_size;

                // ode_system consists of nSystems independent systems with following properties
                size_t pitchx;
                size_t dimx;
                size_t dimy;
                size_t systemsize;
                size_t nSystems;

                void (*stepfunc)(const size_t* activeIndices, size_t nIndices, const double* t, double* y, double* y_derived, void* userdata);
                void* userdata;

                unique_dev_ptr<double> tmp;
                unique_dev_ptr<double> y0;
                unique_dev_ptr<double> k;
                unique_dev_ptr<double> t_tmp_step;
                unique_dev_ptr<double> y_onestep;
                unique_dev_ptr<double> h_tmp;
                unique_dev_ptr<double> t_tmp;

                cudaStream_t updateStream1;
                cudaStream_t updateStream2;

                cudaStream_t copyStreams[nCopyStreams];

                RK42D(const RK42D& rhs) = delete;
                RK42D& operator=(const RK42D& rhs) = delete;

                RK42D(){}

                RK42D(RK42D&& rhs);

                ~RK42D();

                RK42D& operator=(RK42D&& rhs);

                void init(int deviceId, size_t pitchx, size_t dimx, size_t dimy, size_t nSystems, size_t systemsize, size_t ode_system_size, void* userdata,
                            void (*stepfunc)(const size_t* const activeIndices, size_t nIndices,
                                                const double* const t, double* const y, double* const y_derived, void* userdata));

                void step(const size_t* activeIndicesH, const size_t* activeIndicesD, size_t nIndices, double* y, const double* t, const double* h);

                void step_apply(const size_t* activeIndicesH, const size_t* activeIndicesD, size_t nIndices, const double* t, const double* h,
                                    double* y, double* yerr, const double* dydt_in, double* dydt_out);


                static constexpr unsigned int getOrder(){ return 4; }

            };

        } //namespace stepper

    } // namespace ode

}

#endif
