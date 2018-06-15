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

#ifndef CUDANUSQUIDS_SOLVER_GPU_CUH
#define CUDANUSQUIDS_SOLVER_GPU_CUH

#include <cudanuSQuIDS/cudautils.cuh>
#include <cudanuSQuIDS/types.hpp>

#include <cooperative_groups.h>

using namespace cooperative_groups;


namespace cudanusquids{
	namespace ode{

		/*
            Solver for a single ODE. To be run on the gpu in one thread block.

            Template parameter Stepper has to provide the following functions:

                // return the size (number of doubles) of the required work space to solve a system of size (elements) dimx
                __host__ __device__
                static size_t getMinimumMemorySize(size_t dimx);

                __device__
                Stepper();

                __device__
                ~Stepper();

                //init stepper. buffer = work space, dimx_ = number of elements in system
                __device__
                void init(double* buffer, size_t dimx_, void* userdata_,
                            void (*stepfunc_)(double t, double* const y, double* const y_derived, void* userdata));

                // perform Runge-Kutta step for time t with step size h
                __device__
                void step(double* y, double t, double h);

                // perform Runge-Kutta step with error estimation, for example via step doubling
                __device__
                void step_apply(double t, double h, double* y, double* yerr, const double* dydt_in, double* dydt_out);

                __device__
                static constexpr unsigned int getOrder(); // return order of the used method
        */

        template<typename Stepper>
        struct SolverGPU{

            Stepper stepper;

            double* ode_system; //dimensions pitchx * dimy bytes

            size_t pitchx;
            size_t dimy;

            unsigned int nSteps;

            RKstats stats;

            double h;
            double h_min;
            double h_max;

            double t_begin;
            double t_end;

            void* userdata;
            void (*stepfunc)(double t, double* const y, double* const y_derived, void* userdata);

            double eps_abs = 1.e-6;
            double eps_rel = 1.e-6;

            double* y0evolve;
            double* error;
            double* dydt_in;
            double* dydt_out;

            // returns number of elements (doubles), not bytes
            HOSTDEVICEQUALIFIER
            static size_t getMinimumMemorySize(size_t pitchx, size_t dimy){
                size_t paddedPitchx = pitchx;

                size_t systemsize = paddedPitchx / sizeof(double) * dimy;
                size_t myreqs = systemsize * 4;
                size_t stepperreqs = Stepper::getMinimumMemorySize(systemsize);

                return myreqs + stepperreqs;
            }

            DEVICEQUALIFIER
            SolverGPU(double* buffer, double* ode_system_, size_t pitchx_, size_t dimy_, unsigned int nSteps_, double h_, double h_min_, double h_max_,
                        double t_begin_, double t_end_,
                        void* userdata_,
                        void (*stepfunc_)(double t, double* const y, double* const y_derived, void* userdata))
                : ode_system(ode_system_), pitchx(pitchx_), dimy(dimy_), nSteps(nSteps_), h(h_), h_min(h_min_), h_max(h_max_),
                    t_begin(t_begin_), t_end(t_end_),
                    y0evolve(nullptr), error(nullptr), dydt_in(nullptr), dydt_out(nullptr),
                    userdata(userdata_),
                    stepfunc(stepfunc_)
                {
                    size_t systemsize = pitchx / sizeof(double) * dimy;

                    y0evolve = buffer;
                    error = y0evolve + systemsize;
                    dydt_in = error + systemsize;
                    dydt_out = dydt_in + systemsize;

                    for(size_t j = threadIdx.x + blockDim.x * blockIdx.x; j < systemsize; j += blockDim.x * gridDim.x){
                        y0evolve[j] = 0;
                        error[j] = 0;
                        dydt_in[j] = 0;
                        dydt_out[j] = 0;
                    }
                    systembarrier();

                    double* stepperbuf = dydt_out + systemsize;
                    stepper.init(stepperbuf, systemsize, userdata, stepfunc);
                }

            DEVICEQUALIFIER
            ~SolverGPU(){}

            DEVICEQUALIFIER
            void setAbsolutePrecision(double prec){eps_abs = prec;}

            DEVICEQUALIFIER
            void setRelativePrecision(double prec){eps_rel = prec;}

            DEVICEQUALIFIER
            void systembarrier() const{
                __syncthreads();
            }

            /*
                Check errors and adjust the step size h accordingly
            */
            DEVICEQUALIFIER
            int hadjust(unsigned int ord, const double* y, double* yerr, const double* yp, double* h) const
            {
                int retval = 0;
                double h_new = h_min;

                constexpr double a_y = 1;
                constexpr double a_dydt = 0;
                constexpr double S = 0.9;

                const double h_old = *h;

                double myrmax = std::numeric_limits<double>::min();
                double rmax = std::numeric_limits<double>::min();

                size_t systemsize = pitchx / sizeof(double) * dimy;

                for(size_t j = threadIdx.x + blockDim.x * blockIdx.x;
                            j < systemsize;
                            j += blockDim.x * gridDim.x){
                    const double D0 =
                        eps_rel * (a_y * fabs (y[j])
                                + a_dydt * fabs (h_old * yp[j]))
                        + eps_abs;
                    const double r = fabs (yerr[j]) / fabs (D0);
                    myrmax = (myrmax > r ? myrmax : r);
                }

                // perform max reduce within the block on myrmax and store maximum in rmax
                auto op = [](auto a, auto b){
					return max(a,b);
				};

                auto tile = tiled_partition<32>(this_thread_block());
				for (int i = tile.size() / 2; i > 0; i /= 2) {
					myrmax = op(myrmax, tile.shfl_down(myrmax, i));
				}

                if(threadIdx.x + blockDim.x * blockIdx.x == 0){
                    *((unsigned long long int*)yerr) = 0;
                }
                systembarrier();

				if (tile.thread_rank() == 0) atomicMax((unsigned long long int*)yerr, __double_as_longlong(myrmax));
				systembarrier();

                rmax = __longlong_as_double(*((unsigned long long int*)yerr));
                //reduction finished

                if(threadIdx.x + blockDim.x * blockIdx.x == 0){
                    if (rmax > 1.1){
                        double r = S / pow (rmax, 1.0 / ord);

                        if (r < 0.2)
                            r = 0.2;

                        h_new = r * h_old;
                        retval = -1;
                    }else if (rmax < 0.5){
                        double r = S / pow (rmax, 1.0 / (ord + 1.0));

                        if (r > 5.0)
                            r = 5.0;

                        if (r < 1.0)
                            r = 1.0;

                        h_new = r * h_old;
                        retval = 1;
                    }else{
                        /* no change */
                        h_new = h_old;
                        retval = 0;
                    }
                }

                blockbroadcast(&retval, retval, 0);
                blockbroadcast(&h_new, h_new, 0);

                systembarrier();
                //global broadcast retval and h_new.
                if(threadIdx.x + blockDim.x * blockIdx.x == 0){
                    yerr[0] = retval;
                    yerr[1] = h_new;
                }

                systembarrier();
                *h = yerr[1];
                return yerr[0];
            }

            DEVICEQUALIFIER
            void solveAdaptive(){
                size_t systemsize = pitchx / sizeof(double) * dimy;

                double t_now = t_begin;
                double h_step = h;
                bool final_step = false;
                int sign = 0;

                /* Determine integration direction sign */
                if (h_step > 0.0){
                    sign = 1;
                }else{
                    sign = -1;
                }

                if (sign * (t_end - t_begin) < 0.0){
                    printf("integration limits and/or step direction not consistent\n");
                    stats.status = Status::failure;
                    return;
                }

                const size_t tidx = threadIdx.x + blockDim.x * blockIdx.x;

                while(sign * (t_end - t_now) > 0.0){
                    //evolve begin
                    const double t0 = t_now;
                    double h0 = h_step;
                    const double dt = t_end - t_now;

                    /* save the original system for step() */
                    for(size_t j = tidx; j < systemsize / 2; j += blockDim.x * gridDim.x){
                        reinterpret_cast<double2*>(y0evolve)[j] = reinterpret_cast<const double2*>(ode_system)[j];
                    }

                    for(size_t j = tidx + systemsize/2 * 2; j < systemsize; j += tidx){
                        y0evolve[j] = ode_system[j];
                    }

                    systembarrier();

                    if(stats.steps == 0){
                        stepfunc(t0, ode_system, dydt_in, userdata);
                    }else{
                        for(size_t j = tidx; j < systemsize / 2; j += blockDim.x * gridDim.x){
                            reinterpret_cast<double2*>(dydt_in)[j] = reinterpret_cast<const double2*>(dydt_out)[j];
                        }

                        for(size_t j = tidx + systemsize/2 * 2; j < systemsize; j += tidx){
                            dydt_in[j] = dydt_out[j];
                        }

                        systembarrier();
                    }
                    bool stepOk = true;

                    do{
                        if ((dt >= 0.0 && h0 > dt) || (dt < 0.0 && h0 < dt)){
                            h0 = dt;
                            final_step = true;
                        }else{
                            final_step = false;
                        }

                        stats.steps++;

                        stepper.step_apply(t0, h0, ode_system, error, dydt_in, dydt_out);

                        if (final_step){
                            t_now = t_end;
                        }else{
                            t_now = t0 + h0;
                        }

                        const double h_old = h0;
                        const int hadjust_status = hadjust(Stepper::getOrder(), ode_system, error, dydt_out, &h0);

                        if (hadjust_status == -1)
                        {
                            double t_curr = t_now;
                            double t_next = (t_now) + h0;

                            if (fabs (h0) < fabs (h_old) && t_next != t_curr)
                            {
                                /* Step was decreased. Undo step, and try again with new h0. */
                                for(size_t j = tidx; j < systemsize / 2; j += blockDim.x * gridDim.x){
                                    reinterpret_cast<double2*>(ode_system)[j] = reinterpret_cast<const double2*>(y0evolve)[j];
                                }

                                for(size_t j = tidx + systemsize/2 * 2; j < systemsize; j += tidx){
                                    ode_system[j] = y0evolve[j];
                                }
                                systembarrier();

                                stats.repeats++;
                                stepOk = false;
                            }
                            else
                            {
                                h_step = h0;
                                stats.status = Status::failure;
                                return;
                            }
                        }else{
                            stepOk = true;
                        }
                    }while(stepOk == false);

                    if (!final_step){
                        h_step = h0;
                    }
                    // evolve end

                    if(fabs(h_step) < h_min){
                        stats.status = Status::failure;	//failure
                        return;
                    }

                    if(fabs(h_step) > h_max)
                        h_step = sign * h_max;
                }

                stats.status = Status::success;
            }

            DEVICEQUALIFIER
            void solveFixed(){
                size_t systemsize = pitchx / sizeof(double) * dimy;
                double t_now = t_begin;
                double h_step = (t_end - t_begin) / (double)nSteps;

                int sign = 0;

                if (h_step > 0.0){
                    sign = 1;
                }else{
                    sign = -1;
                }
                if (sign * (t_end - t_begin) < 0.0){
                    printf("integration limits and/or step direction not consistent\n");
                    stats.status = Status::failure;
                }

                const size_t tidx = threadIdx.x + blockDim.x * blockIdx.x;

                for(unsigned int step = 0; step < nSteps && step < nSteps; step++){
                    for(size_t j = tidx; j < systemsize / 2; j += blockDim.x * gridDim.x){
                        reinterpret_cast<double2*>(y0evolve)[j] = reinterpret_cast<const double2*>(ode_system)[j];
                    }

                    for(size_t j = tidx + systemsize/2 * 2; j < systemsize; j += tidx){
                        y0evolve[j] = ode_system[j];
                    }

                    systembarrier();

                    stepfunc(t_now, ode_system, dydt_in, userdata);

                    stepper.step_apply(t_now, h_step, ode_system, error, dydt_in, dydt_out);

                    double htemp = h_step;

                    const int hadjust_status = hadjust(Stepper::getOrder(), ode_system, error, dydt_out, &htemp);

                    if(hadjust_status == -1){
                        for(size_t j = tidx; j < systemsize / 2; j += blockDim.x * gridDim.x){
                            reinterpret_cast<double2*>(ode_system)[j] = reinterpret_cast<const double2*>(y0evolve)[j];
                        }

                        for(size_t j = tidx + systemsize/2 * 2; j < systemsize; j += tidx){
                            ode_system[j] = y0evolve[j];
                        }

                        stats.status = Status::failure;
                        return;
                    }

                    t_now += h_step;

                    stats.steps++;
                }

                stats.status = Status::success;
            }
        };

	}
}


#endif
