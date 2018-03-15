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

#ifndef CUDANUSQUIDS_SOLVER2D_HPP
#define CUDANUSQUIDS_SOLVER2D_HPP

#include <cudanuSQuIDS/cudahelpers.cuh>
#include <cudanuSQuIDS/cudautils.cuh>
#include <cudanuSQuIDS/cuda_unique.cuh>
#include <cudanuSQuIDS/types.hpp>

#include <cudanuSQuIDS/solver2d_kernels.hpp>


namespace cudanusquids{

    namespace ode{

        template<typename Stepper, size_t nCopyStreams = 4>
        struct Solver2D{

            Stepper stepper;

            cudaStream_t updateStream1;
            cudaStream_t updateStream2;

            cudaStream_t copyStreams[nCopyStreams];

            const unsigned int nSteps;

            double eps_abs = 1.e-6;
            double eps_rel = 1.e-6;

            int id;

            double* const ode_system;
            size_t ode_system_size; // system has ode_system_size elements

            // ode_system consists of nSystems independent systems with following properties
            size_t pitchx;
            size_t dimx;
            size_t dimy;
            size_t systemsize;
            size_t nSystems;

            double h;
            double h_min;
            double h_max;

            bool showProgress = false;
            void (*updateProgress)(double progress, double maxProgress);

            void (*stepfunc)(const size_t* activeIndices, size_t nIndices, const double* t, double* y, double* y_derived, void* userdata);
            void* userdata;

            unique_dev_ptr<double> y0evolve;
            unique_dev_ptr<double> error;
            unique_dev_ptr<double> dydt_in;
            unique_dev_ptr<double> dydt_out;

            unique_dev_ptr<size_t> activeSystemsTimeloopD;
            unique_dev_ptr<size_t> activeSystemsHloopD;

            unique_pinned_ptr<size_t> activeSystemsTimeloopH;
            unique_pinned_ptr<size_t> activeSystemsHloopH;

            unique_pinned_ptr<double> tnowH;
            unique_dev_ptr<double> tnowD;

            unique_pinned_ptr<double> t0H;
            unique_dev_ptr<double> t0D;

            unique_pinned_ptr<double> hstepH;
            unique_dev_ptr<double> hstepD;

            unique_pinned_ptr<double> h0H;
            unique_dev_ptr<double> h0D;

            unique_pinned_ptr<double> holdH;
            unique_dev_ptr<double> holdD;

            unique_pinned_ptr<int> hadjustRetvalsH;
            unique_dev_ptr<int> hadjustRetvalsD;

            std::vector<double> t_begin;
            std::vector<double> t_end;

            std::vector<RKstats> stats;

            Solver2D(int deviceId, double* ode_system, size_t pitchx_, size_t dimx_, size_t dimy_, size_t nSystems_,
                        unsigned int nSteps_, double h_, double h_min_, double h_max_,
                        const std::vector<double>& t_begin_Host , const std::vector<double>& t_end_Host ,void* userdata,
                        void (*stepfunc_)(const size_t* activeIndices, size_t nIndices, const double* t, double* y, double* y_derived, void* userdata),
                        void (*updateProgress_)(double progress, double maxProgress))
                : id(deviceId),
                    ode_system(ode_system),
                    pitchx(pitchx_),
                    dimx(dimx_),
                    dimy(dimy_),
                    nSystems(nSystems_),
                    systemsize(pitchx_ / sizeof(double) * dimy_),
                    ode_system_size(nSystems_ * pitchx_ / sizeof(double) * dimy_),
                    nSteps(nSteps_),
                    h(h_),
                    h_min(h_min_),
                    h_max(h_max_),
                    t_begin(t_begin_Host),
                    t_end(t_end_Host),
                    userdata(userdata),
                    stepfunc(stepfunc_),
                    updateProgress(updateProgress_)
            {
                int nGpus;
                cudaGetDeviceCount(&nGpus); CUERR;

                if(id < 0 || id >= nGpus)
                    throw std::runtime_error("Solver2D::Solver2D : invalid device id");

                cudaSetDevice(id); CUERR;

                stepper.init(deviceId, pitchx, dimx, dimy, nSystems, systemsize, ode_system_size, userdata, stepfunc);

                y0evolve = make_unique_dev<double>(id, ode_system_size);
                error = make_unique_dev<double>(id, ode_system_size);
                dydt_in = make_unique_dev<double>(id, ode_system_size);
                dydt_out = make_unique_dev<double>(id, ode_system_size);

                cudaMemset(y0evolve.get(), 0, sizeof(double) * ode_system_size);
                cudaMemset(dydt_in.get(), 0, sizeof(double) * ode_system_size);
                cudaMemset(dydt_out.get(), 0, sizeof(double) * ode_system_size);

                activeSystemsTimeloopD = make_unique_dev<size_t>(id, nSystems);
                activeSystemsHloopD = make_unique_dev<size_t>(id, nSystems);

                activeSystemsTimeloopH = make_unique_pinned<size_t>(nSystems);
                activeSystemsHloopH = make_unique_pinned<size_t>(nSystems);

                tnowH = make_unique_pinned<double>(nSystems);
                tnowD = make_unique_dev<double>(id, nSystems);

                t0H = make_unique_pinned<double>(nSystems);
                t0D = make_unique_dev<double>(id, nSystems);

                hstepH = make_unique_pinned<double>(nSystems);
                hstepD = make_unique_dev<double>(id, nSystems);

                h0H = make_unique_pinned<double>(nSystems);
                h0D = make_unique_dev<double>(id, nSystems);

                holdH = make_unique_pinned<double>(nSystems);
                holdD = make_unique_dev<double>(id, nSystems);

                hadjustRetvalsH = make_unique_pinned<int>(nSystems);
                hadjustRetvalsD = make_unique_dev<int>(id, nSystems);

                stats.resize(nSystems);

                cudaStreamCreate(&updateStream1); CUERR;
                cudaStreamCreate(&updateStream2); CUERR;

                for(size_t i = 0; i < nCopyStreams; i++){
                    cudaStreamCreate(&copyStreams[i]); CUERR;
                }
            }

            ~Solver2D(){
                cudaStreamDestroy(updateStream1); CUERR;
                cudaStreamDestroy(updateStream2); CUERR;

                for(size_t i = 0; i < nCopyStreams; i++){
                    cudaStreamDestroy(copyStreams[i]); CUERR;
                }
            }

            void setShowProgress(bool show){
                showProgress = show;
            }

            void setAbsolutePrecision(double prec){
                eps_abs = prec;
            }

            void setRelativePrecision(double prec){
                eps_rel = prec;
            }

            void solveFixed(){

                size_t nActiveSystemsTimeloop = nSystems;

                for(size_t i = 0; i < nSystems; i++){
                    tnowH.get()[i] = t_begin[i];
                    hstepH.get()[i] = (t_end[i] - t_begin[i]) / (double)nSteps;

                    int sign = 0;

                    if (hstepH.get()[i] > 0.0){
                        sign = 1;
                    }else{
                        sign = -1;
                    }
                    if (sign * (t_end[i] - t_begin[i]) < 0.0){
                        printf("integration limits and/or step direction not consistent\n");
                        stats[i].status = Status::failure;
                    }
                    if (sign * (t_end[i] - t_begin[i]) > 0.0){
                        activeSystemsTimeloopH.get()[i] = i;
                    }
                }
                cudaMemcpyAsync(activeSystemsTimeloopD.get(), activeSystemsTimeloopH.get(), sizeof(size_t) * nActiveSystemsTimeloop, H2D, copyStreams[0]); CUERR;
                cudaMemcpyAsync(tnowD.get(), tnowH.get(), sizeof(double) * nActiveSystemsTimeloop, H2D, copyStreams[1]); CUERR;
                cudaMemcpyAsync(hstepD.get(), hstepH.get(), sizeof(double) * nActiveSystemsTimeloop, H2D, copyStreams[2]); CUERR;

                if(showProgress)
                    updateProgress(0.0, 1.0);

                cudaStreamSynchronize(copyStreams[0]); CUERR;
                cudaStreamSynchronize(copyStreams[1]); CUERR;
                cudaStreamSynchronize(copyStreams[2]); CUERR;

                std::vector<size_t> failedSystems;
                std::vector<size_t> newActiveSystems(nSystems);

                bool activeSystemsListNeedsUpdate = false;

                for(unsigned int step = 0; step < nSteps && nActiveSystemsTimeloop > 0 && step < nSteps; step++){

                    cudaMemcpyAsync(y0evolve.get(), ode_system, sizeof(double) * ode_system_size, D2D, copyStreams[0]); CUERR;
                    cudaStreamSynchronize(copyStreams[0]); CUERR;

                    stepfunc(activeSystemsTimeloopD.get(), nActiveSystemsTimeloop, tnowD.get(), ode_system, dydt_in.get(), userdata);

                    stepper.step_apply(activeSystemsTimeloopH.get(),activeSystemsTimeloopD.get(), nActiveSystemsTimeloop, tnowD.get(), hstepD.get(),
                                        ode_system, error.get(), dydt_in.get(), dydt_out.get());

                    cudaMemcpyAsync(holdD.get(), hstepD.get(), sizeof(double) * nActiveSystemsTimeloop, D2D, updateStream1); CUERR;

                    callSolver2dHadjustKernelAsync(
                                hadjustRetvalsD.get(), Stepper::getOrder(),
                                activeSystemsTimeloopD.get(), nActiveSystemsTimeloop,
                                ode_system, error.get(), dydt_out.get(),
                                holdD.get(), systemsize, eps_abs, eps_rel, dimx, dimy, pitchx,
                                dim3(1, nActiveSystemsTimeloop, 1), dim3(256, 1, 1), updateStream1);

                    cudaMemcpyAsync(hadjustRetvalsH.get(), hadjustRetvalsD.get(), sizeof(int) * nActiveSystemsTimeloop, D2H, updateStream1); CUERR;
                    cudaStreamSynchronize(updateStream1); CUERR;

                    for(size_t i = 0; i < nActiveSystemsTimeloop; i++){
                        const size_t si = activeSystemsTimeloopH.get()[i];
                        if(hadjustRetvalsH.get()[si] == -1){
                            /* system i failed */
                            cudaMemcpyAsync(ode_system + si * systemsize,
                                y0evolve.get() + si * systemsize,
                                sizeof(double) * systemsize,
                                D2D,
                                copyStreams[i % nCopyStreams]); CUERR;

                            stats[si].status = Status::failure;

                            // insert si into sorted failedSystems vector
                            const auto upperbound = std::upper_bound(failedSystems.begin(), failedSystems.end(), si);
                            if(upperbound == failedSystems.cend())
                                failedSystems.push_back(si);
                            else
                                failedSystems.insert(upperbound, si);

                            activeSystemsListNeedsUpdate = true;
                        }else{
                            /* everything is good */
                            tnowH.get()[si] += hstepH.get()[si];

                            stats[si].steps++;
                        }
                    }
                    cudaMemcpyAsync(tnowD.get(), tnowH.get(), sizeof(double) * nActiveSystemsTimeloop, H2D, copyStreams[0]); CUERR;

                    for(size_t i = 0; i < nCopyStreams; i++)
                        cudaStreamSynchronize(copyStreams[i]); CUERR;

                    if(activeSystemsListNeedsUpdate){
                        // remove every element in failedSystems from activeSystems
                        std::set_difference (activeSystemsTimeloopH.get(),
                                            activeSystemsTimeloopH.get() + nActiveSystemsTimeloop,
                                            failedSystems.begin(),
                                            failedSystems.end(),
                                            newActiveSystems.begin());

                        nActiveSystemsTimeloop = nSystems - failedSystems.size();

                        std::copy(newActiveSystems.begin(), newActiveSystems.end(), activeSystemsTimeloopH.get());
                        cudaMemcpyAsync(activeSystemsTimeloopD.get(),
                                        activeSystemsTimeloopH.get(),
                                        sizeof(size_t) * nActiveSystemsTimeloop,
                                        H2D,
                                        copyStreams[0]); CUERR;
                        cudaStreamSynchronize(copyStreams[0]); CUERR;

                        activeSystemsListNeedsUpdate = false;
                    }

                    if(showProgress)
                                updateProgress((double)step, nSteps);
                }

                if(showProgress)
                    updateProgress(1.0, 1.0);
            }

            void solveAdaptive(){
                size_t nActiveSystemsTimeloop = 0;
                size_t nActiveSystemsHloop = 0;

                std::vector<int> signHost(nSystems, 0);
                std::vector<bool> final_stepHost(nSystems, 0);
                std::vector<double> dt(nSystems, 0);

                // this is used to determine current progress
                double totalTEnd = 0.0;

                for(size_t i = 0; i < nSystems; i++){

                    hstepH.get()[i] = h;
                    tnowH.get()[i] = t_begin[i];

                    if (hstepH.get()[i] > 0.0){
                        signHost[i] = 1;
                    }else{
                        signHost[i] = -1;
                    }
                    if (signHost[i] * (t_end[i] - t_begin[i]) < 0.0){
                        printf("system %lu, integration limits and/or step direction not consistent\n", i);
                        stats[i].status = Status::failure;
                    }
                    // make a list of systems which need to be calculated, i.e [0, 1, 2, ..., nSystems-1]
                    if (signHost[i] * (t_end[i] - t_begin[i]) > 0.0){
                        activeSystemsTimeloopH.get()[nActiveSystemsTimeloop] = i;
                        nActiveSystemsTimeloop++;
                    }

                    totalTEnd += t_end[i];
                }

                cudaMemcpyAsync(tnowD.get(), tnowH.get(), sizeof(double) * nSystems, H2D, copyStreams[1]); CUERR;
                cudaMemcpyAsync(hstepD.get(), hstepH.get(), sizeof(double) * nSystems, H2D, copyStreams[2]); CUERR;

                cudaStreamSynchronize(copyStreams[1]); CUERR;
                cudaStreamSynchronize(copyStreams[2]); CUERR;

                bool isFirstIter = true;

                if(showProgress)
                    updateProgress(0.0, 1.0);

                // loop until every system is calculated
                while(nActiveSystemsTimeloop > 0){

                    std::copy(tnowH.get(), tnowH.get() + nSystems, t0H.get());
                    cudaMemcpyAsync(t0D.get(), tnowD.get(), sizeof(double) * nSystems, D2D, copyStreams[0]); CUERR;

                    std::copy(hstepH.get(), hstepH.get() + nSystems, h0H.get());
                    cudaMemcpyAsync(h0D.get(), hstepD.get(), sizeof(double) * nSystems, D2D, copyStreams[1]); CUERR;

                    // calculate remaining time for each active system
                    for(size_t i = 0; i < nActiveSystemsTimeloop; i++){
                            const size_t si = activeSystemsTimeloopH.get()[i];
                            dt[si] = t_end[si] - t0H.get()[si];
                    }

                    // copy each active system to y0evolve.
                    for(size_t i = 0; i < nActiveSystemsTimeloop; i++){

                        // group copy operations of successive systems into single copy operations
                        size_t begin = activeSystemsTimeloopH.get()[i];
                        size_t copyBatch = 1;
                        while(i+1 < nActiveSystemsTimeloop && activeSystemsTimeloopH.get()[i+1] == activeSystemsTimeloopH.get()[i] + 1){
                            copyBatch++;
                            i++;
                        }

                        cudaMemcpyAsync(y0evolve.get() + begin * systemsize,
                                        ode_system + begin * systemsize,
                                        sizeof(double) * systemsize * copyBatch,
                                        D2D,
                                        copyStreams[i % nCopyStreams]); CUERR;
                    }

                    cudaMemcpyAsync(activeSystemsTimeloopD.get(),
                                    activeSystemsTimeloopH.get(),
                                    sizeof(size_t) * nActiveSystemsTimeloop,
                                    H2D,
                                    copyStreams[0]);

                    if(isFirstIter){
                        for(size_t i = 0; i < nCopyStreams; i++)
                            cudaStreamSynchronize(copyStreams[i]); CUERR;

                        stepfunc(activeSystemsTimeloopD.get(), nActiveSystemsTimeloop, t0D.get(), ode_system, dydt_in.get(), userdata);

                        isFirstIter = false;
                    }else{
                        // in other iterations we can reuse the data which was needed by step size control

                        // copy the data of active systems to dydt_in
                        for(size_t i = 0; i < nActiveSystemsTimeloop; i++){

                            size_t begin = activeSystemsTimeloopH.get()[i];
                            size_t copyBatch = 1;
                            while(i+1 < nActiveSystemsTimeloop && activeSystemsTimeloopH.get()[i+1] == activeSystemsTimeloopH.get()[i] + 1){
                                copyBatch++;
                                i++;
                            }

                            cudaMemcpyAsync(dydt_in.get() + begin * systemsize,
                                            dydt_out.get() + begin * systemsize,
                                            sizeof(double) * systemsize * copyBatch,
                                            D2D,
                                            copyStreams[i % nCopyStreams]); CUERR;
                        }
                        for(size_t i = 0; i < nCopyStreams; i++)
                            cudaStreamSynchronize(copyStreams[i]); CUERR;
                    }

                    std::copy(activeSystemsTimeloopH.get(), activeSystemsTimeloopH.get() + nSystems, activeSystemsHloopH.get());

                    nActiveSystemsHloop = nActiveSystemsTimeloop;

                    // "h loop". loop until every step was calculated without decreasing the step size h
                    do{

                        /*
                            For each active system, decrease step size if neccessary so we won't get beyond t_end.
                            If this decrease is needed, we know this will be the last step, assuming the step succeeds
                        */

                        for(size_t i = 0; i < nActiveSystemsHloop; i++){

                            const size_t si = activeSystemsHloopH.get()[i];

                            if ((dt[si] >= 0.0 && h0H.get()[si] > dt[si]) || (dt[si] < 0.0 && h0H.get()[si] < dt[si])){
                                h0H.get()[si] = dt[si];
                                final_stepHost[si] = true;
                            }else{
                                final_stepHost[si] = false;
                            }
                            stats[si].steps++;
                            holdH.get()[si] = h0H.get()[si];
                        }

                        cudaMemcpyAsync(h0D.get(),
                                        h0H.get(),
                                        sizeof(double) * nSystems,
                                        H2D,
                                        copyStreams[0]); CUERR;

                        cudaMemcpyAsync(activeSystemsHloopD.get(),
                                        activeSystemsHloopH.get(),
                                        sizeof(size_t) * nActiveSystemsHloop,
                                        H2D,
                                        copyStreams[1]); CUERR;

                        cudaStreamSynchronize(copyStreams[0]); CUERR;
                        cudaStreamSynchronize(copyStreams[1]); CUERR;

                        stepper.step_apply(activeSystemsHloopH.get(), activeSystemsHloopD.get(),
                                            nActiveSystemsHloop, t0D.get(), h0D.get(),
                                            ode_system, error.get(), dydt_in.get(), dydt_out.get());

                        for(size_t i = 0; i < nActiveSystemsHloop; i++){
                            const size_t si = activeSystemsHloopH.get()[i];

                            if (final_stepHost[si]){
                                tnowH.get()[si] = t_end[si];
                            }else{
                                tnowH.get()[si] = t0H.get()[si] + h0H.get()[si];
                            }
                        }

                        cudaMemcpyAsync(tnowD.get(), tnowH.get(), sizeof(double) * nSystems, H2D, copyStreams[1]); CUERR;
                        cudaStreamSynchronize(copyStreams[1]); CUERR;

                        callSolver2dHadjustKernelAsync(
                                hadjustRetvalsD.get(), Stepper::getOrder(),
                                activeSystemsHloopD.get(), nActiveSystemsHloop,
                                ode_system, error.get(), dydt_out.get(),
                                h0D.get(), systemsize, eps_abs, eps_rel, dimx, dimy, pitchx,
                                dim3(1, nSystems, 1), dim3(256, 1, 1), updateStream1);

                        cudaMemcpyAsync(hadjustRetvalsH.get(), hadjustRetvalsD.get(), sizeof(int) * nSystems, D2H, updateStream1);
                        cudaMemcpyAsync(h0H.get(), h0D.get(), sizeof(double) * nSystems, D2H, updateStream1);

                        cudaStreamSynchronize(updateStream1); CUERR;

                        int nActiveSystemsHloopNew = 0;
                        std::vector<size_t> activeSystemsHloopHNew(nSystems, 0);

                        /* for each system check if the step size was decreased*/
                        for(size_t i = 0; i < nActiveSystemsHloop; i++){

                            const size_t si = activeSystemsHloopH.get()[i];

                            if (hadjustRetvalsH.get()[i] == -1)
                            {
                                const double t_curr = tnowH.get()[si];
                                const double t_next = tnowH.get()[si] + h0H.get()[si];

                                if (fabs (h0H.get()[si]) < fabs (holdH.get()[si]) && t_next != t_curr)
                                {
                                    // reset system
                                    cudaMemcpyAsync(ode_system + si * systemsize,
                                                    y0evolve.get() + si * systemsize,
                                                    sizeof(double) * systemsize,
                                                    D2D,
                                                    copyStreams[i % nCopyStreams]); CUERR;

                                    stats[si].repeats++;

                                    // add system id to active systems for next try
                                    activeSystemsHloopHNew[nActiveSystemsHloopNew] = si;
                                    nActiveSystemsHloopNew++;
                                }
                                else
                                {
                                    stats[si].status = Status::failure;
                                }
                            }else{
                                /* no change, or increase, everything is good */
                            }

                        }

                        std::copy(activeSystemsHloopHNew.begin(), activeSystemsHloopHNew.end(), activeSystemsHloopH.get());
                        nActiveSystemsHloop = nActiveSystemsHloopNew;

                        for(size_t i = 0; i < nCopyStreams; i++)
                            cudaStreamSynchronize(copyStreams[i]); CUERR;

                    }while(nActiveSystemsHloop > 0); // H loop end

                    //for each active system which performed a step check that step size does not exceed h_max or h_min

                    for(size_t i = 0; i < nActiveSystemsTimeloop; i++){
                        const size_t si = activeSystemsTimeloopH.get()[i];

                        if(!final_stepHost[si])
                            hstepH.get()[si] = h0H.get()[si];

                        if(fabs(hstepH.get()[si]) < h_min)
                            stats[si].status = Status::failure;	//failure

                        if(fabs(hstepH.get()[si]) > h_max)
                            hstepH.get()[si] = signHost[si] * h_max;

                    }

                    nActiveSystemsTimeloop = 0;

                    // update list of systems which need to be calculated for the next time loop iteration
                    for(size_t i = 0; i < nSystems; i++){
                        // if system did not reach end yet
                        if(signHost[i] * (t_end[i] - tnowH.get()[i]) > 0.0){
                            activeSystemsTimeloopH.get()[nActiveSystemsTimeloop] = i;
                            nActiveSystemsTimeloop++;
                        }
                    }

                    if(showProgress){
                        double totalTNow = 0.0;
                        for(size_t i = 0; i < nSystems; i++){
                            totalTNow += tnowH.get()[i];
                        }
                        updateProgress((double)totalTNow, totalTEnd);
                    }


                } // Time loop end

                if(showProgress)
                    updateProgress(1.0, 1.0);
            }

        };

	}
}


#endif
