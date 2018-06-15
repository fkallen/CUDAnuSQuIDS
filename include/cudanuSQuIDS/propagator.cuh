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

#ifndef CUDANUSQUIDS_PROPAGATOR_CUH
#define CUDANUSQUIDS_PROPAGATOR_CUH

#include <cudanuSQuIDS/bodygpu.cuh>
#include <cudanuSQuIDS/const.cuh>
#include <cudanuSQuIDS/cudahelpers.cuh>
#include <cudanuSQuIDS/cudautils.cuh>
#include <cudanuSQuIDS/cuda_unique.cuh>
#include <cudanuSQuIDS/interaction_structure.hpp>
#include <cudanuSQuIDS/ode.cuh>
#include <cudanuSQuIDS/parameterobject.hpp>
#include <cudanuSQuIDS/physics.cuh>
#include <cudanuSQuIDS/physics_kernels.cuh>
#include <cudanuSQuIDS/propagator_kernels.cuh>
#include <cudanuSQuIDS/types.hpp>

#include <nuSQuIDS/nuSQuIDS.h>


#include <vector>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <iterator>
#include <iomanip>
#include <numeric>
#include <thread>
#include <chrono>
#include <map>

namespace cudanusquids{

	template<int NFLVS, class body_t, class Ops>
	class CudaNusquids;

    struct PropagatorImpl{

        int NFLVS = 3;
        int susize = NFLVS * NFLVS;

        detail::EvalType previousEvalType = detail::EvalType::None;

        std::shared_ptr<ParameterObject> parameterObject;

        //input data
        std::vector<double> initialFlux;

    //final states

        unique_pinned_ptr<double> finalStatesHost;

        std::vector<double> results;

        unique_dev_ptr<double> energyListgpu;
        unique_dev_ptr<double> initialFluxgpu;
        unique_dev_ptr<double> DM2datagpu;
        unique_dev_ptr<double> H0_arraygpu;
        unique_dev_ptr<double> delEgpu;
        unique_dev_ptr<double> B0projgpu;
        unique_dev_ptr<double> B1projgpu;
        unique_dev_ptr<double> evolB1projgpu;
        unique_dev_ptr<double> statesgpu;

//interaction data
        unique_dev_ptr<double> gr_arraygpu;
        unique_dev_ptr<double> nc_arraygpu;
        unique_dev_ptr<double> tau_hadlep_arraygpu;
        unique_dev_ptr<double> tau_lep_arraygpu;
        unique_dev_ptr<double> tau_decay_fluxesgpu;
        unique_dev_ptr<double> tau_bar_decay_fluxesgpu;
        unique_dev_ptr<double> currentFluxes; //[cosine, rho, flavor, energy]

        InteractionStructureGpu intstructgpu;

        InteractionStateBufferGpu intstates;
        std::shared_ptr<InteractionStructure> intstructcpu;

        unique_dev_ptr<void*> additionalDataPointersGpu;
        std::vector<unique_dev_ptr<char>> additionalDatagpuvec;

        // memory pitch in bytes for energy dimension
        size_t currentFluxesPitch;
        size_t h0pitch;
        size_t evolB1pitch;
        size_t statespitch;
// distance between two consecutive elements of the same density matrix
        int fluxOffset;
		int h0offset;
        int evolB1offset;
        int statesOffset;
        int b0offset;
        int b1offset;

        int deviceId;

        // ODE solver statistics
        unique_pinned_ptr<ode::RKstats> rkstatsHost;
        unique_dev_ptr<ode::RKstats> rkstatsgpu;

    // Problem dimensions
        int n_cosines = 0;
    // Maximum number of paths with parameterObject->GetNumE() energy nodes that can be calculated in parallel (hardware limit, e.g. available memory).
    //Larger problems will be batched
        int max_n_cosines = 0;

    // parameters which are used in the current derivation step

        const double* timesgpu;
        double* derivationInputgpu;
        double* derivationOutputgpu;
        const size_t* activePathsgpu;
        size_t nPaths;

    // initialization flags
        bool isInit_intstruct = false;
        bool interactionsAreInitialized = false;
        bool interactionArraysAreAllocated = false;

    // gpu streams

        static constexpr int nCopyStreams = 2;

        cudaStream_t updateTimeStream;
        cudaStream_t evolveProjectorsStream;
        cudaStream_t updateIntStructStream;
        cudaStream_t updateNCarraysStream;
        cudaStream_t updateTauarraysStream;
        cudaStream_t updateGRarraysStream;
        cudaStream_t calculateFluxStream;

        cudaStream_t hiStream;
        cudaStream_t gammaRhoStream;
        cudaStream_t interactionsRhoStream;

        cudaStream_t copyStreams[nCopyStreams];

        PropagatorImpl(int deviceId_,
                       int nflvs_,
                        int n_cosines_,
                        int batchsizeLimit,
                        std::shared_ptr<ParameterObject>& params)
                    : parameterObject(params),
                      NFLVS(nflvs_),
                      susize(NFLVS * NFLVS),
                      deviceId(deviceId_),
                      n_cosines(n_cosines_),
                      max_n_cosines(std::min(batchsizeLimit, n_cosines)){

            int nGpus;
            cudaGetDeviceCount(&nGpus); CUERR;

            if(deviceId < 0 || deviceId >= nGpus)
                throw std::runtime_error("Propagator : invalid device id");

            cudaSetDevice(deviceId); CUERR;
            // create streams
            cudaStreamCreate(&updateTimeStream); CUERR;
            cudaStreamCreate(&evolveProjectorsStream); CUERR;
            cudaStreamCreate(&updateIntStructStream); CUERR;
            cudaStreamCreate(&updateNCarraysStream); CUERR;
            cudaStreamCreate(&updateTauarraysStream); CUERR;
            cudaStreamCreate(&updateGRarraysStream); CUERR;
            cudaStreamCreate(&calculateFluxStream); CUERR;
            cudaStreamCreate(&hiStream); CUERR;
            cudaStreamCreate(&gammaRhoStream); CUERR;
            cudaStreamCreate(&interactionsRhoStream); CUERR;

            for(int i = 0; i < nCopyStreams; i++)
                cudaStreamCreate(&copyStreams[i]); CUERR;

            results.resize(NFLVS * n_cosines * parameterObject->GetNumRho() * parameterObject->GetNumE());
            initialFlux.resize(n_cosines  * parameterObject->GetNumE() * parameterObject->GetNumRho() * NFLVS);

            finalStatesHost = make_unique_pinned<double>(n_cosines * parameterObject->GetNumRho() * parameterObject->GetNumE() * susize);

            energyListgpu = make_unique_dev<double>(deviceId, parameterObject->GetNumE());
            initialFluxgpu = make_unique_dev<double>(deviceId, max_n_cosines  * parameterObject->GetNumE() * parameterObject->GetNumRho() * NFLVS);

            DM2datagpu = make_unique_dev<double>(deviceId, susize);

            {
                double* ptr;
                cudaMallocPitch(&ptr, &h0pitch, sizeof(double) * parameterObject->GetNumE(), susize); CUERR;
                H0_arraygpu = make_unique_dev<double>(deviceId, ptr);
            }

            B0projgpu = make_unique_dev<double>(deviceId, susize * NFLVS);
            B1projgpu = make_unique_dev<double>(deviceId, susize * NFLVS * parameterObject->GetNumRho());

            {
                double* ptr;
                cudaMallocPitch(&ptr, &evolB1pitch, sizeof(double) * parameterObject->GetNumE(), max_n_cosines * parameterObject->GetNumRho() * NFLVS * susize); CUERR;
                evolB1projgpu = make_unique_dev<double>(deviceId, ptr);
            }

            {
                double* ptr;
                cudaMallocPitch(&ptr, &statespitch, sizeof(double) * parameterObject->GetNumE(), max_n_cosines * parameterObject->GetNumRho() * susize); CUERR;
                statesgpu = make_unique_dev<double>(deviceId, ptr);

                cudaMemset(statesgpu.get(), 0, max_n_cosines * parameterObject->GetNumRho() * susize * statespitch);
            }

            {
                double* ptr;
                cudaMallocPitch(&ptr, &currentFluxesPitch, sizeof(double) * parameterObject->GetNumE(), max_n_cosines * parameterObject->GetNumRho() * NFLVS); CUERR;
                currentFluxes = make_unique_dev<double>(deviceId, ptr);
            }

            if(statespitch % sizeof(double) != 0)
                throw std::runtime_error("error. unexpected gpu memory padding on states");

            if(evolB1pitch % sizeof(double) != 0)
                throw std::runtime_error("error. unexpected gpu memory padding on evolb1projectors");

            if(h0pitch % sizeof(double) != 0)
                throw std::runtime_error("error. unexpected gpu memory padding on H0 array");

            if(currentFluxesPitch % sizeof(double) != 0)
                throw std::runtime_error("error. unexpected gpu memory padding on current fluxes");

            statesOffset = statespitch / sizeof(double);
            evolB1offset = evolB1pitch / sizeof(double);
            h0offset = h0pitch / sizeof(double);
            b0offset = NFLVS;
            b1offset = parameterObject->GetNumRho() * NFLVS;
			fluxOffset = currentFluxesPitch / sizeof(double);

            delEgpu = make_unique_dev<double>(deviceId, parameterObject->GetNumE());

            //copy energy list to gpu
            auto energyList = parameterObject->GetERange();
            cudaMemcpyAsync(energyListgpu.get(), &energyList[0], sizeof(double) * parameterObject->GetNumE(), H2D, copyStreams[0]); CUERR;
            std::vector<double> delE(parameterObject->GetNumE(), 0.0);

            for(int i = 0; i < parameterObject->GetNumE()-1; i++)
                delE[i+1] = energyList[i+1] - energyList[i];

            cudaMemcpyAsync(delEgpu.get(), delE.data(), sizeof(double) * (parameterObject->GetNumE()), H2D, copyStreams[0]); CUERR;

            cudaStreamSynchronize(copyStreams[0]);

            if(!interactionArraysAreAllocated && parameterObject->Get_CanUseInteractions()){
                intstates = make_InteractionStateBufferGpu(deviceId, parameterObject->GetNumRho(), NFLVS, parameterObject->GetNumE(), max_n_cosines);
                gr_arraygpu = make_unique_dev<double>(deviceId, max_n_cosines * parameterObject->GetNumE());
                nc_arraygpu = make_unique_dev<double>(deviceId, max_n_cosines * parameterObject->GetNumRho() * NFLVS * parameterObject->GetNumE());
                tau_hadlep_arraygpu = make_unique_dev<double>(deviceId, max_n_cosines * parameterObject->GetNumRho() * parameterObject->GetNumE());
                tau_lep_arraygpu = make_unique_dev<double>(deviceId, max_n_cosines * parameterObject->GetNumRho() * parameterObject->GetNumE());
                tau_decay_fluxesgpu = make_unique_dev<double>(deviceId, max_n_cosines * parameterObject->GetNumE());
                tau_bar_decay_fluxesgpu = make_unique_dev<double>(deviceId, max_n_cosines * parameterObject->GetNumE());

                interactionArraysAreAllocated = true;
            }

            rkstatsHost = make_unique_pinned<ode::RKstats>(n_cosines); CUERR;
            rkstatsgpu = make_unique_dev<ode::RKstats>(deviceId, max_n_cosines); CUERR;

        }

        ~PropagatorImpl(){
            cudaSetDevice(deviceId); CUERR;

            cudaStreamDestroy(updateTimeStream); CUERR;
            cudaStreamDestroy(evolveProjectorsStream); CUERR;
            cudaStreamDestroy(updateIntStructStream); CUERR;
            cudaStreamDestroy(updateNCarraysStream); CUERR;
            cudaStreamDestroy(updateTauarraysStream); CUERR;
            cudaStreamDestroy(updateGRarraysStream); CUERR;
            cudaStreamDestroy(calculateFluxStream); CUERR;
            cudaStreamDestroy(hiStream); CUERR;
            cudaStreamDestroy(gammaRhoStream); CUERR;
            cudaStreamDestroy(interactionsRhoStream); CUERR;

            for(size_t i = 0; i < nCopyStreams; i++)
                cudaStreamDestroy(copyStreams[i]); CUERR;

            if(isInit_intstruct){
                destroy_InteractionStructureGpu(intstructgpu);
                destroy_InteractionStateBufferGpu(intstates);
            }
        }

        template<class Physics>
        void init_host_physics(Physics* physics){
            // let physics objects point to correct device memory
            for(int i = 0; i < n_cosines; i++){

                int indexInBatch = i % max_n_cosines;

                // problem dimensions
                physics[i].set_max_n_cosines(max_n_cosines);
                physics[i].set_n_rhos(parameterObject->GetNumRho());
                physics[i].set_n_energies(parameterObject->GetNumE());
                physics[i].set_indexInBatch(indexInBatch);
                physics[i].set_globalPathId(i);

                physics[i].set_basis(parameterObject->getBasis());
                physics[i].set_neutrinoType(parameterObject->getNeutrinoType());

                // variables for data access
                physics[i].set_b0offset(b0offset);
                physics[i].set_b1offset(b1offset);
                physics[i].set_evolB1pitch(evolB1pitch);
                physics[i].set_evoloffset(evolB1offset);
                physics[i].set_h0pitch(h0pitch);
                physics[i].set_h0offset(h0offset);
                physics[i].set_statesOffset(statesOffset);
                physics[i].set_statesPitch(statespitch);
				physics[i].set_fluxOffset(fluxOffset);
                physics[i].set_fluxPitch(currentFluxesPitch);

                // data arrays
                physics[i].set_states(statesgpu.get() + indexInBatch * statespitch / sizeof(double) * parameterObject->GetNumRho() * susize);
                physics[i].set_b0proj(B0projgpu.get());
                physics[i].set_b1proj(B1projgpu.get());
                physics[i].set_evolB1proj(evolB1projgpu.get() + indexInBatch * evolB1pitch / sizeof(double) * parameterObject->GetNumRho() * NFLVS * susize);
                physics[i].set_H0_array(H0_arraygpu.get());
                physics[i].set_dm2(DM2datagpu.get());
                physics[i].set_energyList(energyListgpu.get());
                physics[i].set_delE(delEgpu.get());
                physics[i].set_fluxes(currentFluxes.get() + indexInBatch * currentFluxesPitch / sizeof(double) * parameterObject->GetNumRho() * NFLVS);

                if(parameterObject->Get_CanUseInteractions()){
                    physics[i].set_nc_array(nc_arraygpu.get() + indexInBatch * parameterObject->GetNumRho() * NFLVS * parameterObject->GetNumE());
                    physics[i].set_tau_decay_fluxes(tau_decay_fluxesgpu.get() + indexInBatch * parameterObject->GetNumE());
                    physics[i].set_tau_bar_decay_fluxes(tau_bar_decay_fluxesgpu.get() + indexInBatch * parameterObject->GetNumE());
                    physics[i].set_tau_hadlep_array(tau_hadlep_arraygpu.get() + indexInBatch * parameterObject->GetNumRho() * parameterObject->GetNumE());
                    physics[i].set_tau_lep_array(tau_lep_arraygpu.get() + indexInBatch * parameterObject->GetNumRho() * parameterObject->GetNumE());
                    physics[i].set_gr_array(gr_arraygpu.get() + indexInBatch * parameterObject->GetNumE());
                    physics[i].set_intstate(intstates[indexInBatch]);
                }
            }
        }

        /*
           Functions to evolve systems
        */

        template<class Physics, class Track>
        void evolve(Physics* cpuphysics,
                    Physics* gpuphysics,
                    Track* tracks){
            previousEvalType = detail::EvalType::None; //invalidate results of previous simulation

            switch(parameterObject->Get_SolverType()){
                case SolverType::Version1:
                            evolveVersion1(cpuphysics,
                                           gpuphysics,
                                           tracks);
                            break;
                case SolverType::Version2:
                            evolveVersion2(cpuphysics,
                                           gpuphysics,
                                           tracks);
                            break;
                default: throw std::runtime_error("Propagator::evolve: invalid solverType.");
            }
        }

        template<class Physics, class Track>
        void evolveVersion1(Physics* cpuphysics,
                            Physics* gpuphysics,
                            Track* tracks){
            switch(parameterObject->Get_StepperType()){
            case ode::StepperType::RK4:
                evolveVersion1_impl<ode::stepper::RK42D>(cpuphysics,
                                                         gpuphysics,
                                                         tracks);
                break;
            default:
                throw std::runtime_error("Unknown stepper type");
            };
        }

        template<class Physics, class Track>
        void evolveVersion2(Physics* cpuphysics,
                            Physics* gpuphysics,
                            Track* tracks){
            switch(parameterObject->Get_StepperType()){
            case ode::StepperType::RK4:
                evolveVersion2_impl<ode::stepper::RK4>(cpuphysics,
                                                       gpuphysics,
                                                       tracks);
                break;
            default:
                throw std::runtime_error("Unknown stepper type");
            };

        }

        template<class Stepper, class Physics, class Track>
        void evolveVersion1_impl(Physics* cpuphysics,
                                 Physics* gpuphysics,
                                 Track* tracks){
            cudaSetDevice(deviceId); CUERR;

            struct odehelper{
                PropagatorImpl* impl;
                Physics* gpuphysics;
            };

            odehelper helper{this, gpuphysics};

            // interface to ODE solver
            auto RHS = [](const size_t* const activeIndices, size_t nIndices, const double* const t,
                                double* const y, double* const y_derived, void* userdata){

                const odehelper* helper = static_cast<const odehelper*>(userdata);

                PropagatorImpl* nsq = helper->impl;
                nsq->timesgpu = t;
                nsq->derivationInputgpu = y;
                nsq->derivationOutputgpu = y_derived;
                nsq->activePathsgpu = activeIndices;
                nsq->nPaths = nIndices;
                nsq->derive(helper->gpuphysics);
            };

            // progress function
            auto progressFunc = [](double p, double mp){
                        double frac = 100.0 * p / mp;
                        printf("\r%3.6f %%", frac);
                        if(frac >= 100.0)
                            printf("\n");
            };

            beforeEvolution(cpuphysics, tracks); CUERR;

            std::vector<double> t_begin(max_n_cosines);
            std::vector<double> t_end(max_n_cosines);

            const int iters = SDIV(n_cosines, max_n_cosines);

            for(int currentCosineBatch = 0; currentCosineBatch < iters; currentCosineBatch++){
                const int batchsize = currentCosineBatch < iters - 1 ? max_n_cosines : n_cosines - currentCosineBatch * max_n_cosines;

                beforeBatchEvolution(cpuphysics,
                                     gpuphysics,
                                     tracks,
                                     currentCosineBatch);

                // get begin and end of current paths
                for(int i = 0; i < batchsize; i++){
                    t_begin[i] = tracks[currentCosineBatch * max_n_cosines + i].getXBegin();
                    t_end[i] = tracks[currentCosineBatch * max_n_cosines + i].getXEnd();
                }

                //set up ODE solver
                ode::Solver2D<Stepper> solver(deviceId, statesgpu.get() , statespitch,
                                                parameterObject->GetNumE(), NFLVS * NFLVS * parameterObject->GetNumRho(), batchsize,
                                                parameterObject->Get_NumSteps(),
                                                parameterObject->Get_h(), parameterObject->Get_h_min(), parameterObject->Get_h_max(),
                                                t_begin, t_end, &helper,
                                                RHS, // step function
                                                progressFunc);

                solver.setShowProgress(parameterObject->Get_ProgressBar());CUERR;
                solver.setAbsolutePrecision(parameterObject->Get_abs_error());CUERR;
                solver.setRelativePrecision(parameterObject->Get_rel_error());CUERR;

                if(parameterObject->Get_NumSteps() == 0){
                    solver.solveAdaptive();CUERR;
                }else{
                    solver.solveFixed();CUERR;
                }

                // save runge kutta stats
                std::copy(solver.stats.begin(), solver.stats.end(), rkstatsHost.get() + currentCosineBatch * max_n_cosines);CUERR;

                afterBatchEvolution(cpuphysics,
                                    gpuphysics,
                                    tracks,
                                    currentCosineBatch);
            }
        }


        template<class Stepper, class Physics, class Track>
        void evolveVersion2_impl(Physics* cpuphysics,
                                 Physics* gpuphysics,
                                 Track* tracks){
            cudaSetDevice(deviceId); CUERR;

            const size_t workspacePerSolver = ode::SolverGPU<Stepper>::getMinimumMemorySize(statespitch, NFLVS * NFLVS * parameterObject->GetNumRho());
            auto solverworkspace = make_unique_dev<double>(deviceId, workspacePerSolver * max_n_cosines); CUERR;

            beforeEvolution(cpuphysics, tracks);

            int evolveBlockDimx = 128;
            int evolveGridDimx = 1;

            const int iters = SDIV(n_cosines, max_n_cosines);

            for(int currentCosineBatch = 0; currentCosineBatch < iters; currentCosineBatch++){
                const int batchsize = currentCosineBatch < iters - 1 ? max_n_cosines : n_cosines - currentCosineBatch * max_n_cosines;

                beforeBatchEvolution(cpuphysics,
                                     gpuphysics,
                                     tracks,
                                     currentCosineBatch);

                dim3 evolveblock(evolveBlockDimx, 1, 1);
                dim3 evolvegrid(evolveGridDimx, batchsize, 1);

                callEvolveKernel(gpuphysics , batchsize, !parameterObject->Get_CanUseInteractions(),
                                rkstatsgpu.get(), parameterObject->Get_NumSteps(),
                                parameterObject->Get_h(), parameterObject->Get_h_min(), parameterObject->Get_h_max(),
                                parameterObject->Get_abs_error(), parameterObject->Get_rel_error(),
                                parameterObject->Get_StepperType(), solverworkspace.get(),
                                evolvegrid, evolveblock, hiStream);
                cudaStreamSynchronize(hiStream);
                CUERR;

                // copy runge kutta stats to host
                cudaMemcpy(rkstatsHost.get() + currentCosineBatch * max_n_cosines,
                                rkstatsgpu.get(),
                                sizeof(ode::RKstats) * batchsize,
                                D2H); CUERR;

                afterBatchEvolution(cpuphysics,
                                    gpuphysics,
                                    tracks,
                                    currentCosineBatch);
            }
        }


        /*
            Evolution helper functions
        */

        template<class Physics, class Track>
        void beforeEvolution(Physics* cpuphysics, Track* cputracks){
            if(parameterObject->Get_CanUseInteractions()){
                initializeInteractions(cpuphysics); CUERR;
            }

            //set track position to start
            for(int i = 0; i < n_cosines; i++){
                const double begin = cputracks[i].getXBegin();
                cputracks[i].setCurrentX(begin);
            }

            updatePhysicsObject(cpuphysics); CUERR;
        }

        template<class Physics>
        void updatePhysicsObject(Physics* cpuphysics){
            for(int i = 0; i < n_cosines; i++){
                cpuphysics[i].set_flags(parameterObject->getFlags());
                cpuphysics[i].set_intstruct(intstructgpu);
            }
        }

        template<class Physics, class Track>
        void beforeBatchEvolution(Physics* cpuphysics,
                                  Physics* gpuphysics,
                                  Track* tracks,
                                  int currentCosineBatch){
            const int iters = SDIV(n_cosines, max_n_cosines);
            const int batchsize = currentCosineBatch < iters - 1 ? max_n_cosines : n_cosines - currentCosineBatch * max_n_cosines;

            // set up the current batch

            // set correct tracks in physics objects
            for(int i = 0; i < batchsize; i++){
                cpuphysics[currentCosineBatch * max_n_cosines + i].set_track(tracks[currentCosineBatch * max_n_cosines + i]);
            }

            cudaMemcpyAsync(gpuphysics,cpuphysics + currentCosineBatch * max_n_cosines, sizeof(Physics) * batchsize, H2D, hiStream); CUERR;

            // copy correct initial flux to gpu
            cudaMemcpyAsync(initialFluxgpu.get(),
                    initialFlux.data() + currentCosineBatch * max_n_cosines * parameterObject->GetNumRho() * parameterObject->GetNumE() * NFLVS,
                    sizeof(double) * batchsize * parameterObject->GetNumRho() * parameterObject->GetNumE() * NFLVS,
                    H2D, hiStream); CUERR;

            dim3 block(128,1,1);
            dim3 grid(SDIV(parameterObject->GetNumE(), block.x), batchsize, 1);

            // initialize the states from flux
            callSetInitialStatesKernel(statesgpu.get(), initialFluxgpu.get(), B0projgpu.get(), B1projgpu.get(),
                                batchsize, parameterObject->GetNumRho(), NFLVS, parameterObject->GetNumE(),
                                parameterObject->Get_FluxBasis(),
                                statesOffset, statespitch, grid, block, hiStream);

            cudaStreamSynchronize(hiStream);
        }


        template<class Physics, class Track>
        void afterBatchEvolution(Physics* cpuphysics,
                                  Physics* gpuphysics,
                                  Track* tracks,
                                  int currentCosineBatch){
            const int iters = SDIV(n_cosines, max_n_cosines);
            const int batchsize = currentCosineBatch < iters - 1 ? max_n_cosines : n_cosines - currentCosineBatch * max_n_cosines;
            CUERR;
            //copy solved states from gpu to cpu
            cudaMemcpy2DAsync(finalStatesHost.get() + currentCosineBatch * max_n_cosines * parameterObject->GetNumRho() * parameterObject->GetNumE() * NFLVS * NFLVS,
                parameterObject->GetNumE() * sizeof(double),
                statesgpu.get(),
                statespitch,
                parameterObject->GetNumE() * sizeof(double),
                batchsize * parameterObject->GetNumRho() * NFLVS * NFLVS,
                D2H,hiStream); CUERR;

            // copy Physics from gpu to cpu (to not override Physics state with next batch)
            cudaMemcpyAsync(cpuphysics + currentCosineBatch * max_n_cosines,
                            gpuphysics,
                            sizeof(Physics) * batchsize,
                            H2D,
                            hiStream); CUERR;

            // copy solved tracks to tracks array
            for(int i = 0; i < batchsize; i++){
                tracks[currentCosineBatch * max_n_cosines + i] = cpuphysics[currentCosineBatch * max_n_cosines + i].get_track();
            }

            cudaStreamSynchronize(hiStream);
        }



        /*
            Prederive functions for Version 1.
            Prederive functions for Version 2 are members of the physics objects and thus not defined in Propagator.
        */

        template<class Physics>
        void prederive(Physics* gpuphysics){

            setDerivationPointers(gpuphysics, derivationInputgpu, derivationOutputgpu, activePathsgpu, nPaths, updateTimeStream);
            updateTime(gpuphysics, timesgpu, activePathsgpu, nPaths, updateTimeStream);
            cudaStreamSynchronize(updateTimeStream); CUERR; //need correct time for projector evoluation
            evolveProjectorsAsync(gpuphysics, timesgpu, activePathsgpu, nPaths, evolveProjectorsStream);

			if(parameterObject->Get_CanUseInteractions()){

                cudaStreamSynchronize(updateTimeStream); CUERR; // need correct densities and electronfractions
                updateInteractionStructAsync(gpuphysics, timesgpu, activePathsgpu, nPaths, updateIntStructStream);

				if(parameterObject->Get_InteractionsRhoTerms()){
					cudaStreamSynchronize(evolveProjectorsStream); CUERR;
					cudaStreamSynchronize(updateIntStructStream); CUERR;
					updateInteractionArrays(gpuphysics);
				}

            }else{
                cudaStreamSynchronize(updateTimeStream); CUERR;
                cudaStreamSynchronize(evolveProjectorsStream); CUERR;
            }

            addToPrederive(gpuphysics);
        }

        template<class Physics>
        void setDerivationPointers(Physics* gpuphysics, double* y_in, double* y_out, const size_t* const activePaths, const size_t nPaths, cudaStream_t stream){
                dim3 block(1,1,1);
                dim3 grid;
                grid.x = 1;
                grid.y = SDIV(nPaths, block.y);
                grid.z = 1;

                callSetDerivationPointersKernel(activePaths, nPaths, gpuphysics, y_in, y_out, grid, block, stream);
        }

        template<class Physics>
        void updateTime(Physics* gpuphysics, const double* const times, const size_t* const activePaths, const size_t nPaths, cudaStream_t stream){
                dim3 block(1,1,1);
                dim3 grid;
                grid.x = 1;
                grid.y = SDIV(nPaths, block.y);
                grid.z = 1;

                callUpdateTimeKernel(times, activePaths, nPaths, gpuphysics, grid, block, stream);
        }

        template<class Physics>
        void evolveProjectorsAsync(Physics* gpuphysics, const double* const times, const size_t* const activePaths, const size_t nPaths, cudaStream_t stream){

            dim3 block(64,1,1);
            dim3 grid;
            grid.x = SDIV(parameterObject->GetNumE(), block.x);
            grid.y = SDIV(nPaths, 1);
            grid.z = 1;

            callEvolveProjectorsKernel(times, activePaths, nPaths,
                            gpuphysics, grid, block, stream);
        }

        template<class Physics>
        void updateInteractionStructAsync(Physics* gpuphysics, const double* const times, const size_t* const activePaths, const size_t nPaths,
                                                            cudaStream_t stream){
            dim3 block(64,1,1);
            dim3 grid;
            grid.x = SDIV(parameterObject->GetNumE(), block.x);
            grid.y = SDIV(nPaths, 1);
            grid.z = 1;

            cudaStreamSynchronize(updateTimeStream); CUERR;

            callUpdateInteractionStructKernel(activePaths, nPaths,
                                    gpuphysics, grid, block, stream);
        }

        template<class Physics>
        void calculateCurrentFlavorFluxesAsync(Physics* gpuphysics, const size_t* const d_activePaths, size_t nPaths_,
                                                    const double* const d_states, double* d_fluxes,
                                                    cudaStream_t stream){

            dim3 block(64,1,1);
            dim3 grid;
            grid.x = SDIV(parameterObject->GetNumE(), block.x);
            grid.y = SDIV(nPaths_, 1);
            grid.z = 1;

            callCalculateCurrentFlavorFluxesKernel(d_activePaths, nPaths_,
                                    gpuphysics, grid, block, stream);
        }

        template<class Physics>
        void updateInteractionArrays(Physics* gpuphysics){

            calculateCurrentFlavorFluxesAsync(gpuphysics, activePathsgpu, nPaths, derivationInputgpu, currentFluxes.get(), calculateFluxStream);
            cudaStreamSynchronize(calculateFluxStream); CUERR;

            if(parameterObject->getFlags().useNCInteractions){

                dim3 blockNC(64,1,1);
                dim3 gridNC;
                gridNC.x = SDIV(parameterObject->GetNumE(), blockNC.x);
                gridNC.y = SDIV(nPaths, 1);
                gridNC.z = 1;

                callUpdateNCArraysKernel(activePathsgpu, nPaths, gpuphysics, gridNC, blockNC, updateNCarraysStream);
            }

            if(parameterObject->getFlags().useTauRegeneration){

                dim3 blocktau1(128,1,1);
                dim3 gridtau1;
                gridtau1.x = SDIV(parameterObject->GetNumE(), blocktau1.x);
                gridtau1.y = SDIV(nPaths, 1);
                gridtau1.z = 1;

                dim3 blocktau2(128,1,1);
                dim3 gridtau2;
                gridtau2.x = SDIV(parameterObject->GetNumE(), blocktau2.x);
                gridtau2.y = SDIV(nPaths, 1);
                gridtau2.z = 1;

                callUpdateTauArraysKernelPart1(activePathsgpu, nPaths, gpuphysics, gridtau1, blocktau1, updateTauarraysStream);
                callUpdateTauArraysKernelPart2(activePathsgpu, nPaths, gpuphysics, gridtau2, blocktau2, updateTauarraysStream);
            }


            if(parameterObject->getFlags().useGlashowResonance && parameterObject->getNeutrinoType() != nusquids::NeutrinoType::neutrino){

                dim3 blockGlashow(64,1,1);
                dim3 gridGlashow;
                gridGlashow.x = SDIV(parameterObject->GetNumE(), blockGlashow.x);
                gridGlashow.y = nPaths;
                gridGlashow.z = 1;

                callUpdateGRArraysKernel(activePathsgpu, nPaths, gpuphysics, gridGlashow, blockGlashow, updateGRarraysStream);
            }


            cudaStreamSynchronize(updateNCarraysStream); CUERR;
            cudaStreamSynchronize(updateTauarraysStream); CUERR;
            cudaStreamSynchronize(updateGRarraysStream); CUERR;
        }

        template<class Physics>
        void addToPrederive(Physics* gpuphysics){
            cudaSetDevice(deviceId); CUERR;

            dim3 block(64,1,1);
            dim3 grid;
            grid.x = SDIV(parameterObject->GetNumE(), block.x);
            grid.y = nPaths;
            grid.z = 1;

            callAddToPrederiveKernel(timesgpu, activePathsgpu, nPaths, gpuphysics, grid, block, updateTimeStream);

            cudaStreamSynchronize(updateTimeStream); CUERR;
        }

        /*
            Derivation function for Version 1.
            Derivation function for Version 2 is member of the physics objects and thus not defined in Propagator.
        */
        template<class Physics>
        void derive(Physics* gpuphysics){

            dim3 blockDerive(128, 1, 1);
            dim3 gridDerive(SDIV(parameterObject->GetNumE(), blockDerive.x), SDIV(nPaths, 1), 1);

            prederive(gpuphysics);

            if(parameterObject->Get_CanUseInteractions()){

                callDeriveKernel(
                        activePathsgpu,
                        nPaths,
                        gpuphysics, gridDerive, blockDerive, hiStream);
            }else{
                callDeriveOscKernel(
                        activePathsgpu,
                        nPaths,
                        gpuphysics, gridDerive, blockDerive, hiStream);
            }

            callEndDeriveKernel(
                    activePathsgpu,
                    nPaths,
                    gpuphysics, gridDerive, blockDerive, hiStream);

            cudaStreamSynchronize(hiStream); CUERR;
        }

        /*
            Data initialization functions
        */
        template<class Physics>
        void initializeInteractions(Physics* cpuphysics){

            if(parameterObject->Get_CanUseInteractions() && !interactionsAreInitialized){

                cudaSetDevice(deviceId); CUERR;

                if(isInit_intstruct){
                    destroy_InteractionStructureGpu(intstructgpu);
                }

                intstructgpu = make_InteractionStructureGpu(deviceId, *intstructcpu);

                for(int i = 0; i < n_cosines; i++){
                    cpuphysics[i].set_intstruct(intstructgpu);
                }

                isInit_intstruct = true;
                interactionsAreInitialized = true;
            }
        }

        void initB0Proj(){
            /*
                Get Projectors from parameter object and copy them to gpu
            */
            cudaSetDevice(deviceId); CUERR;

            parameterObject->initializeProjectors();

            auto tmp = make_unique_pinned<double>(NFLVS*NFLVS*NFLVS);
            for(int flv = 0; flv < NFLVS; flv++){
                auto proj = parameterObject->GetMassProj(flv);
                for(int j = 0; j < NFLVS * NFLVS; j++){
                    tmp.get()[flv + j * b0offset] = proj[j];
                }
            }
            cudaStream_t mystream = (cudaStream_t) 0;
            cudaMemcpyAsync(B0projgpu.get(), tmp.get(), sizeof(double) * NFLVS*NFLVS*NFLVS, H2D, mystream); CUERR;
            cudaStreamSynchronize(mystream); CUERR;
        }

        void initB1Proj(){
            /*
                Get Projectors from parameter object and copy them to gpu
            */
            cudaSetDevice(deviceId); CUERR;

            parameterObject->initializeProjectors();

            auto tmp = make_unique_pinned<double>(parameterObject->GetNumRho()*NFLVS*NFLVS*NFLVS);
            for(int rho = 0; rho < parameterObject->GetNumRho(); rho++){
                for(int flv = 0; flv < NFLVS; flv++){
                    auto proj = parameterObject->GetFlavorProj(flv, rho);
                    for(int j = 0; j < NFLVS * NFLVS; j++){
                        tmp.get()[rho * NFLVS + flv + j * b1offset] = proj[j];
                    }
                }
            }
            cudaStream_t mystream = (cudaStream_t) 0;
            cudaMemcpyAsync(B1projgpu.get(), tmp.get(), sizeof(double) * parameterObject->GetNumRho()*NFLVS*NFLVS*NFLVS, H2D, mystream); CUERR;
            cudaStreamSynchronize(mystream); CUERR;
        }

        void initDM2(){
            /*
                Get DM2 from parameter object and copy to gpu
            */
            cudaSetDevice(deviceId); CUERR;

            parameterObject->initializeProjectors();

            auto dm2vec = parameterObject->getDM2();
            auto tmp = make_unique_pinned<double>(NFLVS*NFLVS);
            for(int j = 0; j < NFLVS * NFLVS; j++){
                tmp.get()[j] = dm2vec[j];
            }
            cudaStream_t mystream = (cudaStream_t) 0;
            cudaMemcpyAsync(DM2datagpu.get(), tmp.get(), sizeof(double) * NFLVS*NFLVS, H2D, mystream); CUERR;
            cudaStreamSynchronize(mystream); CUERR;
        }

        template<class Physics>
        void initH0array(Physics* cpuphysics, Physics* gpuphysics){

            cudaSetDevice(deviceId); CUERR;

            initDM2();

            if(!parameterObject->getFlags().useCoherentRhoTerms){
                cudaMemset2DAsync(H0_arraygpu.get(), h0pitch, 0, parameterObject->GetNumE(), NFLVS * NFLVS, hiStream); CUERR;
            }else{
                dim3 block(128, 1, 1);
                dim3 grid(SDIV(parameterObject->GetNumE(), block.x), 1, 1);

                cudaMemcpyAsync(gpuphysics, cpuphysics, sizeof(Physics) * 1, H2D, hiStream); CUERR;
                callInitH0arrayKernel(H0_arraygpu.get(), h0pitch, h0offset, gpuphysics, grid, block, hiStream);
            }

            cudaStreamSynchronize(hiStream);
        }


        /*
            Functions to get expectation values
        */
        template<class Physics>
        double EvalFlavorAtNode(Physics* cpuphysics,
                                Physics* gpuphysics,
                                int flavor,
                                int index_cosine,
                                int index_rho,
                                int index_energy){
            //If results are not present on the host, calculate results and transfer to host.
            if(previousEvalType != detail::EvalType::NodeFlavor){

                cudaSetDevice(deviceId); CUERR;

                std::vector<size_t> activepathshost(max_n_cosines);
                auto activepathsdevice = make_unique_dev<size_t>(deviceId, max_n_cosines); CUERR;

                auto fluxes = make_unique_dev<double>(deviceId, max_n_cosines * parameterObject->GetNumE() * parameterObject->GetNumRho() * NFLVS); CUERR;

                const unsigned int iters = SDIV(n_cosines, max_n_cosines);

                for(unsigned int currentCosineBatch = 0; currentCosineBatch < iters; currentCosineBatch++){
                    const unsigned int batchsize = currentCosineBatch < iters - 1 ? max_n_cosines : n_cosines - currentCosineBatch * max_n_cosines;

                    // copy solved states from cpu to gpu

                    cudaMemcpy2DAsync (statesgpu.get(),
                        statespitch,
                        finalStatesHost.get() + currentCosineBatch * max_n_cosines * parameterObject->GetNumRho() * parameterObject->GetNumE() * NFLVS * NFLVS,
                        parameterObject->GetNumE() * sizeof(double),
                        parameterObject->GetNumE() * sizeof(double),
                        batchsize * parameterObject->GetNumRho() * NFLVS * NFLVS,
                        H2D,
                        calculateFluxStream); CUERR;

                    // copy Physics objects to gpu
                    cudaMemcpyAsync(gpuphysics, cpuphysics + currentCosineBatch * max_n_cosines, sizeof(Physics) * batchsize, H2D, calculateFluxStream); CUERR;

                    // make list of paths to calculate
                    for(size_t i = 0; i < batchsize; i++) activepathshost[i] = i;

                    cudaMemcpyAsync(activepathsdevice.get(),
                            activepathshost.data(),
                            sizeof(size_t) * batchsize,
                            H2D, calculateFluxStream); CUERR;

                    // calculate the fluxes
                    dim3 block(64,1,1);
                    dim3 grid;
                    grid.x = SDIV(parameterObject->GetNumE(), block.x);
                    grid.y = SDIV(batchsize, 1);
                    grid.z = 1;

                    callEvalFlavorsAtNodesKernel(activepathsdevice.get(), batchsize,
                            fluxes.get(), parameterObject->GetNumRho(), parameterObject->GetNumE(), NFLVS,
                            gpuphysics,
                            grid, block, calculateFluxStream);

                    //copy fluxes from gpu to cpu
                    cudaMemcpyAsync(results.data() + currentCosineBatch * max_n_cosines * parameterObject->GetNumRho() * parameterObject->GetNumE() * NFLVS,
                                    fluxes.get(),
                                    sizeof(double) * batchsize * parameterObject->GetNumRho() * NFLVS * parameterObject->GetNumE(),
                                    D2H,
                                    calculateFluxStream); CUERR;

                    cudaStreamSynchronize(calculateFluxStream); CUERR;
                }
                previousEvalType = detail::EvalType::NodeFlavor;
            }

            const size_t resultIndex = size_t(index_cosine) * size_t(parameterObject->GetNumRho()) * size_t(NFLVS) * size_t(parameterObject->GetNumE())
                                        + size_t(index_rho) * size_t(NFLVS) * size_t(parameterObject->GetNumE())
                                        + size_t(flavor) * size_t(parameterObject->GetNumE())
                                        + size_t(index_energy);

            const double result = results[resultIndex];

            return result;
        }


        template<class Physics>
        double EvalMassAtNode(Physics* cpuphysics,
                                Physics* gpuphysics,
                                int flavor,
                                int index_cosine,
                                int index_rho,
                                int index_energy){
            //If results are not present on the host, calculate results and transfer to host.
            if(previousEvalType != detail::EvalType::NodeMass){

                cudaSetDevice(deviceId); CUERR;

                std::vector<size_t> activepathshost(max_n_cosines);
                auto activepathsdevice = make_unique_dev<size_t>(deviceId, max_n_cosines); CUERR;

                auto fluxes = make_unique_dev<double>(deviceId, max_n_cosines * parameterObject->GetNumE() * parameterObject->GetNumRho() * NFLVS); CUERR;

                const unsigned int iters = SDIV(n_cosines, max_n_cosines);

                for(unsigned int currentCosineBatch = 0; currentCosineBatch < iters; currentCosineBatch++){
                    const unsigned int batchsize = currentCosineBatch < iters - 1 ? max_n_cosines : n_cosines - currentCosineBatch * max_n_cosines;

                    // copy solved states from cpu to gpu

                    cudaMemcpy2DAsync (statesgpu.get(),
                        statespitch,
                        finalStatesHost.get() + currentCosineBatch * max_n_cosines * parameterObject->GetNumRho() * parameterObject->GetNumE() * NFLVS * NFLVS,
                        parameterObject->GetNumE() * sizeof(double),
                        parameterObject->GetNumE() * sizeof(double),
                        batchsize * parameterObject->GetNumRho() * NFLVS * NFLVS,
                        H2D,
                        calculateFluxStream); CUERR;

                    // copy Physics objects to gpu
                    cudaMemcpyAsync(gpuphysics, cpuphysics + currentCosineBatch * max_n_cosines, sizeof(Physics) * batchsize, H2D, calculateFluxStream); CUERR;

                    // make list of paths to calculate
                    for(size_t i = 0; i < batchsize; i++) activepathshost[i] = i;

                    cudaMemcpyAsync(activepathsdevice.get(),
                            activepathshost.data(),
                            sizeof(size_t) * batchsize,
                            H2D, calculateFluxStream); CUERR;

                    // calculate the fluxes
                    dim3 block(64,1,1);
                    dim3 grid;
                    grid.x = SDIV(parameterObject->GetNumE(), block.x);
                    grid.y = SDIV(batchsize, 1);
                    grid.z = 1;

                    callEvalMassesAtNodesKernel(activepathsdevice.get(), batchsize,
                            fluxes.get(), parameterObject->GetNumRho(), parameterObject->GetNumE(), NFLVS,
                            gpuphysics,
                            grid, block, calculateFluxStream);

                    //copy fluxes from gpu to cpu
                    cudaMemcpyAsync(results.data() + currentCosineBatch * max_n_cosines * parameterObject->GetNumRho() * parameterObject->GetNumE() * NFLVS,
                                    fluxes.get(),
                                    sizeof(double) * batchsize * parameterObject->GetNumRho() * NFLVS * parameterObject->GetNumE(),
                                    D2H,
                                    calculateFluxStream); CUERR;

                    cudaStreamSynchronize(calculateFluxStream); CUERR;
                }
                previousEvalType = detail::EvalType::NodeMass;
            }

			const size_t resultIndex = size_t(index_cosine) * size_t(parameterObject->GetNumRho()) * size_t(NFLVS) * size_t(parameterObject->GetNumE())
							+ size_t(index_rho) * size_t(NFLVS) * size_t(parameterObject->GetNumE())
							+ size_t(flavor) * size_t(parameterObject->GetNumE())
							+ size_t(index_energy);

            const double result = results[resultIndex];

            return result;
        }

        /*
            Functions to notify about updated parameter object
        */
        template<class Physics>
        void simulationFlagsChanged(Physics* physics){
            //if interactions were never enabled before, but are enabled now, allocate arrays
            if(!interactionArraysAreAllocated && parameterObject->Get_CanUseInteractions()){
                intstates = make_InteractionStateBufferGpu(deviceId, parameterObject->GetNumRho(), NFLVS, parameterObject->GetNumE(), max_n_cosines);
                gr_arraygpu = make_unique_dev<double>(deviceId, max_n_cosines * parameterObject->GetNumE());
                nc_arraygpu = make_unique_dev<double>(deviceId, max_n_cosines * parameterObject->GetNumRho() * NFLVS * parameterObject->GetNumE());
                tau_hadlep_arraygpu = make_unique_dev<double>(deviceId, max_n_cosines * parameterObject->GetNumRho() * parameterObject->GetNumE());
                tau_lep_arraygpu = make_unique_dev<double>(deviceId, max_n_cosines * parameterObject->GetNumRho() * parameterObject->GetNumE());
                tau_decay_fluxesgpu = make_unique_dev<double>(deviceId, max_n_cosines * parameterObject->GetNumE());
                tau_bar_decay_fluxesgpu = make_unique_dev<double>(deviceId, max_n_cosines * parameterObject->GetNumE());

                for(int i = 0; i < n_cosines; i++){
                    int indexInBatch = i % max_n_cosines;
                    physics[i].set_nc_array(nc_arraygpu.get() + indexInBatch * parameterObject->GetNumRho() * NFLVS * parameterObject->GetNumE());
                    physics[i].set_tau_decay_fluxes(tau_decay_fluxesgpu.get() + indexInBatch * parameterObject->GetNumE());
                    physics[i].set_tau_bar_decay_fluxes(tau_bar_decay_fluxesgpu.get() + indexInBatch * parameterObject->GetNumE());
                    physics[i].set_tau_hadlep_array(tau_hadlep_arraygpu.get() + indexInBatch * parameterObject->GetNumRho() * parameterObject->GetNumE());
                    physics[i].set_tau_lep_array(tau_lep_arraygpu.get() + indexInBatch * parameterObject->GetNumRho() * parameterObject->GetNumE());
                    physics[i].set_gr_array(gr_arraygpu.get() + indexInBatch * parameterObject->GetNumE());
                    physics[i].set_intstate(intstates[indexInBatch]);
                }

                interactionArraysAreAllocated = true;
            }
        }

        template<class Physics>
        void additionalDataChanged(Physics* physics){
            additionalDatagpuvec.clear();
            additionalDataPointersGpu.reset();

            const auto& addDataCPU = parameterObject->Get_AdditionalDataCpu();
            if(addDataCPU.size() != 0){

                auto addDataPointersCpu = make_unique_pinned<void*>(addDataCPU.size());
                additionalDataPointersGpu = make_unique_dev<void*>(deviceId, addDataCPU.size());

                for(size_t i = 0; i < addDataCPU.size(); i++){
                    additionalDatagpuvec.push_back(std::move(make_unique_dev<char>(deviceId, addDataCPU[i].size)));
                    addDataPointersCpu.get()[i] = (void*)additionalDatagpuvec[i].get();

                    if(addDataCPU[i].data.size() > 0){
                        if(addDataCPU[i].data.size() != addDataCPU[i].size)
                            throw std::runtime_error("Propagator: Corrupted additional data.");

                        cudaMemcpyAsync(additionalDatagpuvec[i].get(), addDataCPU[i].data.data(), addDataCPU[i].size, H2D, hiStream); CUERR;
                    }
                }

                cudaMemcpyAsync(additionalDataPointersGpu.get(), addDataPointersCpu.get(), sizeof(void*) * addDataCPU.size(), H2D, hiStream); CUERR;
                cudaStreamSynchronize(hiStream);

                for(int i = 0; i < n_cosines; i++){
                        physics[i].set_additionalData(additionalDataPointersGpu.get());
                }
            }
        }

        template<class Physics>
        void mixingParametersChanged(Physics* cpuphysics, Physics* gpuphysics){
			initB0Proj();
            initB1Proj();
            initH0array(cpuphysics, gpuphysics);
		}

        /*
            Getter and setter.
        */

        void setInitialFlux(const std::vector<double>& initialFlux_){
            if(size_t(n_cosines)  * size_t(parameterObject->GetNumE()) * size_t(parameterObject->GetNumRho()) * size_t(NFLVS) != initialFlux_.size())
                throw std::runtime_error("Propagator::setInitialFlux: Propagator was not created for this number of states");

            initialFlux = initialFlux_;
        }

        ode::RKstats getRKstats(int index_cosine) const{
            return rkstatsHost.get()[index_cosine];
        }

        void setInteractionStructure(std::shared_ptr<InteractionStructure> intstruct){
            intstructcpu = intstruct;
        }
    };


    template<int NFLVS_, class body_t, class Op_t = PhysicsOps>
    class Propagator{
		friend class CudaNusquids<NFLVS_, body_t, Op_t>;
        template<int, class, class> friend class Propagator;

        static constexpr int NFLVS = NFLVS_;
        using Body = body_t;
        using Ops = Op_t;

        using Track = typename Body::Track;
        using Physics_t = Physics<NFLVS, Body, Ops>;
        using Impl_t = PropagatorImpl;

        static_assert(NFLVS == 3 || NFLVS == 4, "Propagator: NFLVS must be 3 or 4.");

        std::unique_ptr<Impl_t> impl;

//body and tracks
        Body bodygpu;
        unique_pinned_ptr<Track> tracks; // host array of tracks

        unique_dev_ptr<Physics_t> nsqgpu; // device array of Physics_t
        unique_pinned_ptr<Physics_t> nsqcpu; // host array of Physics_t

    private:
        Propagator(const Propagator& other) = delete;
        Propagator& operator=(const Propagator& other) = delete;
    public:

        //conversion constructor.
        //convert body type of other into Body
        //other must not be accessed afterwards
        template<class Prob_t>
        Propagator(Prob_t& other) : impl(std::move(other.impl)){
            static_assert(NFLVS == other.NFLVS, "Body conversion: NFLVS does not match");

            tracks = make_unique_pinned<Track>(impl->n_cosines);

            nsqcpu = make_unique_pinned<Physics_t>(impl->max_n_cosines); // host array of device Physics pointers
            nsqgpu = make_unique_dev<Physics_t>(impl->deviceId, impl->max_n_cosines);

            impl->init_host_physics(nsqcpu.get());

			mixingParametersChanged();
			simulationFlagsChanged();
        }

        Propagator(int deviceId,
                   int n_cosines,
                   int batchsizeLimit,
                   std::shared_ptr<ParameterObject>& params){

            if(!params)
                throw std::runtime_error("Propagator::Propagator: params are null");
            impl.reset(new Impl_t(deviceId, NFLVS, n_cosines, batchsizeLimit, params));

            tracks = make_unique_pinned<Track>(n_cosines);

            nsqcpu = make_unique_pinned<Physics_t>(impl->max_n_cosines); // host array of device Physics pointers
            nsqgpu = make_unique_dev<Physics_t>(deviceId, impl->max_n_cosines);

            impl->init_host_physics(nsqcpu.get());

			mixingParametersChanged();
			simulationFlagsChanged();
        }

        Propagator(Propagator&& other){
            *this = std::move(other);
        }

        Propagator& operator=(Propagator&& other){
	        impl = std::move(other.impl);
            bodygpu = std::move(other.bodygpu);
            tracks = std::move(other.tracks);
            nsqcpu = std::move(other.nsqcpu);
            nsqgpu = std::move(other.nsqgpu);

            return *this;
        }

        void evolve(){
            impl->evolve(nsqcpu.get(),
                         nsqgpu.get(),
                         tracks.get());
        }


        /*
            Functions to get expectation values
        */

        double EvalFlavorAtNode(int flavor, int index_cosine, int index_rho, int index_energy){
            return impl->EvalFlavorAtNode(nsqcpu.get(),
                                            nsqgpu.get(),
                                            flavor,
                                            index_cosine,
                                            index_rho,
                                            index_energy);
        }

        double EvalMassAtNode(int flavor, int index_cosine, int index_rho, int index_energy){
            return impl->EvalMassAtNode(nsqcpu.get(),
                                            nsqgpu.get(),
                                            flavor,
                                            index_cosine,
                                            index_rho,
                                            index_energy);
        }

        /*
            Functions to notify about updated parameter object
        */

		void mixingParametersChanged(){
            impl->mixingParametersChanged(nsqcpu.get(), nsqgpu.get());
		}

		void simulationFlagsChanged(){
            impl->simulationFlagsChanged(nsqcpu.get());
		}

        void additionalDataChanged(){
            impl->additionalDataChanged(nsqcpu.get());
        }

        /*
            Getter and setter.
        */

        void setInitialFlux(const std::vector<double>& initialFlux_){
            impl->setInitialFlux(initialFlux_);
        }

        void setBody(const body_t& body_){
            bodygpu = body_;

            for(int i = 0; i < impl->n_cosines; i++){
                nsqcpu.get()[i].set_body(bodygpu);
            }
        }

        void setTracks(const std::vector<Track>& tracks_){
            if(size_t(impl->n_cosines) != tracks_.size()){
                throw std::runtime_error("setTracks error, must provide one track per cosine bin.");
            }

			std::copy(tracks_.begin(), tracks_.end(), tracks.get());
        }

        ode::RKstats getRKstats(int index_cosine) const{
            return impl->getRKstats(index_cosine);
        }

        void setInteractionStructure(std::shared_ptr<InteractionStructure> intstruct){
            impl->setInteractionStructure(intstruct);
        }

    };


} //namespace end

#endif
