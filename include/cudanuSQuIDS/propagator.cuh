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
	
	template<unsigned int NFLVS, class body_t, unsigned int BATCH_SIZE_LIMIT, class Ops>
	class CudaNusquids;


    template<unsigned int NFLVS, class body_t, unsigned int BATCH_SIZE_LIMIT = 400, class Ops = PhysicsOps<NFLVS, body_t>>
    class Propagator{
		friend class CudaNusquids<NFLVS, body_t, BATCH_SIZE_LIMIT, Ops>;

        using MyPhysics = Physics<NFLVS, body_t, Ops>;

        static_assert(BATCH_SIZE_LIMIT > 0, "BATCH_SIZE_LIMIT must be greater than zero");
        static_assert(NFLVS == 3 || NFLVS == 4, "Propagator: NFLVS must be 3 or 4.");

        static constexpr unsigned int susize = NFLVS * NFLVS;

        detail::EvalType previousEvalType = detail::EvalType::None;


		std::shared_ptr<ParameterObject> parameterObject;

    // Data

    //input data
		std::vector<double> cosineList;
        std::vector<double> initialFlux;

    //final states

        unique_pinned_ptr<double> finalStatesHost;

        std::vector<double> results;

        unique_dev_ptr<double> energyListgpu;
        unique_dev_ptr<double> cosineListgpu;
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

//body and tracks
        body_t bodygpu;
        unique_pinned_ptr<typename body_t::Track> tracks; // host array of device track pointers

// memory pitch in bytes for energy dimension
        size_t currentFluxesPitch;
        size_t h0pitch;
        size_t evolB1pitch;
        size_t statespitch;
// distance between two consecutive elements of the same density matrix
        size_t statesOffset;
        size_t evolB1offset;
        size_t h0offset;
        size_t b0offset;
        size_t b1offset;


        int deviceId;

        // ODE solver statistics
        unique_pinned_ptr<ode::RKstats> rkstatsHost;
        unique_dev_ptr<ode::RKstats> rkstatsgpu;

    // Problem dimensions
        size_t n_cosines = 0;
    // Maximum number of paths with parameterObject->GetNumE() energy nodes that can be calculated in parallel (hardware limit, e.g. available memory).
    //Larger problems will be batched
        size_t max_n_cosines = 0;

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

        static constexpr size_t nCopyStreams = 2;

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

        unique_dev_ptr<MyPhysics> nsqgpu; // device array of MyPhysics
        unique_pinned_ptr<MyPhysics> nsqcpu; // host array of MyPhysics

    private:
        Propagator(const Propagator& other) = delete;
        Propagator& operator=(const Propagator& other) = delete;
    public:

    Propagator(int deviceId_, size_t n_cosines_, std::shared_ptr<ParameterObject>& params)
            : deviceId(deviceId_), n_cosines(n_cosines_), parameterObject(params){

	    if(!params) throw std::runtime_error("Propagator::Propagator: params are null");

            int nGpus;
            cudaGetDeviceCount(&nGpus); CUERR;

            if(deviceId < 0 || deviceId >= nGpus)
                throw std::runtime_error("Propagator : invalid device id");

            cudaSetDevice(deviceId); CUERR;

            max_n_cosines = std::min(size_t(BATCH_SIZE_LIMIT), n_cosines);

            results.resize(NFLVS * n_cosines * parameterObject->GetNumRho() * parameterObject->GetNumE());
            cosineList.resize(n_cosines);
            initialFlux.resize(n_cosines  * parameterObject->GetNumE() * parameterObject->GetNumRho() * NFLVS);

            finalStatesHost = make_unique_pinned<double>(n_cosines * parameterObject->GetNumRho() * parameterObject->GetNumE() * susize);

            energyListgpu = make_unique_dev<double>(deviceId, parameterObject->GetNumE());
            cosineListgpu = make_unique_dev<double>(deviceId, max_n_cosines);
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


            delEgpu = make_unique_dev<double>(deviceId, parameterObject->GetNumE() - 1);
            tracks = make_unique_pinned<typename body_t::Track>(n_cosines);

            if(!interactionArraysAreAllocated && parameterObject->GetUseInteractions()){
                intstates = make_InteractionStateBufferGpu(deviceId, parameterObject->GetNumRho(), NFLVS, parameterObject->GetNumE(), max_n_cosines);
                gr_arraygpu = make_unique_dev<double>(deviceId, max_n_cosines * parameterObject->GetNumE());
                nc_arraygpu = make_unique_dev<double>(deviceId, max_n_cosines * parameterObject->GetNumRho() * NFLVS * parameterObject->GetNumE());
                tau_hadlep_arraygpu = make_unique_dev<double>(deviceId, max_n_cosines * parameterObject->GetNumRho() * parameterObject->GetNumE());
                tau_lep_arraygpu = make_unique_dev<double>(deviceId, max_n_cosines * parameterObject->GetNumRho() * parameterObject->GetNumE());
                tau_decay_fluxesgpu = make_unique_dev<double>(deviceId, max_n_cosines * parameterObject->GetNumE());
                tau_bar_decay_fluxesgpu = make_unique_dev<double>(deviceId, max_n_cosines * parameterObject->GetNumE());

                interactionArraysAreAllocated = true;
            }

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

            for(size_t i = 0; i < nCopyStreams; i++)
                cudaStreamCreate(&copyStreams[i]); CUERR;

            rkstatsHost = make_unique_pinned<ode::RKstats>(n_cosines); CUERR;
            rkstatsgpu = make_unique_dev<ode::RKstats>(deviceId, max_n_cosines); CUERR;

            nsqcpu = make_unique_pinned<MyPhysics>(max_n_cosines); // host array of device Physics pointers
            nsqgpu = make_unique_dev<MyPhysics>(deviceId, max_n_cosines);

            // let MyPhysics objects point to correct device memory
            for(size_t i = 0; i < n_cosines; i++){

                size_t indexInBatch = i % max_n_cosines;

                // problem dimensions
                nsqcpu.get()[i].set_max_n_cosines(max_n_cosines);
                nsqcpu.get()[i].set_n_rhos(parameterObject->GetNumRho());
                nsqcpu.get()[i].set_n_energies(parameterObject->GetNumE());
                nsqcpu.get()[i].set_indexInBatch(indexInBatch);
                nsqcpu.get()[i].set_globalPathId(i);

                nsqcpu.get()[i].set_basis(parameterObject->getBasis());
                nsqcpu.get()[i].set_neutrinoType(parameterObject->getNeutrinoType());

                // variables for data access
                nsqcpu.get()[i].set_b0offset(b0offset);
                nsqcpu.get()[i].set_b1offset(b1offset);
                nsqcpu.get()[i].set_evolB1pitch(evolB1pitch);
                nsqcpu.get()[i].set_evoloffset(evolB1offset);
                nsqcpu.get()[i].set_h0pitch(h0pitch);
                nsqcpu.get()[i].set_h0offset(h0offset);
                nsqcpu.get()[i].set_statesOffset(statesOffset);
                nsqcpu.get()[i].set_statesPitch(statespitch);
                nsqcpu.get()[i].set_fluxPitch(currentFluxesPitch);

                // data arrays
                nsqcpu.get()[i].set_states(statesgpu.get() + indexInBatch * statespitch / sizeof(double) * parameterObject->GetNumRho() * susize);
                nsqcpu.get()[i].set_b0proj(B0projgpu.get());
                nsqcpu.get()[i].set_b1proj(B1projgpu.get());
                nsqcpu.get()[i].set_evolB1proj(evolB1projgpu.get() + indexInBatch * evolB1pitch / sizeof(double) * parameterObject->GetNumRho() * NFLVS * susize);
                nsqcpu.get()[i].set_H0_array(H0_arraygpu.get());
                nsqcpu.get()[i].set_dm2(DM2datagpu.get());
                nsqcpu.get()[i].set_energyList(energyListgpu.get());
                nsqcpu.get()[i].set_delE(delEgpu.get());
                nsqcpu.get()[i].set_fluxes(currentFluxes.get() + indexInBatch * currentFluxesPitch / sizeof(double) * parameterObject->GetNumRho() * NFLVS);

                if(parameterObject->GetUseInteractions()){
                    nsqcpu.get()[i].set_nc_array(nc_arraygpu.get() + indexInBatch * parameterObject->GetNumRho() * NFLVS * parameterObject->GetNumE());
                    nsqcpu.get()[i].set_tau_decay_fluxes(tau_decay_fluxesgpu.get() + indexInBatch * parameterObject->GetNumE());
                    nsqcpu.get()[i].set_tau_bar_decay_fluxes(tau_bar_decay_fluxesgpu.get() + indexInBatch * parameterObject->GetNumE());
                    nsqcpu.get()[i].set_tau_hadlep_array(tau_hadlep_arraygpu.get() + indexInBatch * parameterObject->GetNumRho() * parameterObject->GetNumE());
                    nsqcpu.get()[i].set_tau_lep_array(tau_lep_arraygpu.get() + indexInBatch * parameterObject->GetNumRho() * parameterObject->GetNumE());
                    nsqcpu.get()[i].set_gr_array(gr_arraygpu.get() + indexInBatch * parameterObject->GetNumE());
                    nsqcpu.get()[i].set_intstate(intstates[indexInBatch]);
                }
            }

            //copy energy list to gpu
            auto energyList = parameterObject->GetERange();
            cudaMemcpyAsync(energyListgpu.get(), &energyList[0], sizeof(double) * parameterObject->GetNumE(), H2D, copyStreams[0]); CUERR;
            std::vector<double> delE(parameterObject->GetNumE() - 1);

            for(size_t i = 0; i < parameterObject->GetNumE()-1; i++)
                delE[i] = energyList[i+1] - energyList[i];

            cudaMemcpyAsync(delEgpu.get(), delE.data(), sizeof(double) * (parameterObject->GetNumE() - 1), H2D, copyStreams[0]); CUERR;

            cudaStreamSynchronize(copyStreams[0]);


			mixingParametersChanged();
			simulationFlagsChanged();
        }

        Propagator(Propagator&& other){
            *this = std::move(other);

            cudaSetDevice(deviceId);

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

            for(size_t i = 0; i < nCopyStreams; i++)
                cudaStreamCreate(&copyStreams[i]); CUERR;
        }

        Propagator& operator=(Propagator&& other){

	    parameterObject = std::move(other.parameterObject);
            cosineList = std::move(other.cosineList);
            finalStatesHost = std::move(other.finalStatesHost);
            results = std::move(other.results);
	    initialFlux = std::move(other.initialFlux);
            energyListgpu = std::move(other.energyListgpu);
            cosineListgpu = std::move(other.cosineListgpu);
            initialFluxgpu = std::move(other.initialFluxgpu);
            DM2datagpu = std::move(other.DM2datagpu);
            H0_arraygpu = std::move(other.H0_arraygpu);
            delEgpu = std::move(other.delEgpu);
            B0projgpu = std::move(other.B0projgpu);
            B1projgpu = std::move(other.B1projgpu);
            evolB1projgpu = std::move(other.evolB1projgpu);
            statesgpu = std::move(other.statesgpu);
            intstructcpu = std::move(other.intstructcpu);
            intstructgpu = other.intstructgpu;
            bodygpu = std::move(other.bodygpu);

            tracks = std::move(other.tracks);

            gr_arraygpu = std::move(other.gr_arraygpu);
            nc_arraygpu = std::move(other.nc_arraygpu);
            tau_hadlep_arraygpu = std::move(other.tau_hadlep_arraygpu);
            tau_lep_arraygpu = std::move(other.tau_lep_arraygpu);
            tau_decay_fluxesgpu = std::move(other.tau_decay_fluxesgpu);
            tau_bar_decay_fluxesgpu = std::move(other.tau_bar_decay_fluxesgpu);
            currentFluxes = std::move(other.currentFluxes);

            rkstatsHost = std::move(other.rkstatsHost);
            rkstatsgpu = std::move(other.rkstatsgpu);

            nsqcpu = std::move(other.nsqcpu);
            nsqgpu = std::move(other.nsqgpu);
			
	    additionalDataPointersGpu = std::move(other.additionalDataPointersGpu);
	    additionalDatagpuvec = std::move(other.additionalDatagpuvec);

            intstates = std::move(other.intstates);

            currentFluxesPitch = other.currentFluxesPitch;
            h0pitch = other.h0pitch;
            evolB1pitch = other.evolB1pitch;
            statespitch = other.statespitch;
            statesOffset = other.statesOffset;
            evolB1offset = other.evolB1offset;
            h0offset = other.h0offset;
            b0offset = other.b0offset;
            b1offset = other.b1offset;
            deviceId = other.deviceId;

            n_cosines = other.n_cosines;
            max_n_cosines = other.max_n_cosines;

	    timesgpu = other.timesgpu;
	    derivationInputgpu = other.derivationInputgpu;
	    derivationOutputgpu = other.derivationOutputgpu;
	    activePathsgpu = other.activePathsgpu;
	    nPaths = other.nPaths;

	    previousEvalType = other.previousEvalType;

            isInit_intstruct = other.isInit_intstruct;
            interactionsAreInitialized = other.interactionsAreInitialized;
            interactionArraysAreAllocated = other.interactionArraysAreAllocated;

            other.isInit_intstruct = false;
            other.interactionsAreInitialized = false;
            other.interactionArraysAreAllocated = false;

            return *this;
        }



        ~Propagator(){
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

        /*
           Functions to evolve systems
        */

        void evolve(){
            previousEvalType = detail::EvalType::None; //invalidate results of previous simulation

            switch(parameterObject->Get_SolverType()){
                case SolverType::Version1:
				            evolveVersion1();
                            break;
                case SolverType::Version2:
                            evolveVersion2();
                            break;
                default: throw std::runtime_error("Propagator::evolve: invalid solverType.");
            }
        }

        void evolveVersion1(){
            switch(parameterObject->Get_StepperType()){
            case ode::StepperType::RK4:
                evolveVersion1_impl<ode::stepper::RK42D>();
                break;
            default:
                throw std::runtime_error("Unknown stepper type");
            };
        }

        void evolveVersion2(){
            switch(parameterObject->Get_StepperType()){
            case ode::StepperType::RK4:
                evolveVersion2_impl<ode::stepper::RK4>();
                break;
            default:
                throw std::runtime_error("Unknown stepper type");
            };

        }

        template<class Stepper>
        void evolveVersion1_impl(){
            cudaSetDevice(deviceId); CUERR;

            // interface to ODE solver
            auto RHS = [](const size_t* const activeIndices, size_t nIndices, const double* const t,
                                double* const y, double* const y_derived, void* userdata){
                Propagator* nsq = static_cast<Propagator*>(userdata);
                nsq->timesgpu = t;
                nsq->derivationInputgpu = y;
                nsq->derivationOutputgpu = y_derived;
                nsq->activePathsgpu = activeIndices;
                nsq->nPaths = nIndices;
                nsq->derive();
            };

            // progress function
            auto progressFunc = [](double p, double mp){
                        double frac = 100.0 * p / mp;
                        printf("\r%3.6f %%", frac);
                        if(frac >= 100.0)
                            printf("\n");
            };

            beforeEvolution(); CUERR;

            std::vector<double> t_begin(max_n_cosines);
            std::vector<double> t_end(max_n_cosines);

            const unsigned int iters = SDIV(n_cosines, max_n_cosines);

            for(unsigned int currentCosineBatch = 0; currentCosineBatch < iters; currentCosineBatch++){
                const unsigned int batchsize = currentCosineBatch < iters - 1 ? max_n_cosines : n_cosines - currentCosineBatch * max_n_cosines;

                beforeBatchEvolution(currentCosineBatch);

                // get begin and end of current paths
                for(size_t i = 0; i < batchsize; i++){
                    t_begin[i] = tracks.get()[currentCosineBatch * max_n_cosines + i].getXBegin();
                    t_end[i] = tracks.get()[currentCosineBatch * max_n_cosines + i].getXEnd();
                }

                //set up ODE solver
                ode::Solver2D<Stepper> solver(deviceId, statesgpu.get() , statespitch,
                                                parameterObject->GetNumE(), NFLVS * NFLVS * parameterObject->GetNumRho(), batchsize,
                                                parameterObject->Get_NumSteps(),
                                                parameterObject->Get_h(), parameterObject->Get_h_min(), parameterObject->Get_h_max(),
                                                t_begin, t_end, (void*)this,
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

                afterBatchEvolution(currentCosineBatch);
            }
        }

        template<class Stepper>
        void evolveVersion2_impl(){
            cudaSetDevice(deviceId); CUERR;

            size_t workspacePerSolver = ode::SolverGPU<Stepper>::getMinimumMemorySize(statespitch, NFLVS * NFLVS * parameterObject->GetNumRho());
            auto solverworkspace = make_unique_dev<double>(deviceId, workspacePerSolver * max_n_cosines); CUERR;

            beforeEvolution();

            size_t evolveBlockDimx = 128;
            size_t evolveGridDimx = 1;

            const unsigned int iters = SDIV(n_cosines, max_n_cosines);

            for(unsigned int currentCosineBatch = 0; currentCosineBatch < iters; currentCosineBatch++){
                const unsigned int batchsize = currentCosineBatch < iters - 1 ? max_n_cosines : n_cosines - currentCosineBatch * max_n_cosines;

                beforeBatchEvolution(currentCosineBatch);

                dim3 evolveblock(evolveBlockDimx, 1, 1);
                dim3 evolvegrid(evolveGridDimx, batchsize, 1);

                const bool anyInteractions = !(parameterObject->getFlags().useNonCoherentRhoTerms
                                                || parameterObject->getFlags().useNCInteractions
                                                || parameterObject->getFlags().useTauRegeneration
                                                || parameterObject->getFlags().useGlashowResonance);

                callEvolveKernel(nsqgpu.get() , batchsize, anyInteractions,
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

                afterBatchEvolution(currentCosineBatch);
            }
        }

        /*
            Evolution helper functions
        */

        void beforeEvolution(){
            if(parameterObject->GetUseInteractions()){
                initializeInteractions(); CUERR;
            }

            //set track position to start
            for(unsigned int i = 0; i < n_cosines; i++){
                const double begin = tracks.get()[i].getXBegin();
                tracks.get()[i].setCurrentX(begin);
            }

            updatePhysicsObject(); CUERR;
        }

        void updatePhysicsObject(){
            for(size_t i = 0; i < n_cosines; i++){
                nsqcpu.get()[i].set_flags(parameterObject->getFlags());
                nsqcpu.get()[i].set_intstruct(intstructgpu);
            }
        }

        void beforeBatchEvolution(unsigned int currentCosineBatch){
            const unsigned int iters = SDIV(n_cosines, max_n_cosines);
            const unsigned int batchsize = currentCosineBatch < iters - 1 ? max_n_cosines : n_cosines - currentCosineBatch * max_n_cosines;

            // set up the current batch

            // set correct tracks in physics objects
            for(size_t i = 0; i < batchsize; i++){
                nsqcpu.get()[currentCosineBatch * max_n_cosines + i].set_track(tracks.get()[currentCosineBatch * max_n_cosines + i]);
            }

            cudaMemcpyAsync(nsqgpu.get(), nsqcpu.get() + currentCosineBatch * max_n_cosines, sizeof(MyPhysics) * batchsize, H2D, hiStream); CUERR;

            // copy correct cosine values to gpu
            cudaMemcpyAsync(cosineListgpu.get(), cosineList.data() + currentCosineBatch * max_n_cosines, sizeof(double) * batchsize, H2D, hiStream); CUERR;

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

        void afterBatchEvolution(unsigned int currentCosineBatch){
            const unsigned int iters = SDIV(n_cosines, max_n_cosines);
            const unsigned int batchsize = currentCosineBatch < iters - 1 ? max_n_cosines : n_cosines - currentCosineBatch * max_n_cosines;
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
            cudaMemcpyAsync(nsqcpu.get() + currentCosineBatch * max_n_cosines, nsqgpu.get(), sizeof(MyPhysics) * batchsize, H2D, hiStream); CUERR;

            // copy solved tracks to tracks array
            for(size_t i = 0; i < batchsize; i++){
                tracks.get()[currentCosineBatch * max_n_cosines + i] = nsqcpu.get()[currentCosineBatch * max_n_cosines + i].get_track();
			}

            cudaStreamSynchronize(hiStream);
        }

        /*
            Prederive functions for Version 1.
            Prederive functions for Version 2 are members of the physics objects and thus not defined in Propagator.
        */

        void setDerivationPointers(double* y_in, double* y_out, const size_t* const activePaths, const size_t nPaths, cudaStream_t stream){
                dim3 block(1,1,1);
                dim3 grid;
                grid.x = 1;
                grid.y = SDIV(nPaths, block.y);
                grid.z = 1;

                callSetDerivationPointersKernel(activePaths, nPaths, nsqgpu.get(), y_in, y_out, grid, block, stream);
        }

        void updateTime(const double* const times, const size_t* const activePaths, const size_t nPaths, cudaStream_t stream){
                dim3 block(1,1,1);
                dim3 grid;
                grid.x = 1;
                grid.y = SDIV(nPaths, block.y);
                grid.z = 1;

                callUpdateTimeKernel(times, activePaths, nPaths, nsqgpu.get(), grid, block, stream);
        }

        void evolveProjectorsAsync(const double* const times, const size_t* const activePaths, const size_t nPaths, cudaStream_t stream){

            dim3 block(64,1,1);
            dim3 grid;
            grid.x = SDIV(parameterObject->GetNumE(), block.x);
            grid.y = SDIV(nPaths, 1);
            grid.z = 1;

            callEvolveProjectorsKernel(times, activePaths, nPaths,
                            nsqgpu.get(), grid, block, stream);
        }

        void updateInteractionStructAsync(const double* const times, const size_t* const activePaths, const size_t nPaths,
                                                            cudaStream_t stream){
            dim3 block(64,1,1);
            dim3 grid;
            grid.x = SDIV(parameterObject->GetNumE(), block.x);
            grid.y = SDIV(nPaths, 1);
            grid.z = 1;

            cudaStreamSynchronize(updateTimeStream); CUERR;

            callUpdateInteractionStructKernel(activePaths, nPaths,
                                    nsqgpu.get(), grid, block, stream);
        }

        void calculateCurrentFlavorFluxesAsync(const size_t* const d_activePaths, size_t nPaths_,
                                                    const double* const d_states, double* d_fluxes,
                                                    cudaStream_t stream){

            dim3 block(64,1,1);
            dim3 grid;
            grid.x = SDIV(parameterObject->GetNumE(), block.x);
            grid.y = SDIV(nPaths_, 1);
            grid.z = 1;

            callCalculateCurrentFlavorFluxesKernel(d_activePaths, nPaths_,
                                    nsqgpu.get(), grid, block, stream);
        }


        void updateInteractionArrays(){
            if(!(parameterObject->getFlags().useNCInteractions || parameterObject->getFlags().useTauRegeneration || parameterObject->getFlags().useGlashowResonance))
                return;

            calculateCurrentFlavorFluxesAsync(activePathsgpu, nPaths, derivationInputgpu, currentFluxes.get(), calculateFluxStream);
            cudaStreamSynchronize(calculateFluxStream); CUERR;

            if(parameterObject->getFlags().useNCInteractions){

                dim3 blockNC(64,1,1);
                dim3 gridNC;
                gridNC.x = SDIV(parameterObject->GetNumE(), blockNC.x);
                gridNC.y = SDIV(nPaths, 1);
                gridNC.z = 1;

                callUpdateNCArraysKernel(activePathsgpu, nPaths, nsqgpu.get(), gridNC, blockNC, updateNCarraysStream);
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

                callUpdateTauArraysKernelPart1(activePathsgpu, nPaths, nsqgpu.get(), gridtau1, blocktau1, updateTauarraysStream);
                callUpdateTauArraysKernelPart2(activePathsgpu, nPaths, nsqgpu.get(), gridtau2, blocktau2, updateTauarraysStream);
            }


            if(parameterObject->getFlags().useGlashowResonance && parameterObject->getNeutrinoType() != nusquids::NeutrinoType::neutrino){

                dim3 blockGlashow(64,1,1);
                dim3 gridGlashow;
                gridGlashow.x = SDIV(parameterObject->GetNumE(), blockGlashow.x);
                gridGlashow.y = nPaths;
                gridGlashow.z = 1;

                callUpdateGRArraysKernel(activePathsgpu, nPaths, nsqgpu.get(), gridGlashow, blockGlashow, updateGRarraysStream);
            }


            cudaStreamSynchronize(updateNCarraysStream); CUERR;
            cudaStreamSynchronize(updateTauarraysStream); CUERR;
            cudaStreamSynchronize(updateGRarraysStream); CUERR;
        }

        void addToPrederive(){
            cudaSetDevice(deviceId); CUERR;

            dim3 block(64,1,1);
            dim3 grid;
            grid.x = SDIV(parameterObject->GetNumE(), block.x);
            grid.y = nPaths;
            grid.z = 1;

            callAddToPrederiveKernel(timesgpu, activePathsgpu, nPaths, nsqgpu.get(), grid, block, updateTimeStream);

            cudaStreamSynchronize(updateTimeStream); CUERR;
        }

        void prederive(){

            cudaSetDevice(deviceId); CUERR;
            setDerivationPointers(derivationInputgpu, derivationOutputgpu, activePathsgpu, nPaths, updateTimeStream);
            updateTime(timesgpu, activePathsgpu, nPaths, updateTimeStream);
            cudaStreamSynchronize(updateTimeStream); CUERR; //need correct time for projector evoluation
            evolveProjectorsAsync(timesgpu, activePathsgpu, nPaths, evolveProjectorsStream);

            if((parameterObject->getFlags().useNonCoherentRhoTerms || parameterObject->getFlags().useNCInteractions || parameterObject->getFlags().useTauRegeneration || parameterObject->getFlags().useGlashowResonance)){

                cudaStreamSynchronize(updateTimeStream); CUERR; // need correct densities and electronfractions
                updateInteractionStructAsync(timesgpu, activePathsgpu, nPaths, updateIntStructStream);
                cudaStreamSynchronize(evolveProjectorsStream); CUERR;
                cudaStreamSynchronize(updateIntStructStream); CUERR;
                updateInteractionArrays();

            }else{
                cudaStreamSynchronize(updateTimeStream); CUERR;
                cudaStreamSynchronize(evolveProjectorsStream); CUERR;
            }

            addToPrederive();
        }

        /*
            Derivation function for Version 1.
            Derivation function for Version 2 is member of the physics objects and thus not defined in Propagator.
        */

        void derive(){
            cudaSetDevice(deviceId); CUERR;

            dim3 blockDerive(128, 1, 1);
            dim3 gridDerive(SDIV(parameterObject->GetNumE(), blockDerive.x), SDIV(nPaths, 1), 1);

            prederive();

            if((parameterObject->getFlags().useNonCoherentRhoTerms
                || parameterObject->getFlags().useNCInteractions
                || parameterObject->getFlags().useTauRegeneration
                || parameterObject->getFlags().useGlashowResonance)){

                callDeriveKernel(
                        activePathsgpu,
                        nPaths,
                        nsqgpu.get(), gridDerive, blockDerive, hiStream);
            }else{
                callDeriveOscKernel(
                        activePathsgpu,
                        nPaths,
                        nsqgpu.get(), gridDerive, blockDerive, hiStream);
            }

            callEndDeriveKernel(
                    activePathsgpu,
                    nPaths,
                    nsqgpu.get(), gridDerive, blockDerive, hiStream);

            cudaStreamSynchronize(hiStream); CUERR;
        }

        /*
            Data initialization functions
        */

        void initializeInteractions(){

            if(parameterObject->GetUseInteractions() && !interactionsAreInitialized){
                CUERR;

                cudaSetDevice(deviceId); CUERR;

                if(isInit_intstruct){
                    destroy_InteractionStructureGpu(intstructgpu);
                }

                intstructgpu = make_InteractionStructureGpu(deviceId, *intstructcpu);

                for(size_t i = 0; i < n_cosines; i++){
                    nsqcpu.get()[i].set_intstruct(intstructgpu);
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
            for(unsigned int flv = 0; flv < NFLVS; flv++){
                auto proj = parameterObject->GetMassProj(flv);
                for(unsigned int j = 0; j < NFLVS * NFLVS; j++){
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
            for(unsigned int rho = 0; rho < parameterObject->GetNumRho(); rho++){
                for(unsigned int flv = 0; flv < NFLVS; flv++){
                    auto proj = parameterObject->GetFlavorProj(flv, rho);
                    for(unsigned int j = 0; j < NFLVS * NFLVS; j++){
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
            for(unsigned int j = 0; j < NFLVS * NFLVS; j++){
                tmp.get()[j] = dm2vec[j];
            }
            cudaStream_t mystream = (cudaStream_t) 0;
            cudaMemcpyAsync(DM2datagpu.get(), tmp.get(), sizeof(double) * NFLVS*NFLVS, H2D, mystream); CUERR;
            cudaStreamSynchronize(mystream); CUERR;
        }

        void initH0array(){

            cudaSetDevice(deviceId); CUERR;

            initDM2();

            if(!parameterObject->getFlags().useCoherentRhoTerms){
                cudaMemset2DAsync(H0_arraygpu.get(), h0pitch, 0, parameterObject->GetNumE(), NFLVS * NFLVS, hiStream); CUERR;
            }else{
                dim3 block(128, 1, 1);
                dim3 grid(SDIV(parameterObject->GetNumE(), block.x), 1, 1);

                cudaMemcpyAsync(nsqgpu.get(), nsqcpu.get(), sizeof(MyPhysics) * 1, H2D, hiStream); CUERR;
                callInitH0arrayKernel(H0_arraygpu.get(), h0pitch, h0offset, nsqgpu.get(), grid, block, hiStream);
            }

            cudaStreamSynchronize(hiStream);
        }

        /*
            Functions to get expectation values
        */

        double EvalFlavorAtNode(size_t flavor, size_t index_cosine, size_t index_rho, size_t index_energy){
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
                    cudaMemcpyAsync(nsqgpu.get(), nsqcpu.get() + currentCosineBatch * max_n_cosines, sizeof(MyPhysics) * batchsize, H2D, calculateFluxStream); CUERR;

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
                            nsqgpu.get(),
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

            const size_t resultIndex = index_cosine * parameterObject->GetNumRho()* NFLVS * parameterObject->GetNumE()
                                        + index_rho * NFLVS * parameterObject->GetNumE()
                                        + flavor * parameterObject->GetNumE()
                                        + index_energy;

            const double result = results[resultIndex];

            return result;
        }

        double EvalMassAtNode(size_t flavor, size_t index_cosine, size_t index_rho, size_t index_energy){
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
                    cudaMemcpyAsync(nsqgpu.get(), nsqcpu.get() + currentCosineBatch * max_n_cosines, sizeof(MyPhysics) * batchsize, H2D, calculateFluxStream); CUERR;

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
                            nsqgpu.get(),
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

            const size_t resultIndex = index_cosine * parameterObject->GetNumRho()* NFLVS * parameterObject->GetNumE()
                                        + index_rho * NFLVS * parameterObject->GetNumE()
                                        + flavor * parameterObject->GetNumE()
                                        + index_energy;

            const double result = results[resultIndex];

            return result;
        }

        /*
            Functions to notify about updated parameter object
        */

		void mixingParametersChanged(){
			initB0Proj();
            initB1Proj();
            initH0array();
		}

		void simulationFlagsChanged(){
            //if interactions were never enabled before, but are enabled now, allocate arrays
            if(!interactionArraysAreAllocated && parameterObject->GetUseInteractions()){
                intstates = make_InteractionStateBufferGpu(deviceId, parameterObject->GetNumRho(), NFLVS, parameterObject->GetNumE(), max_n_cosines);
                gr_arraygpu = make_unique_dev<double>(deviceId, max_n_cosines * parameterObject->GetNumE());
                nc_arraygpu = make_unique_dev<double>(deviceId, max_n_cosines * parameterObject->GetNumRho() * NFLVS * parameterObject->GetNumE());
                tau_hadlep_arraygpu = make_unique_dev<double>(deviceId, max_n_cosines * parameterObject->GetNumRho() * parameterObject->GetNumE());
                tau_lep_arraygpu = make_unique_dev<double>(deviceId, max_n_cosines * parameterObject->GetNumRho() * parameterObject->GetNumE());
                tau_decay_fluxesgpu = make_unique_dev<double>(deviceId, max_n_cosines * parameterObject->GetNumE());
                tau_bar_decay_fluxesgpu = make_unique_dev<double>(deviceId, max_n_cosines * parameterObject->GetNumE());

				for(size_t i = 0; i < n_cosines; i++){
					size_t indexInBatch = i % max_n_cosines;
                    nsqcpu.get()[i].set_nc_array(nc_arraygpu.get() + indexInBatch * parameterObject->GetNumRho() * NFLVS * parameterObject->GetNumE());
                    nsqcpu.get()[i].set_tau_decay_fluxes(tau_decay_fluxesgpu.get() + indexInBatch * parameterObject->GetNumE());
                    nsqcpu.get()[i].set_tau_bar_decay_fluxes(tau_bar_decay_fluxesgpu.get() + indexInBatch * parameterObject->GetNumE());
                    nsqcpu.get()[i].set_tau_hadlep_array(tau_hadlep_arraygpu.get() + indexInBatch * parameterObject->GetNumRho() * parameterObject->GetNumE());
                    nsqcpu.get()[i].set_tau_lep_array(tau_lep_arraygpu.get() + indexInBatch * parameterObject->GetNumRho() * parameterObject->GetNumE());
                    nsqcpu.get()[i].set_gr_array(gr_arraygpu.get() + indexInBatch * parameterObject->GetNumE());
                    nsqcpu.get()[i].set_intstate(intstates[indexInBatch]);
				}

                interactionArraysAreAllocated = true;
            }
		}

        void additionalDataChanged(){
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

                for(size_t i = 0; i < n_cosines; i++){
                        nsqcpu.get()[i].set_additionalData(additionalDataPointersGpu.get());
                }
            }
        }

        /*
            Getter and setter.
        */

        void setInitialFlux(const std::vector<double>& initialFlux_){
            if(n_cosines  * parameterObject->GetNumE() * parameterObject->GetNumRho() * NFLVS != initialFlux_.size())
                throw std::runtime_error("Propagator::setInitialFlux: Propagator was not created for this number of states");

            initialFlux = initialFlux_;
        }

        void setBody(const body_t& body_){
            bodygpu = body_;

            for(size_t i = 0; i < n_cosines; i++){
                nsqcpu.get()[i].set_body(bodygpu);
            }
        }

        void setTracks(const std::vector<typename body_t::Track>& tracks_){
            if(n_cosines != tracks_.size()){
                throw std::runtime_error("setTracks error, must provide one track per cosine bin.");
            }

			std::copy(tracks_.begin(), tracks_.end(), tracks.get());
        }

        void setCosineList(const std::vector<double>& list){
            if(n_cosines != list.size()) throw std::runtime_error("nusquids_core was not created for this number of cosine nodes");

            cosineList = list;
        }

        ode::RKstats getRKstats(size_t index_cosine) const{
            return rkstatsHost.get()[index_cosine];
        }
    };

} //namespace end

#endif
