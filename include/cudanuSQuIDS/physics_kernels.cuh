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

#ifndef CUDANUSQUIDS_PHYSICS_KERNELS_CUH
#define CUDANUSQUIDS_PHYSICS_KERNELS_CUH

#include <cudanuSQuIDS/cudahelpers.cuh>
#include <cudanuSQuIDS/physics.cuh>
#include <cudanuSQuIDS/ode.cuh>
#include <cudanuSQuIDS/cudautils.cuh>


#include <stdexcept>
#include <cuda_profiler_api.h>

namespace cudanusquids{


/*
 *
 * Expectation values
 *
 */

template<class physics_t>
KERNEL
void evalFlavorsAtNodesKernel(const size_t* const activePaths, const size_t nPaths,
                       double* output, size_t nrhos, size_t nenergies, size_t nflv,
                            physics_t* nsq){

	for(size_t index_y = blockIdx.y; index_y < nPaths; index_y += gridDim.y){

		const size_t index_cosine = activePaths[index_y];

        for(size_t index_rho = 0; index_rho < nrhos; index_rho += 1){
            for(size_t index_flv = 0; index_flv < nflv; index_flv += 1){
                for(size_t index_energy = threadIdx.x + blockIdx.x * blockDim.x; index_energy < nenergies; index_energy += blockDim.x * gridDim.x){
                    output[index_cosine * nrhos * nflv * nenergies
                            + index_rho * nflv * nenergies
                            + index_flv * nenergies
                            + index_energy]
                        = nsq[index_cosine].evalFlavorAtNode(index_flv, index_rho, index_energy);
                }
            }
        }
	}
}

template<class physics_t>
KERNEL
void evalMassesAtNodesKernel(const size_t* const activePaths, const size_t nPaths,
                       double* output, size_t nrhos, size_t nenergies, size_t nflv,
                            physics_t* nsq){

	for(size_t index_y = blockIdx.y; index_y < nPaths; index_y += gridDim.y){

		const size_t index_cosine = activePaths[index_y];

        for(size_t index_rho = 0; index_rho < nrhos; index_rho += 1){
            for(size_t index_flv = 0; index_flv < nflv; index_flv += 1){
                for(size_t index_energy = threadIdx.x + blockIdx.x * blockDim.x; index_energy < nenergies; index_energy += blockDim.x * gridDim.x){
                    output[index_cosine * nrhos * nflv * nenergies
                            + index_rho * nflv * nenergies
                            + index_flv * nenergies
                            + index_energy]
                        = nsq[index_cosine].evalMassAtNode(index_flv, index_rho, index_energy);
                }
            }
        }
	}
}

/*
	The monolithic kernel of Version2
*/
template<typename Stepper, bool HI_only, class physics_t>
KERNEL
__launch_bounds__(128,4)
void evolveKernel(physics_t* nsqs, size_t n_cosines,
    ode::RKstats* stats, unsigned int nSteps, double h, double h_min, double h_max, double epsabs, double epsrel,
    double* solverworkspace)
{

    // ODE solver interface
    auto RHS = [](double t, double* const y, double* const y_derived, void* userdata) -> void{
        physics_t* nsq = static_cast<physics_t*>(userdata);

        if(HI_only){
            nsq->prederive_osc(t, y, y_derived);
            nsq->derive_osc();
            nsq->endDerive();
        }else{
            nsq->prederiveFull(t, y, y_derived);
            nsq->deriveFull();
            nsq->endDerive();
        }
    };

    // loop over paths
    for(size_t index_cosine = blockIdx.y; index_cosine < n_cosines; index_cosine += gridDim.y){

        physics_t& nsq = nsqs[index_cosine];

		double tbegin = nsq.get_track().getXBegin();
		double tend = nsq.get_track().getXEnd();

		size_t myworkspaceSize = ode::SolverGPU<Stepper>::getMinimumMemorySize(nsq.get_statesPitch(),physics_t::NumNeu*physics_t::NumNeu* nsq.get_n_rhos());
		double* myworkspace = solverworkspace + index_cosine * myworkspaceSize;

        ode::SolverGPU<Stepper> solver(myworkspace, nsq.get_states() ,
		                nsq.get_statesPitch(),physics_t::NumNeu*physics_t::NumNeu* nsq.get_n_rhos(),
		                nSteps, h, h_min, h_max,
		                tbegin, tend,
		                (void*)&nsq,
		                RHS);


		solver.setAbsolutePrecision(epsabs);
		solver.setRelativePrecision(epsrel);

		if(nSteps == 0){
		    solver.solveAdaptive();
		}else{
		    solver.solveFixed();
		}

		if(threadIdx.x + blockDim.x * blockIdx.x == 0){
            //printf("path %lu done\n", index_cosine);
            if(solver.stats.status == ode::Status::success){
                    nsq.get_track().setCurrentX(tend);
            }

            stats[index_cosine] = solver.stats;
		}
    }
}


/*
 *
 * GPU kernels to prederive and derive physics_t objects in Version1
 *
 */


template<class physics_t>
KERNEL
void setDerivationPointersKernel(const size_t* const activePaths, const size_t nPaths,
                        physics_t* nsq,
                        const double* y, double* y_derived){

	for(size_t index_y = blockIdx.y; index_y < nPaths; index_y += gridDim.y){

        const size_t index_cosine = activePaths[index_y];

		const double* my_y = getPitchedElement(y,
                                                index_cosine * nsq[index_cosine].get_n_rhos() * physics_t::NumNeu * physics_t::NumNeu,
                                                0,
                                                nsq[index_cosine].get_statesPitch());

		double* my_y_derived = getPitchedElement(y_derived,
                                                index_cosine * nsq[index_cosine].get_n_rhos() * physics_t::NumNeu * physics_t::NumNeu,
                                                0,
                                                nsq[index_cosine].get_statesPitch());

		nsq[index_cosine].setDerivationPointers(my_y, my_y_derived);
	}
}

template<class physics_t>
KERNEL
void endDeriveKernel(const size_t* const activePaths, const size_t nPaths,
                        physics_t* nsq){

	for(size_t index_y = blockIdx.y; index_y < nPaths; index_y += gridDim.y){

        const size_t index_cosine = activePaths[index_y];

		nsq[index_cosine].endDerive();
	}
}

template<class physics_t>
KERNEL
void updateTimeKernel(const double* const times, const size_t* const activePaths, const size_t nPaths,
                        physics_t* nsq){

	for(size_t index_y = blockIdx.y; index_y < nPaths; index_y += gridDim.y){
		const size_t index_cosine = activePaths[index_y];
		const double time = times[index_cosine];
        nsq[index_cosine].set_t(time);
		nsq[index_cosine].updateTracks();
	}
}

template<class physics_t>
KERNEL
void addToPrederiveKernel(const double* const times, const size_t* const activePaths, const size_t nPaths,
                        physics_t* nsq){

	for(size_t index_y = blockIdx.y; index_y < nPaths; index_y += gridDim.y){

		const size_t index_cosine = activePaths[index_y];
		const double time = times[index_cosine];

		nsq[index_cosine].addToPrederive(time);
	}
}

template<class physics_t>
KERNEL
void evolveProjectorsKernel(const double* const times, const size_t* const activePaths, const size_t nPaths,
                            physics_t* nsq){

	for(size_t index_y = blockIdx.y; index_y < nPaths; index_y += gridDim.y){

		const size_t index_cosine = activePaths[index_y];
		const double time = times[index_cosine];

		nsq[index_cosine].evolveProjectors(time);
	}
}

template<class physics_t>
KERNEL
void updateInteractionStructKernel(const size_t* const activePaths, const size_t nPaths,
                            physics_t* nsq){

	for(size_t index_y = blockIdx.y; index_y < nPaths; index_y += gridDim.y){

		const size_t index_cosine = activePaths[index_y];

		nsq[index_cosine].updateInteractionStruct();
	}
}

template<class physics_t>
KERNEL
void calculateCurrentFlavorFluxesKernel(const size_t* const activePaths, const size_t nPaths,
                            physics_t* nsq){

	for(size_t index_y = blockIdx.y; index_y < nPaths; index_y += gridDim.y){

		const size_t index_cosine = activePaths[index_y];

		nsq[index_cosine].calculateCurrentFlavorFluxes();
	}
}

template<class physics_t>
KERNEL
void updateNCArraysKernel(const size_t* const activePaths, const size_t nPaths,
                            physics_t* nsq){
	for(size_t index_y = blockIdx.y; index_y < nPaths; index_y += gridDim.y){

		const size_t index_cosine = activePaths[index_y];

		nsq[index_cosine].updateNCArrays();
	}
}

template<class physics_t>
KERNEL
void updateTauArraysKernelPart1(const size_t* const activePaths, const size_t nPaths,
                            physics_t* nsq){
	for(size_t index_y = blockIdx.y; index_y < nPaths; index_y += gridDim.y){

		const size_t index_cosine = activePaths[index_y];

		nsq[index_cosine].updateTauArraysPart1();
	}
}

template<class physics_t>
KERNEL
void updateTauArraysKernelPart2( const size_t* const activePaths, const size_t nPaths,
                            physics_t* nsq){
	for(size_t index_y = blockIdx.y; index_y < nPaths; index_y += gridDim.y){

		const size_t index_cosine = activePaths[index_y];

		nsq[index_cosine].updateTauArraysPart2();
	}
}

template<class physics_t>
KERNEL
void updateGRArraysKernel(const size_t* const activePaths, const size_t nPaths,
                            physics_t* nsq){
	for(size_t index_y = blockIdx.y; index_y < nPaths; index_y += gridDim.y){

		const size_t index_cosine = activePaths[index_y];

		nsq[index_cosine].updateGRArrays();
	}
}

template<class physics_t>
KERNEL
void deriveKernel(const size_t* const activePaths, const size_t nPaths,
                            physics_t* nsq){
	for(size_t index_y = blockIdx.y; index_y < nPaths; index_y += gridDim.y){

		const size_t index_cosine = activePaths[index_y];

		nsq[index_cosine].deriveFull();
	}
}

template<class physics_t>
KERNEL
void deriveOscKernel(const size_t* const activePaths, const size_t nPaths,
                            physics_t* nsq){
	for(size_t index_y = blockIdx.y; index_y < nPaths; index_y += gridDim.y){

		const size_t index_cosine = activePaths[index_y];

		nsq[index_cosine].derive_osc();
	}
}

/*
 *
 * Wrapper functions for above kernels
 *
 */

template<class physics_t>
void callEvolveKernel(physics_t* nsqs, size_t n_cosines, bool isOnlyOsc,
    ode::RKstats* stats, unsigned int nSteps, double h, double h_min, double h_max, double epsabs, double epsrel,
    ode::StepperType stepper, double* solverworkspace,
	dim3 grid, dim3 block, cudaStream_t stream = 0){

    switch(stepper){
        case ode::StepperType::RK4:
            if(isOnlyOsc){
                evolveKernel<ode::stepper::RK4, true><<<grid, block, 0, stream>>>(nsqs, n_cosines, stats, nSteps, h, h_min, h_max, epsabs, epsrel, solverworkspace);
            }else{
                evolveKernel<ode::stepper::RK4, false><<<grid, block, 0, stream>>>(nsqs, n_cosines, stats, nSteps, h, h_min, h_max, epsabs, epsrel, solverworkspace);
            }
            break;
        default: throw std::runtime_error("callEvolveKernel: invalid stepper.");
    }

    CUERR;

}

template<class physics_t>
void callEvalFlavorsAtNodesKernel(const size_t* activePaths, size_t nPaths,
                        double* output, size_t nrhos, size_t nenergies, size_t nflv,
                        physics_t* nsq,
                        dim3 grid, dim3 block, cudaStream_t stream = 0){

    evalFlavorsAtNodesKernel<<<grid, block,0, stream>>>(activePaths, nPaths, output, nrhos, nenergies, nflv, nsq);
    CUERR;
}

template<class physics_t>
void callEvalMassesAtNodesKernel(const size_t* activePaths, size_t nPaths,
                        double* output, size_t nrhos, size_t nenergies, size_t nflv,
                        physics_t* nsq,
                        dim3 grid, dim3 block, cudaStream_t stream = 0){

    evalMassesAtNodesKernel<<<grid, block,0, stream>>>(activePaths, nPaths, output, nrhos, nenergies, nflv, nsq);
    CUERR;
}

template<class physics_t>
void callSetDerivationPointersKernel(const size_t* const activePaths, const size_t nPaths,
                        physics_t* nsq,
                        const double* y, double* y_derived,
                        dim3 grid, dim3 block, cudaStream_t stream = 0){

    setDerivationPointersKernel<<<grid, block,0, stream>>>(activePaths, nPaths, nsq, y, y_derived);
    CUERR;
}

template<class physics_t>
void callEndDeriveKernel(const size_t* const activePaths, const size_t nPaths,
                        physics_t* nsq,
                        dim3 grid, dim3 block, cudaStream_t stream = 0){

    endDeriveKernel<<<grid, block,0, stream>>>(activePaths, nPaths, nsq);
    CUERR;
}

template<class physics_t>
void callUpdateTimeKernel(const double* const times, const size_t* const activePaths, const size_t nPaths,
                        physics_t* nsq,
                        dim3 grid, dim3 block, cudaStream_t stream = 0){

    updateTimeKernel<<<grid, block,0, stream>>>(times, activePaths, nPaths, nsq);
    CUERR;
}

template<class physics_t>
void callEvolveProjectorsKernel(const double* const times, const size_t* const activePaths, const size_t nPaths,
                        physics_t* nsq,
                        dim3 grid, dim3 block, cudaStream_t stream = 0){

    evolveProjectorsKernel<<<grid, block,0, stream>>>(times, activePaths, nPaths, nsq);
    CUERR;
}

template<class physics_t>
void callUpdateInteractionStructKernel(const size_t* const activePaths, const size_t nPaths,
                            physics_t* nsq,
                            dim3 grid, dim3 block, cudaStream_t stream = 0){

    updateInteractionStructKernel<<<grid, block, 0, stream>>>(activePaths, nPaths, nsq);
    CUERR;
}

template<class physics_t>
void callCalculateCurrentFlavorFluxesKernel(const size_t* const activePaths, const size_t nPaths,
                            physics_t* nsq,
                            dim3 grid, dim3 block, cudaStream_t stream = 0){

    calculateCurrentFlavorFluxesKernel<<<grid, block, 0, stream>>>(activePaths, nPaths, nsq);
    CUERR;
}

template<class physics_t>
void callUpdateNCArraysKernel(const size_t* const activePaths, const size_t nPaths,
                            physics_t* nsq,
                            dim3 grid, dim3 block, cudaStream_t stream = 0){

    updateNCArraysKernel<<<grid, block, 0, stream>>>(activePaths, nPaths, nsq);
    CUERR;
}

template<class physics_t>
void callUpdateTauArraysKernelPart1(const size_t* const activePaths, const size_t nPaths,
                            physics_t* nsq,
                            dim3 grid, dim3 block, cudaStream_t stream = 0){

    updateTauArraysKernelPart1<<<grid, block, 0, stream>>>(activePaths, nPaths, nsq);
    CUERR;
}

template<class physics_t>
void callUpdateTauArraysKernelPart2(const size_t* const activePaths, const size_t nPaths,
                            physics_t* nsq,
                            dim3 grid, dim3 block, cudaStream_t stream = 0){

    updateTauArraysKernelPart2<<<grid, block, 0, stream>>>(activePaths, nPaths, nsq);
    CUERR;
}

template<class physics_t>
void callUpdateGRArraysKernel(const size_t* const activePaths, const size_t nPaths,
                            physics_t* nsq,
                            dim3 grid, dim3 block, cudaStream_t stream = 0){

    updateGRArraysKernel<<<grid, block, 0, stream>>>(activePaths, nPaths, nsq);
    CUERR;
}

template<class physics_t>
void callDeriveKernel(const size_t* const activePaths, const size_t nPaths,
                            physics_t* nsq,
                            dim3 grid, dim3 block, cudaStream_t stream = 0){

    deriveKernel<<<grid, block, 0, stream>>>(activePaths, nPaths, nsq);
    CUERR;
}

template<class physics_t>
void callDeriveOscKernel(const size_t* const activePaths, const size_t nPaths,
                            physics_t* nsq,
                            dim3 grid, dim3 block, cudaStream_t stream = 0){

    deriveOscKernel<<<grid, block, 0, stream>>>(activePaths, nPaths, nsq);
    CUERR;
}

template<class physics_t>
void callAddToPrederiveKernel(const double* const times, const size_t* const activePaths, const size_t nPaths,
                        physics_t* nsq,
                        dim3 grid, dim3 block, cudaStream_t stream = 0){

    addToPrederiveKernel<<<grid, block, 0, stream>>>(times, activePaths, nPaths, nsq);
    CUERR;
}
}

#endif
