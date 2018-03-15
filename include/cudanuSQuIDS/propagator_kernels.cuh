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

#ifndef CUDANUSQUIDS_PROPAGATOR_KERNELS_CUH
#define CUDANUSQUIDS_PROPAGATOR_KERNELS_CUH

#include <cudanuSQuIDS/cudahelpers.cuh>
#include <cudanuSQuIDS/const.cuh>
#include <cudanuSQuIDS/cudautils.cuh>
#include <cudanuSQuIDS/physics.cuh>
#include <cudanuSQuIDS/sumath.cuh>

#include <nuSQuIDS/nuSQuIDS.h>

namespace cudanusquids{

	void callSetInitialStatesKernel(double* const states, const double* const initialFlux, const double* const b0proj, const double* const b1proj,
                                const size_t n_cosines, const size_t n_rhos, const size_t n_flvs, const size_t n_energies,
                                const nusquids::Basis basis,
                                const size_t statesOffset, size_t statesPitch,
                                dim3 grid, dim3 block, cudaStream_t stream = 0);

/*
 *
 * Kernels to initialize GPU data used by Physics objects
 *
 */

    template<class physics_t>
    KERNEL
    void initH0arrayKernel(double* const H0_array, const size_t h0pitch, const size_t h0offset, physics_t* nsq){

        for(size_t index_energy = threadIdx.x + blockDim.x * blockIdx.x; index_energy < nsq[0].get_n_energies(); index_energy += blockDim.x * gridDim.x){

            double* h0data = getPitchedElement(H0_array, 0, index_energy, h0pitch);

            double tmp[physics_t::NumNeu * physics_t::NumNeu];
            nsq[0].H0(tmp, 0, index_energy);

	        for(size_t i = 0; i < physics_t::NumNeu * physics_t::NumNeu; ++i){
                	h0data[i * h0offset] = tmp[i];
	        }
        }
    }

    template<typename T>
    void callPlacementNewAndAssignKernel(T* newptr, T* toAssign, size_t n, dim3 grid, dim3 block, cudaStream_t stream = 0){

        placementNewAndAssignKernel<<<grid, block, 0, stream>>>(newptr, toAssign, n);
        CUERR;
    }

    template<class physics_t>
    void callInitH0arrayKernel(double* const H0_array, const size_t h0pitch, const size_t h0offset, physics_t* nsq,
                                dim3 grid, dim3 block, cudaStream_t stream = 0){

        initH0arrayKernel<<<grid, block, 0, stream>>>(H0_array, h0pitch, h0offset, nsq);
        CUERR;
    }
}

#endif
