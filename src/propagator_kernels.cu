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

#include <cudanuSQuIDS/propagator_kernels.cuh>
#include <cudanuSQuIDS/cudautils.cuh>

#include <nuSQuIDS/nuSQuIDS.h>

namespace cudanusquids{

	KERNEL
	void setInitialStatesKernel(double* const states, const double* const initialFlux, const double* const b0proj, const double* const b1proj,
                                const size_t n_cosines, const size_t n_rhos, const size_t n_flvs, const size_t n_energies,
                                const nusquids::Basis basis,
                                const size_t statesOffset, size_t statesPitch){

		for(size_t index_cosine = blockIdx.y; index_cosine < n_cosines; index_cosine += gridDim.y){
			for(size_t index_rho = 0; index_rho < n_rhos; index_rho++){

                double* statesCosRho = getPitchedElement(states,
                                        index_cosine * n_rhos * n_flvs * n_flvs
                                            + index_rho * n_flvs * n_flvs,
                                        0,
                                        statesPitch);

				for(size_t index_energy = threadIdx.x + blockDim.x * blockIdx.x; index_energy < n_energies; index_energy += blockDim.x * gridDim.x){

                    double* state = statesCosRho + index_energy;

					for(size_t i = 0; i <  n_flvs * n_flvs; i++){
						state[i * statesOffset] = 0;
					}

					const double* myFlux = initialFlux
								+ index_cosine * n_rhos * n_energies * n_flvs
								+ index_rho * n_energies * n_flvs
								+ index_energy * n_flvs;

					for(size_t flv = 0;  flv < n_flvs; flv++){
                        const double* projptr = nullptr;
                        size_t projoffset = 0;

						if(basis == nusquids::Basis::mass){
                            projptr = b0proj + flv;
                            projoffset = n_flvs;
						}else{
                            projptr = b1proj + index_rho * n_flvs + flv;
                            projoffset = n_rhos * n_flvs;
						}

						for(size_t i = 0; i <  n_flvs * n_flvs; i++){
                            state[i * statesOffset] += myFlux[flv] * projptr[i * projoffset];
						}
					}
				}
			}
		}
	}

	void callSetInitialStatesKernel(double* const states, const double* const initialFlux, const double* const b0proj, const double* const b1proj,
                                const size_t n_cosines, const size_t n_rhos, const size_t n_flvs, const size_t n_energies,
                                const nusquids::Basis basis,
                                const size_t statesOffset, size_t statesPitch,
                                dim3 grid, dim3 block, cudaStream_t stream){

        setInitialStatesKernel<<<grid, block, 0, stream>>>(states, initialFlux, b0proj, b1proj, n_cosines, n_rhos, n_flvs, n_energies, basis, statesOffset, statesPitch);
        CUERR;
	}

}
