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

#ifndef CUDANUSQUIDS_CUSTOMPHYSICS_CUH
#define CUDANUSQUIDS_CUSTOMPHYSICS_CUH

#include <cudanuSQuIDS/cudahelpers.cuh>
#include <cudanuSQuIDS/physics.cuh>

// AdditionalData is cast to this type to avoid messing with the raw void** pointers
struct Helper{
        const double* __restrict__ const NSI;
        double* __restrict__ const NSIevolved;
};

/*
	CUDAnuSQuIDS uses static polymorphism for physics operators. If CustomPhysicsOps is passed to CudaNusquids,
	the physics operators of CustomPhysicsOps will be used instead of the default operators.

*/

struct CustomPhysicsOps{
    cudanusquids::PhysicsOps baseOps; // make the default physics available, if required.

    HOSTDEVICEQUALIFIER
    CustomPhysicsOps(){}
    HOSTDEVICEQUALIFIER
    ~CustomPhysicsOps(){}

    template<class Physics>
    DEVICEQUALIFIER
    void addToPrederive(Physics& base, double time) const{
        baseOps.addToPrederive(base, time);

	/*
		Evolve NSI and store in NSIevolved. Assumes same hamiltonian for neutrinos/antineutrinos
	*/

        Helper* helper = (Helper*)base.get_additionalData();

	//for better performance we store the matrices like a struct of arrays, i.e. every 0th element, then every 1st element,...
	//Then, to access the i-th element of a matrix, we use [i * NSIevolvedoffset] instead of [i]
	const size_t NSIevolvedoffset = base.get_n_energies();
        double* NSIevolvedptr = helper->NSIevolved
                                + base.get_indexInBatch() * base.get_n_energies() * Physics::NFLV * Physics::NFLV; // get data of correct path

	//loop over energies, using one thread per energy, if possible
        for(size_t index_energy = threadIdx.x + blockIdx.x * blockDim.x;
            index_energy < base.get_n_energies();
            index_energy += blockDim.x * gridDim.x){

		//Get vacuum hamiltonian. Similar to NSIevolvedoffset, the hamiltonian will be accessed via [i * base.get_h0offset()]
		const double* h0data = getPitchedElement(base.get_H0_array(), 0, index_energy, base.get_h0pitch());

		double NSIlocal[Physics::NFLV * Physics::NFLV];
		double NSIevolvedlocal[Physics::NFLV * Physics::NFLV];

		//copy NSI from global memory to registers (or local memory)
		#pragma unroll
		for(unsigned int i = 0; i < Physics::NFLV * Physics::NFLV; i++)
			NSIlocal[i] = helper->NSI[i];

		//Evolve NSIlocal using the vacuum hamiltonian and store result in NSIevolvedlocal
		cudanusquids::sumath::evolve(NSIevolvedlocal, NSIlocal, time, h0data, base.get_h0offset());

		//copy NSIevolvedlocal to global memory
		#pragma unroll
		for(unsigned int i = 0; i < Physics::NFLV * Physics::NFLV; i++)
			NSIevolvedptr[index_energy + i * NSIevolvedoffset] = NSIevolvedlocal[i];
        }
    }

    template<class Physics>
    DEVICEQUALIFIER
    void H0(const Physics& base, double out[], size_t index_rho, size_t index_energy) const{
        baseOps.H0(base, out, index_rho, index_energy);
    }

    template<class Physics>
    DEVICEQUALIFIER
    void HI(const Physics& base, double out[],
            size_t index_rho, size_t index_energy) const{
        //calculate default potential
		baseOps.HI(base, out, index_rho, index_energy);

        //calculate new potential, asumming same hamiltonian for neutrinos/antineutrinos
		double CC = cudanusquids::Const::HI_constants() * base.get_density() * base.get_electronFraction();
		if((index_rho == 1 && base.get_neutrinoType() == nusquids::NeutrinoType::both) || base.get_neutrinoType() == nusquids::NeutrinoType::antineutrino){
                CC *= -1;
		}

		double potential[Physics::NFLV * Physics::NFLV];
		double NSIevolvedlocal[Physics::NFLV * Physics::NFLV];
        Helper* helper = (Helper*)base.get_additionalData();
		double* NSIevolvedptr = helper->NSIevolved
                                + base.get_indexInBatch() * base.get_n_energies() * Physics::NFLV * Physics::NFLV // get data of correct path
                                + index_energy; // get data of correct energy bin
		const size_t NSIevolvedoffset = base.get_n_energies();

		#pragma unroll
		for(unsigned int i = 0; i < Physics::NFLV * Physics::NFLV; i++)
			NSIevolvedlocal[i] = NSIevolvedptr[i * NSIevolvedoffset];

		#pragma unroll
		for(unsigned int i = 0; i < Physics::NFLV * Physics::NFLV; i++)
			potential[i] = 3.0 * CC * NSIevolvedlocal[i];

		//add potential to default potential
		#pragma unroll
		for(unsigned int i = 0; i < Physics::NFLV * Physics::NFLV; i++)
			out[i] += potential[i];
    }

    template<class Physics>
    DEVICEQUALIFIER
    void GammaRho(const Physics& base, double out[],
                    size_t index_rho, size_t index_energy) const{
        baseOps.GammaRho(base, out, index_rho, index_energy);
    }

    template<class Physics>
    DEVICEQUALIFIER
    void InteractionsRho(const Physics& base, double out[],
                                size_t index_rho, size_t index_energy) const{
        baseOps.InteractionsRho(base, out, index_rho, index_energy);
    }
};

#endif
