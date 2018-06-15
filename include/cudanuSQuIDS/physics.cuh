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

#ifndef CUDANUSQUIDS_PHYSICS_CUH
#define CUDANUSQUIDS_PHYSICS_CUH

#include <cudanuSQuIDS/cudahelpers.cuh>
#include <cudanuSQuIDS/bodygpu.cuh>
#include <cudanuSQuIDS/interaction_structure.hpp>
#include <cudanuSQuIDS/const.cuh>
#include <cudanuSQuIDS/sumath.cuh>
#include <cudanuSQuIDS/types.hpp>
#include <cudanuSQuIDS/cudautils.cuh>

#include <nuSQuIDS/nuSQuIDS.h>

#if __CUDACC_VER_MAJOR__ >= 9
    #include <cooperative_groups.h>
    using namespace cooperative_groups;
#endif


namespace cudanusquids{

    #define GETSET(name, type) __host__ __device__ type get_##name() const{\
                                    return name; \
                                }\
                                __host__ __device__ type& get_##name(){\
                                    return name; \
                                }\
                               __host__ __device__ void set_##name(type a##name) {\
                                    name = a##name ; \
                                }

/// \class Physics
/// \brief Handles the derivation of all bins of a specific neutrino path at a specific time
template<int NFLV_, class body_t, class Op_t>
struct Physics{

    // change this if more number of flavors are supported in sumath.cuh
	static_assert(3 <= NFLV_ && NFLV_ <= 4, "Physics: not (3 <= NFLV && NFLV <= 4)");

    static constexpr int NFLV = NFLV_;
    using BodyType = body_t;
    using Ops = Op_t; // the default Op_t in CudaNusquids is struct PhysicsOps

/*
        Op_t has to provide the following functions:

        __host__ __device__
        PhysicsOps();

        __host__ __device__
        ~PhysicsOps();

        template<class Physics>
        __device__
        void addToPrederive(Physics& base, double time) const;

        template<class Physics>
        __device__
        void H0(const Physics& base, double out[], int index_rho, int index_energy) const;

        template<class Physics>
        __device__
        void HI(const Physics& base, double out[],
                int index_rho, int index_energy) const;

        template<class Physics>
        __device__
        void GammaRho(const Physics& base, double out[],
                        int index_rho, int index_energy) const;

        template<class Physics>
        __device__
        void InteractionsRho(const Physics& base, double out[],
                                    int index_rho, int index_energy) const;
*/

    Ops ops;

    /// \brief List of energy bins. length n_energies
	const double* energyList;
    /// \brief pointer to current state. during derivation, cstates == y, else cstates == states
    const double* cstates;
	const double* y; // set by solver
    /// \brief mass projectors.
    /// \details density matrix, NFLV x NFLV. The i-th entry for mass basis f is b0proj[f + b0offset * i];
    const double* b0proj;
    /// \brief flavor projectors.
    /// \details density matrix, NFLV x NFLV. The i-th entry for flavor basis f with index_rho is b1proj[index_rho * NFLV + flv + i * b1offset]
	const double* b1proj;
    /// \brief Matrix of mass differences.
    /// \details density matrix, NFLV x NFLV
    const double* dm2;
    /// \brief Time-independent hamiltonion for each energy bin.
    /// \details density matrix, NFLV x NFLV. Get pointer to first element of energy bin index_energy via
    /// double* h0data = getPitchedElement(H0_array, 0, index_energy, h0pitch);
    /// then get i-th matrix element with h0data[i * h0offset];
	const double* H0_array;
    /// \brief Energy difference between the energy bins
    /// \details Length n_energies. delE[i+1] = energyList[i+1] - energyList[i], delE[0] = 0.0
	const double* delE;

    GETSET(energyList, const double*)
    GETSET(cstates, const double*)
    GETSET(y, const double*)
    GETSET(b0proj, const double*)
    GETSET(b1proj, const double*)
    GETSET(dm2, const double*)
    GETSET(H0_array, const double*)
    GETSET(delE, const double*)

    /// \brief Neutrino states
    /// \details density matrix, NFLV x NFLV. Get pointer to first element of density matrix of energy bin index_energy and index_rho via
    /// double* statedata = getPitchedElement(cstates, index_rho * NFLV * NFLV, index_energy, statesPitch);
    /// then get i-th matrix element with statedata[i * statesOffset];
	double* states;
	double* y_derived; // set by solver, output of derivation step
    /// \brief Flavor projectors in the interaction picture
    /// \details density matrix, NFLV x NFLV. Get pointer to first element of specific density matrix via
    /// double* evolb1data = getPitchedElement(evolB1proj, index_rho * NFLV * NFLV * NFLV + index_flv * NFLV * NFLV, index_energy, evolB1pitch);
    /// then get i-th matrix element with evolb1data[i * evoloffset];
    double* evolB1proj;

    /// \brief The current neutrino fluxes
    /// \details Stores the current neutrino fluxes during a derivation step. Get pointer via
    /// double* fluxptr = getPitchedElement(fluxes,index_rho * NFLV, flv * fluxOffset + index_energy, fluxPitch)
    /// Then get flux with double flux = *fluxptr;
	double* fluxes;
    //helper arrays to store interaction results
    double* nc_array;
	double* tau_decay_fluxes;
	double* tau_bar_decay_fluxes;
	double* tau_hadlep_array;
	double* tau_lep_array;
	double* gr_array;

    GETSET(states, double*)
    GETSET(y_derived, double*)
    GETSET(evolB1proj, double*)
    GETSET(fluxes, double*)
    GETSET(nc_array, double*)
    GETSET(tau_decay_fluxes, double*)
    GETSET(tau_bar_decay_fluxes, double*)
    GETSET(tau_hadlep_array, double*)
    GETSET(tau_lep_array, double*)
    GETSET(gr_array, double*)

    /// \brief user-defined arrays specified via ParameterObject::registerAdditionalData
    void** additionalData;

    GETSET(additionalData, void**)

    /// \brief The current density
	double density;
    /// \brief The current electron fraction
	double electronFraction;
    /// \brief The current time
	double t;

    GETSET(density, double)
    GETSET(electronFraction, double)
    GETSET(t, double)

    /// \brief maximum number of simultaneous paths
    int max_n_cosines;
    /// \brief Number of neutrino types. n_rhos = 2 if neutrinoType == both, else n_rhos = 1
	int n_rhos;
    /// \brief Number of energy bins
	int n_energies;
    /// \brief The cosine bin local to the current batch. indexInBatch < max_n_cosines
    int indexInBatch;
    /// \brief The cosine bin
    int globalPathId;

    // variables which specify memory layout
    int b0offset;
	int b1offset;
	int evolB1pitch;
	int evoloffset;
	int h0pitch;
	int h0offset;
	int statesOffset;
	int statesPitch;
	int fluxOffset;
	int fluxPitch;

    GETSET(max_n_cosines, int)
    GETSET(n_rhos, int)
    GETSET(n_energies, int)
    GETSET(indexInBatch, int)
    GETSET(globalPathId, int)
    GETSET(b0offset, int)
    GETSET(b1offset, int)
    GETSET(evolB1pitch, int)
    GETSET(evoloffset, int)
    GETSET(h0pitch, int)
    GETSET(h0offset, int)
    GETSET(statesOffset, int)
    GETSET(statesPitch, int)
	GETSET(fluxOffset, int)
    GETSET(fluxPitch, int)

    /// \brief The body
	BodyType body;
    /// \brief The track
	typename BodyType::Track track;

    GETSET(body, BodyType)
    GETSET(track, typename BodyType::Track)

    /// \brief Cross-section lookup table
	InteractionStructureGpu intstruct;
    /// \brief Inverse interaction lengths lookup table
	InteractionStateGpu intstate;

    GETSET(intstruct, InteractionStructureGpu)
    GETSET(intstate, InteractionStateGpu)

    /// \brief Basis of states. (mass or interaction)
	nusquids::Basis basis;
	detail::Flags flags;
    /// \brief NeutrinoType
	nusquids::NeutrinoType neutrinoType;

    GETSET(basis, nusquids::Basis)
    GETSET(flags, detail::Flags)
    GETSET(neutrinoType, nusquids::NeutrinoType)

	HOSTDEVICEQUALIFIER
	Physics(){}

    HOSTDEVICEQUALIFIER
	~Physics(){}

	DEVICEQUALIFIER
	void barrier_path() const{
        __syncthreads();
	}

    DEVICEQUALIFIER
    void endDerive(){
        if(threadIdx.x + blockIdx.x * blockDim.x == 0){
            cstates = states;
        }
        barrier_path();
    }


    /*
     *
     * Expectation values
	 *
     */

    DEVICEQUALIFIER
    double getExpectationValue(const double* op, int opoffset, int index_rho, int index_energy) const{
        const double* h0data = getPitchedElement(H0_array, 0, index_energy, h0pitch);

        double opl[NFLV * NFLV];
        double evolvedOp[NFLV * NFLV];

        #pragma unroll
        for(int i = 0; i < NFLV * NFLV; i++)
            opl[i] = op[i * opoffset];

        sumath::evolve(evolvedOp, opl, t, h0data, h0offset);

        const double* statedata = getPitchedElement(cstates,
                                        index_rho * NFLV * NFLV,
                                        index_energy,
                                        statesPitch);

        #pragma unroll
        for(int i = 0; i < NFLV * NFLV; i++){
            opl[i] = statedata[i*statesOffset];
        }

        return sumath::sutrace(evolvedOp, opl);
    }

    DEVICEQUALIFIER
    double evalMassAtNode(int flv, int index_rho, int index_energy) const{
        double b0[NFLV * NFLV];

        #pragma unroll
        for(int i = 0; i < NFLV * NFLV; i++){
            b0[i] = b0proj[flv + b0offset * i];
        }

        if(basis == nusquids::Basis::mass){
            double state[NFLV * NFLV];

            const double* statedata = getPitchedElement(cstates,
                                            index_rho * NFLV * NFLV,
                                            index_energy,
                                            statesPitch);

            #pragma unroll
            for(int i = 0; i < NFLV * NFLV; i++){
                state[i] = statedata[i*statesOffset];
            }

	    return sumath::sutrace(b0, state);
        }

        return getExpectationValue(b0, 1, index_rho, index_energy);
    }

    DEVICEQUALIFIER
    double evalFlavorAtNode(int flv, int index_rho, int index_energy) const{
        double b1[NFLV * NFLV];

        const double* b1data = b1proj
                            + index_rho * NFLV
                            + flv;

        #pragma unroll
        for(int i = 0; i < NFLV * NFLV; i++){
            b1[i] = b1data[i * b1offset];
        }

        if(basis == nusquids::Basis::mass){
            double state[NFLV * NFLV];

            const double* statedata = getPitchedElement(cstates,
                                            index_rho * NFLV * NFLV,
                                            index_energy,
                                            statesPitch);

            #pragma unroll
            for(int i = 0; i < NFLV * NFLV; i++){
                state[i] = statedata[i*statesOffset];
            }

	    return sumath::sutrace(b1, state);
        }

        const double* statedata2 = getPitchedElement(cstates,
                                        index_rho * NFLV * NFLV,
                                        index_energy,
                                        statesPitch);

        return getExpectationValue(b1, 1, index_rho, index_energy);
    }




/*
    Prederive functions
*/

    DEVICEQUALIFIER
    void setDerivationPointers(const double* yin, double* yout){
        if(threadIdx.x + blockIdx.x * blockDim.x == 0){
            y = yin;
            y_derived = yout;

            cstates = y;
        }
        barrier_path();
    }

    DEVICEQUALIFIER
    double getNucleonNumber() const{
        double num_nuc = (Const::gr() / (Const::cm() * Const::cm() * Const::cm())) * density * 2.0
                            /(Const::proton_mass() + Const::neutron_mass());

        if(num_nuc < 1.0e-10 ){
            num_nuc = Const::Na() / (Const::cm() * Const::cm() * Const::cm()) * 1.0e-10;
        }

        return num_nuc;
    };

    DEVICEQUALIFIER
    void evolveProjectors(double time) const{

        for(int index_energy = threadIdx.x + blockIdx.x * blockDim.x;
            index_energy < n_energies;
            index_energy += blockDim.x * gridDim.x){

            const double* h0data = getPitchedElement(H0_array, 0, index_energy, h0pitch);

            double evolbuf[NFLV * (NFLV - 1)];

            sumath::prepareEvolution(evolbuf, time, h0data, h0offset);

            for(int index_rho = 0; index_rho < n_rhos; index_rho++){

                #pragma unroll
                for(int index_flv = 0; index_flv < NFLV; index_flv++){
                    double proj[NFLV * NFLV];
                    double evolproj[NFLV * NFLV];

                    const double* b1data = b1proj
                                        + index_rho * NFLV
                                        + index_flv;

                    double* evoldata = getPitchedElement(evolB1proj, index_rho * NFLV * NFLV * NFLV + index_flv * NFLV * NFLV,
                                                    index_energy,
                                                    evolB1pitch);

                    #pragma unroll
                    for(int i = 0; i < NFLV * NFLV; i++){
                        proj[i] = b1data[i * b1offset];
                    }

                    sumath::evolve(evolproj, proj, evolbuf);

                    #pragma unroll
                    for(int i = 0; i < NFLV * NFLV; i++){
                        evoldata[i * evoloffset] = evolproj[i];
                    }
                }
            }
        }
    }

    DEVICEQUALIFIER
    void updateInteractionStruct() const{

        const bool useGlashowInt = (flags.useGlashowResonance && (neutrinoType == nusquids::NeutrinoType::both || neutrinoType == nusquids::NeutrinoType::antineutrino));
        const int antirho = (neutrinoType == nusquids::NeutrinoType::both ? 1 : 0);

        const double nucleonNumber = getNucleonNumber();

        double* invlengr_ptr = &(intstate.invlen_GR[0]);

        for(int index_rho = 0; index_rho < n_rhos; index_rho++){
            const int invlenindex = index_rho * NFLV * n_energies;

            double* invlennc_ptr = &(intstate.invlen_NC[invlenindex]);
            double* invlencc_ptr = &(intstate.invlen_CC[invlenindex]);
            double* invlenint_ptr = &(intstate.invlen_INT[invlenindex]);

            for(int index_energy = threadIdx.x + blockIdx.x * blockDim.x;
                    index_energy < n_energies;
                    index_energy += blockDim.x * gridDim.x){

                for(int index_flv = 0; index_flv < NFLV; index_flv++){

                    double invlenNC = intstruct.sigma_NC[index_rho * NFLV * n_energies + index_flv * n_energies + index_energy];
                	double invlenCC = intstruct.sigma_CC[index_rho * NFLV * n_energies + index_flv * n_energies + index_energy];

	                invlenNC *= nucleonNumber;
	                invlenCC *= nucleonNumber;

               		double invlenINT = invlenNC + invlenCC;

                        invlennc_ptr[index_flv * n_energies + index_energy] = invlenNC;
                        invlencc_ptr[index_flv * n_energies + index_energy] = invlenCC;

                	if(index_flv == 0 && useGlashowInt && index_rho == antirho){
		            const double sigmagr = intstruct.sigma_GR[index_energy];
		            const double electrons = nucleonNumber * electronFraction;
		            const double invlengr = sigmagr * electrons;
                    	    invlenINT += invlengr;
                    	    invlengr_ptr[index_energy] = invlengr;
                	}

                   	invlenint_ptr[index_flv * n_energies + index_energy] = invlenINT;
                }
            }
        }
    }

    DEVICEQUALIFIER
    void calculateCurrentFlavorFluxes() const{

        for(int index_rho = 0; index_rho < n_rhos; index_rho++){

            double* fluxCosRho = getPitchedElement(fluxes,
                                            index_rho * NFLV,
                                            0,
                                            fluxPitch);


            for(int index_energy = threadIdx.x + blockIdx.x * blockDim.x;
                index_energy < n_energies;
                index_energy += blockDim.x * gridDim.x){

                double state[NFLV * NFLV];

                const double* statedata = getPitchedElement(cstates,
                                                index_rho * NFLV * NFLV,
                                                index_energy,
                                                statesPitch);

                #pragma unroll
                for(int i = 0; i < NFLV * NFLV; i++){
                    state[i] = statedata[i*statesOffset];
                }

                for(int flv = 0; flv < NFLV; flv++){
                    double proj[NFLV * NFLV];

                    const double* evoldata = getPitchedElement(evolB1proj,
                                                    index_rho * NFLV * NFLV * NFLV + flv * NFLV * NFLV,
                                                    index_energy,
                                                    evolB1pitch);

                    #pragma unroll
                    for(int i = 0; i < NFLV * NFLV; i++){
                        proj[i] = evoldata[i*evoloffset];
                    }

                    const double flux = sumath::sutrace(proj, state);
                    fluxCosRho[flv * fluxOffset + index_energy] = flux;
                }
            }
        }
    }

    DEVICEQUALIFIER
    void updateNCArrays() const{

        for(int index_rho = 0; index_rho < n_rhos; index_rho++){

            const int ncarrayBaseIndex = index_rho * NFLV * n_energies;

            const double* fluxCosRho = getPitchedElement(fluxes,
                                            index_rho * NFLV,
                                            0,
                                            fluxPitch);

            const double* invlennc = &(intstate.invlen_NC[index_rho * NFLV * n_energies]);
            const double* dndenc = &(intstruct.dNdE_NC[index_rho * NFLV * n_energies * n_energies]);

            for(int e1 = threadIdx.x + blockIdx.x * blockDim.x; e1 < n_energies; e1 += blockDim.x * gridDim.x){

                double e1_contribution[3] = {0.0, 0.0, 0.0};

                for(int e2 = 1; e2 < n_energies; e2++){

                    if(e1 < e2){


                        const double mydelE = delE[e2];

                        // calculate contribution for e, mu, tau
                        double invlen[3];
                        double dnde[3];
                        double flux[3];

                        #pragma unroll
                        for(int flv = 0; flv < 3; flv++){
                            invlen[flv] = invlennc[flv * n_energies + e2];
                            flux[flv] = fluxCosRho[flv * fluxOffset + e2];
                        }

                        #pragma unroll
                        for(int flv = 0; flv < 3; flv++){
                            dnde[flv] = dndenc[flv * n_energies * n_energies + e2 * n_energies + e1];
                        }

                        #pragma unroll
                        for(int flv = 0; flv < 3; flv++){
                            e1_contribution[flv] += flux[flv] * invlen[flv] * mydelE * dnde[flv];
                        }
                    }
                }

                #pragma unroll
                for(int flv = 0; flv < 3; flv++){
                    nc_array[ncarrayBaseIndex + flv * n_energies + e1] = e1_contribution[flv];
                }
            }
        }
    }

    DEVICEQUALIFIER
    void updateTauArraysPart1() const{

        constexpr int tau_flv = 2;

            double* decay_flux[2];
            decay_flux[0] = tau_decay_fluxes;
            decay_flux[1] = tau_bar_decay_fluxes;

            double* invlencc[2];
            invlencc[0] = &(intstate.invlen_CC[0 * NFLV * n_energies + tau_flv * n_energies]);
            invlencc[1] = &(intstate.invlen_CC[1 * NFLV * n_energies + tau_flv * n_energies]);


            double* dndecc[2];
            dndecc[0] = &(intstruct.dNdE_CC[0 * NFLV * n_energies * n_energies + tau_flv * n_energies * n_energies]);
            dndecc[1] = &(intstruct.dNdE_CC[1 * NFLV * n_energies * n_energies + tau_flv * n_energies * n_energies]);

            for(int et = threadIdx.x + blockIdx.x * blockDim.x; et < n_energies; et += blockDim.x * gridDim.x){

            double contribution[2] = {0.0, 0.0};

            if(0 < et){
                const double delEet = delE[et];

                #pragma unroll
                for(int rho = 0; rho < 2; rho++){

                    const double* fluxCosRho = getPitchedElement(fluxes,
                                        rho * NFLV,
                                        0,
                                        fluxPitch);

                    for(int en = 2; en < n_energies; en++){


                        if(et < en){

                            const double delEen = delE[en];
                            const double invlen = invlencc[rho][en];

                            double flux = fluxCosRho[tau_flv * fluxOffset + en];

                            if(flux < 0) flux = 0;

                            const double fluxinvlen = flux * invlen * delEen;
                            const double dndecc_ = dndecc[rho][en * n_energies + et];
                            contribution[rho] += fluxinvlen * dndecc_ * delEet;
                        }
                    }
                }
            }

            #pragma unroll
            for(int rho = 0; rho < 2; rho++){
                decay_flux[rho][et] = contribution[rho];
            }
        }
    }

    DEVICEQUALIFIER
    void updateTauArraysPart2() const{

            for(int en = threadIdx.x + blockDim.x * blockIdx.x; en < n_energies; en += blockDim.x * gridDim.x){
                double tau_hadlep_contribution = 0.0;
                double tau_bar_hadlep_contribution = 0.0;
                double tau_lep_contribution = 0.0;
                double tau_bar_lep_contribution = 0.0;

                for(int et = 1; et < n_energies - 1; et++){

                    if(en < et){

                        const double tau_decay_flux = tau_decay_fluxes[et];
                        const double tau_bar_decay_flux = tau_bar_decay_fluxes[et];

                        const double* dndetauallptr = &(intstruct.dNdE_tau_all[et * n_energies]);
                        const double* dndetaulepptr = &(intstruct.dNdE_tau_lep[et * n_energies]);

                        const double dndetauall = dndetauallptr[en];
                        const double dndetaulep = dndetaulepptr[en];

                        tau_hadlep_contribution += tau_decay_flux * dndetauall;
                        tau_lep_contribution += tau_bar_decay_flux * dndetaulep;

                        tau_bar_hadlep_contribution +=  tau_bar_decay_flux * dndetauall;
                        tau_bar_lep_contribution += tau_decay_flux * dndetaulep;
                    }
                }

                tau_hadlep_array[0 * n_energies + en] = tau_hadlep_contribution;
                tau_lep_array[0 * n_energies + en] = tau_lep_contribution;

                tau_hadlep_array[1 * n_energies + en] = tau_bar_hadlep_contribution;
                tau_lep_array[1 * n_energies + en] = tau_bar_lep_contribution;
            }
    }

    DEVICEQUALIFIER
    void updateGRArrays() const{

        constexpr unsigned int electronflavor = 0;

        const int index_rho = (neutrinoType == nusquids::NeutrinoType::both ? 1 : 0);

        const double* fluxCosRho = getPitchedElement(fluxes,
                                            index_rho * NFLV,
                                            0,
                                            fluxPitch);

        for(int e1 = threadIdx.x + blockIdx.x * blockDim.x; e1 < n_energies; e1 += blockDim.x * gridDim.x){
            double e1_contribution = 0.0;

            for(int e2 = 1; e2 < n_energies; e2++){


                if(e1 < e2){
                    const double myDelE = delE[e2];
                    const double invlen = intstate.invlen_GR[e2];
                    const double* dndegr_ptr = &(intstruct.dNdE_GR[e2 * n_energies]);
                    const double flux = fluxCosRho[electronflavor * fluxOffset + e2];
                    const double flux_invlen_dele = flux * invlen * myDelE;

                    const double dnde = dndegr_ptr[e1];
                    e1_contribution += flux_invlen_dele * dnde;
                }
            }

            gr_array[e1] = e1_contribution;
        }

    }


    DEVICEQUALIFIER
    void deriveFull(){
        double result[NFLV*NFLV];
        double tmp1[NFLV*NFLV];
        double myCurrentState[NFLV*NFLV];
        double tmp3[NFLV*NFLV];

        // derive every matrix

        for(int index_rho = 0; index_rho < n_rhos; index_rho++){

            for(int index_energy = threadIdx.x + blockDim.x * blockIdx.x; index_energy < n_energies; index_energy += blockDim.x * gridDim.x){

                #pragma unroll
                for(int i = 0; i < NFLV * NFLV; i++){
                    result[i] = 0;
                    tmp1[i] = 0;
                    tmp3[i] = 0;
                }

                // if neccessary, fetch current density matrix

                if(flags.useCoherentRhoTerms
                    || flags.useNonCoherentRhoTerms){

                    const double* statedataIn = getPitchedElement(y,
                                                    index_rho * NFLV * NFLV,
                                                    index_energy,
                                                    statesPitch);

                    #pragma unroll
                    for(int i = 0; i < NFLV * NFLV; i++)
                        myCurrentState[i] = statedataIn[i * statesOffset];
                }

                if(flags.useCoherentRhoTerms){

                    ops.HI(*this, tmp1, index_rho, index_energy);

                    // result = i*[currentstate , HI]

					sumath::iCommutator(result, myCurrentState, tmp1);

                }

                if(flags.useNonCoherentRhoTerms){

                    ops.GammaRho(*this, tmp1, index_rho, index_energy);

                    // tmp3 = {GammaRho , currentstate}
					sumath::anticommutator(tmp3, tmp1, myCurrentState);

                    // result -= {GammaRho , currentstate}
                    #pragma unroll
                    for(int i = 0; i < NFLV * NFLV; i++)
                        result[i] -= tmp3[i];

                }

                if(flags.useInteractionsRhoTerms){

                    ops.InteractionsRho(*this, tmp1, index_rho, index_energy);

                    // result += InteractionsRho
                    #pragma unroll
                    for(int i = 0; i < NFLV * NFLV; i++)
                        result[i] += tmp1[i];

                }

                //write result back to memory

                double* statedataOut = getPitchedElement(y_derived,
                                                index_rho * NFLV * NFLV,
                                                index_energy,
                                                statesPitch);

                #pragma unroll
                for(int i = 0; i < NFLV * NFLV; i++)
                    statedataOut[i * statesOffset] = result[i];
            }
        }

        barrier_path();
    }

    DEVICEQUALIFIER
    void derive_osc(){

        double result[NFLV*NFLV];
        double tmp1[NFLV*NFLV];
        double myCurrentState[NFLV*NFLV];

        for(int index_rho = 0; index_rho < n_rhos; index_rho++){

            for(int index_energy = threadIdx.x + blockDim.x * blockIdx.x;
                index_energy < n_energies;
                index_energy += blockDim.x * gridDim.x){

                #pragma unroll
                for(int i = 0; i < NFLV * NFLV; i++){
                    result[i] = 0;
                    tmp1[i] = 0;
                }

                const double* statedataIn = getPitchedElement(y,
                                                index_rho * NFLV * NFLV,
                                                index_energy,
                                                statesPitch);

                #pragma unroll
                for(int i = 0; i < NFLV * NFLV; i++)
                    myCurrentState[i] = statedataIn[i * statesOffset];

                ops.HI(*this, tmp1, index_rho, index_energy);

                    // result = i*[currentstate , HI]

				sumath::iCommutator(result, myCurrentState, tmp1);


                //write result back to memory

                double* statedataOut = getPitchedElement(y_derived,
                                                index_rho * NFLV * NFLV,
                                                index_energy,
                                                statesPitch);

                #pragma unroll
                for(int i = 0; i < NFLV * NFLV; i++)
                    statedataOut[i * statesOffset] = result[i];
            }
        }

        barrier_path();
    }

    DEVICEQUALIFIER
    void updateTracks(){
            if(threadIdx.x + blockDim.x * blockIdx.x == 0){
                track.setCurrentX(t);
                density = body.getDensity(track);
                electronFraction = body.getYe(track);
            }
            barrier_path();
    }

    DEVICEQUALIFIER
    void prederiveFull(double time, const double* y, double* y_derived){
        t = time;

        setDerivationPointers(y, y_derived);

        updateTracks();

        evolveProjectors(time);

        barrier_path();

        // update interactions, if necessary
        if(flags.canUseInteractions){

            updateInteractionStruct();

            if(flags.useInteractionsRhoTerms){

                calculateCurrentFlavorFluxes();

                barrier_path(); // need complete flux and interaction structure to calculate interaction arrays

                if(flags.useNCInteractions){

                    updateNCArrays();

                }

                if(flags.useTauRegeneration){

                    updateTauArraysPart1();

                    barrier_path(); // part 1 needs to be completed before part 2

                    updateTauArraysPart2();
                }


                if(flags.useGlashowResonance && neutrinoType != nusquids::NeutrinoType::neutrino){

                    updateGRArrays();

                }
            }

            barrier_path();
        }

        ops.addToPrederive(*this, time);

        barrier_path();
    }

    DEVICEQUALIFIER
    void prederive_osc(double time, const double* y, double* y_derived){
        t = time;

        setDerivationPointers(y, y_derived);

        updateTracks();

        evolveProjectors(time);

        barrier_path();

        ops.addToPrederive(*this, time);

        barrier_path();

    }

    DEVICEQUALIFIER
    void addToPrederive(double time){
        ops.addToPrederive(*this, time);
    }

    DEVICEQUALIFIER
    void H0(double out[], int index_rho, int index_energy) const{
        ops.H0(*this, out, index_rho, index_energy);
    }

    DEVICEQUALIFIER
    void HI(double out[],
            int index_rho, int index_energy) const{
        ops.HI(*this, out, index_rho, index_energy);
    }

    DEVICEQUALIFIER
    void GammaRho(double out[],
                    int index_rho, int index_energy) const{
        ops.GammaRho(*this, out, index_rho, index_energy);
    }

    DEVICEQUALIFIER
    void InteractionsRho(double out[],
                                int index_rho, int index_energy) const{
        ops.InteractionsRho(*this, out, index_rho, index_energy);
    }
};


#undef GETSET
#undef GETREFSET


/// \class PhysicsOps
/// \brief Defines the physical operators used for simulation
///
struct PhysicsOps{

    HOSTDEVICEQUALIFIER
    PhysicsOps(){}
    HOSTDEVICEQUALIFIER
    ~PhysicsOps(){}

    /// \brief Perform custom updates before a derivation step
    template<class Physics>
    DEVICEQUALIFIER
    void addToPrederive(Physics& base, double time) const{

    }

    /// \brief Calculate time-independent part of the Hamiltonian
    /// \param base The physics object to which this function is applied
    /// \param out Output array with length Physics::NFLV * Physics::NFLV
    /// \param index_rho 0, if neutrinotype != Both. Else, 0 = neutrino, 1 = antineutrino
    /// \param index_energy Index of energy bin.
    template<class Physics>
    DEVICEQUALIFIER
    void H0(const Physics& base, double out[], int index_rho, int index_energy) const{
        const double energy = base.get_energyList()[index_energy];

        #pragma unroll
	    for(int i = 0; i < Physics::NFLV * Physics::NFLV; ++i){
                out[i] = base.get_dm2()[i] * (0.5 / energy);
	    }
    }

    /// \brief Calculate time-dependent part of the Hamiltonian
    /// \param base The physics object to which this function is applied
    /// \param out Output array with length Physics::NFLV * Physics::NFLV
    /// \param index_rho 0, if neutrinotype != Both. Else, 0 = neutrino, 1 = antineutrino
    /// \param index_energy Index of energy bin.
    template<class Physics>
    DEVICEQUALIFIER
    void HI(const Physics& base, double out[],
            int index_rho, int index_energy) const{

        #pragma unroll
        for(int i = 0; i < Physics::NFLV * Physics::NFLV; ++i){
            out[i] = 0;
        }

        double CC = Const::HI_constants() * base.get_density() * base.get_electronFraction();
        double NC;

        if (base.get_electronFraction() < 1.0e-10){
            NC = Const::HI_constants() * base.get_density();
        }else {
            NC = CC*(-0.5*(1.0-base.get_electronFraction())/base.get_electronFraction());
        }

        // Antineutrino matter potential flips sign
        if((index_rho == 1 && base.get_neutrinoType() == nusquids::NeutrinoType::both)
            || base.get_neutrinoType() == nusquids::NeutrinoType::antineutrino){
            CC *= -1;
            NC *= -1;
        }

        const double CCNC = CC + NC;

        #pragma unroll
        for(int j = 0; j < 3; j++){
            double proj[Physics::NFLV * Physics::NFLV];

            const double* evoldata = getPitchedElement(base.get_evolB1proj(),
                                                        index_rho * Physics::NFLV * Physics::NFLV * Physics::NFLV + j * Physics::NFLV * Physics::NFLV,
                                                        index_energy,
                                                        base.get_evolB1pitch());

            #pragma unroll
            for(int i = 0; i < Physics::NFLV * Physics::NFLV; i++){
                proj[i] = evoldata[i * base.get_evoloffset()];
            }

            // calculate HI (out)
            #pragma unroll
            for(int i = 0; i < Physics::NFLV * Physics::NFLV; ++i){
                if(j == 0)
                    out[i] += CCNC * proj[i];
                else
                    out[i] += NC * proj[i];
            }
        }

        if(base.get_basis() == nusquids::Basis::mass){
            const double* h0data = getPitchedElement(base.get_H0_array(), 0, index_energy, base.get_h0pitch());
            #pragma unroll
            for(int i = 0; i < Physics::NFLV * Physics::NFLV; ++i){
                out[i] += h0data[i * base.get_h0offset()];
            }
        }
    }

    /// \brief Calculate absorbtion and attenuation
    /// \param base The physics object to which this function is applied
    /// \param out Output array with length Physics::NFLV * Physics::NFLV
    /// \param index_rho 0, if neutrinotype != Both. Else, 0 = neutrino, 1 = antineutrino
    /// \param index_energy Index of energy bin.
    template<class Physics>
    DEVICEQUALIFIER
    void GammaRho(const Physics& base, double out[],
                    int index_rho, int index_energy) const{

        const double* invlenint0_ptr = &(base.get_intstate().invlen_INT[index_rho * Physics::NFLV * base.get_n_energies() + 0 * base.get_n_energies()]);
        const double* invlenint1_ptr = &(base.get_intstate().invlen_INT[index_rho * Physics::NFLV * base.get_n_energies() + 1 * base.get_n_energies()]);
        const double* invlenint2_ptr = &(base.get_intstate().invlen_INT[index_rho * Physics::NFLV * base.get_n_energies() + 2 * base.get_n_energies()]);

        const double* evoldata0 = getPitchedElement(base.get_evolB1proj(), index_rho * Physics::NFLV * Physics::NFLV * Physics::NFLV + 0 * Physics::NFLV * Physics::NFLV,
                                                index_energy,
                                                base.get_evolB1pitch());
        const double* evoldata1 = getPitchedElement(base.get_evolB1proj(), index_rho * Physics::NFLV * Physics::NFLV * Physics::NFLV + 1 * Physics::NFLV * Physics::NFLV,
                                                index_energy,
                                                base.get_evolB1pitch());
        const double* evoldata2 = getPitchedElement(base.get_evolB1proj(), index_rho * Physics::NFLV * Physics::NFLV * Physics::NFLV + 2 * Physics::NFLV * Physics::NFLV,
                                                index_energy,
                                                base.get_evolB1pitch());

        double proj0[Physics::NFLV * Physics::NFLV];
        double proj1[Physics::NFLV * Physics::NFLV];
        double proj2[Physics::NFLV * Physics::NFLV];

        #pragma unroll
        for(int i = 0; i < Physics::NFLV * Physics::NFLV; i++){
            proj0[i] = evoldata0[i * base.get_evoloffset()];
            proj1[i] = evoldata1[i * base.get_evoloffset()];
            proj2[i] = evoldata2[i * base.get_evoloffset()];
        }

        // calculate gamma
        const double invlen0 = invlenint0_ptr[index_energy];
        const double invlen1 = invlenint1_ptr[index_energy];
        const double invlen2 = invlenint2_ptr[index_energy];

        #pragma unroll
        for(int i = 0; i < Physics::NFLV * Physics::NFLV; ++i){
            out[i] = (proj0[i] * invlen0 * 0.5 + proj1[i] * invlen1 * 0.5 + proj2[i] * invlen2 * 0.5) ;
        }
    }

    /// \brief Calculate neutrino interactions
    /// \param base The physics object to which this function is applied
    /// \param out Output array with length Physics::NFLV * Physics::NFLV    
    /// \param index_rho 0, if neutrinotype != Both. Else, 0 = neutrino, 1 = antineutrino
    /// \param index_energy Index of energy bin.
    template<class Physics>
    DEVICEQUALIFIER
    void InteractionsRho(const Physics& base, double out[],
                                int index_rho, int index_energy) const{

        const int evolB1pitchDoubles = base.get_evolB1pitch() / sizeof(double);

        const double* evolB1projCosRho = getPitchedElement(base.get_evolB1proj(), index_rho * Physics::NFLV * Physics::NFLV * Physics::NFLV,
                                                0,
                                                base.get_evolB1pitch());

        const double* evoldata0 = evolB1projCosRho + 0 * Physics::NFLV * Physics::NFLV * evolB1pitchDoubles + index_energy;
        const double* evoldata1 = evolB1projCosRho + 1 * Physics::NFLV * Physics::NFLV * evolB1pitchDoubles + index_energy;
        const double* evoldata2 = evolB1projCosRho + 2 * Physics::NFLV * Physics::NFLV * evolB1pitchDoubles + index_energy;

        double proj0[Physics::NFLV * Physics::NFLV];
        double proj1[Physics::NFLV * Physics::NFLV];
        double proj2[Physics::NFLV * Physics::NFLV];

        #pragma unroll
        for(int i = 0; i < Physics::NFLV * Physics::NFLV; i++){
            proj0[i] = evoldata0[i * base.get_evoloffset()];
            proj1[i] = evoldata1[i * base.get_evoloffset()];
            proj2[i] = evoldata2[i * base.get_evoloffset()];
        }

        double factor_e = 0;
        double factor_mu = 0;
        double factor_tau = 0;

        if(base.get_flags().useNCInteractions){
            const int ie = index_rho * Physics::NFLV * base.get_n_energies() + 0 * base.get_n_energies() + index_energy;
            const int im = index_rho * Physics::NFLV * base.get_n_energies() + 1 * base.get_n_energies() + index_energy;
            const int it = index_rho * Physics::NFLV * base.get_n_energies() + 2 * base.get_n_energies() + index_energy;

            const double factor_e_nc = base.get_nc_array()[ie];
            const double factor_mu_nc = base.get_nc_array()[im];
            const double factor_tau_nc = base.get_nc_array()[it];

            factor_e = factor_e_nc;
            factor_mu = factor_mu_nc;
            factor_tau = factor_tau_nc;
        }
        if(base.get_flags().useTauRegeneration){
            const double factor_e_tauregen = base.get_tau_lep_array()[index_rho * base.get_n_energies() + index_energy];
            const double factor_mu_tauregen = base.get_tau_lep_array()[index_rho * base.get_n_energies() + index_energy];
            const double factor_tau_tauregen = base.get_tau_hadlep_array()[index_rho * base.get_n_energies() + index_energy];

            factor_e += factor_e_tauregen;
            factor_mu += factor_mu_tauregen;
            factor_tau += factor_tau_tauregen;
        }
        if(base.get_flags().useGlashowResonance
            && ((index_rho == 1 && base.get_neutrinoType() == nusquids::NeutrinoType::both)
                || base.get_neutrinoType() == nusquids::NeutrinoType::antineutrino)){

            const double grfactor = base.get_gr_array()[index_energy];

            factor_e += grfactor;
            factor_mu += grfactor;
            factor_tau += grfactor;
        }

        #pragma unroll
        for(int i = 0; i < Physics::NFLV * Physics::NFLV; ++i){
            out[i] = factor_e * proj0[i] + factor_mu * proj1[i] + factor_tau * proj2[i];
        }
    }
};

}

#endif
