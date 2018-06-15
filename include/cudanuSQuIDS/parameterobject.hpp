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

#ifndef CUDANUSQUIDS_PARAMETEROBJECT_HPP
#define CUDANUSQUIDS_PARAMETEROBJECT_HPP

#include <nuSQuIDS/nuSQuIDS.h>
#include <nuSQuIDS/xsections.h>

#include <cudanuSQuIDS/interaction_structure.hpp>
#include <cudanuSQuIDS/types.hpp>

#include <vector>
#include <memory>
#include <map>
#include <mutex>

namespace cudanusquids{

    ///\class ParameterObject
    ///\brief Manages simulation parameters
    struct ParameterObject{

	private:
		//Wrapper to get access to some protected member functions of nuSQUIDS
		struct NusquidsWrapper : public nusquids::nuSQUIDS{

			NusquidsWrapper(nusquids::marray<double,1> energylist,
							unsigned int n_flvs,
							nusquids::NeutrinoType neutype,
							bool interactions,
							std::shared_ptr<nusquids::NeutrinoCrossSections> ncs = nullptr)
				: nusquids::nuSQUIDS(energylist, n_flvs, neutype, interactions, ncs){

			}

			NusquidsWrapper(NusquidsWrapper&& w)
				: nusquids::nuSQUIDS(reinterpret_cast<nusquids::nuSQUIDS&&>(w)){

			}

			void initializeProjectors(){
				std::lock_guard<std::mutex> g(mutex);
				iniProjectors();
			}

			squids::SU_vector getDM2() const{
				squids::SU_vector vec = squids::SU_vector(nsun);
				for(unsigned int i = 1; i < nsun; i++){
					vec += (b0_proj[i])*params.GetEnergyDifference(i);
				}
				return vec;
			}

			nusquids::Basis getBasis() const{
					return basis;
			}

			nusquids::NeutrinoType getNeutrinoType() const{
					return NT;
			}

			std::mutex mutex;
		};

		struct ExtraData{
            size_t size;
            std::vector<char> data;
        };

	public:
        /// \brief Constructor.
        /// @param numPaths Number of neutrino trajectories
        /// @param energylist List of neutrino energies [eV].
        /// @param n_flvs Number of neutrino flavors
        /// @param neutype: neutrino,antineutrino, or both (simultaneous solution).
        /// @param interactions If interactions can be used.
        /// @param ncs Cross section object. (optional)
        ParameterObject(int numPaths,
						nusquids::marray<double,1> energylist,
						int n_flvs,
						nusquids::NeutrinoType neutype,
						bool interactions,
						std::shared_ptr<nusquids::NeutrinoCrossSections> ncs = nullptr);

        /// \brief Set simulation basis. mass or interaction picture
        void Set_Basis(nusquids::Basis basis);
        /// \brief Get simulation basis.
		nusquids::Basis getBasis() const;

        /// \brief Get type of simulated neutrinos
		nusquids::NeutrinoType getNeutrinoType() const;

        /// \brief Set theta_(i+1)_(j+1)
		void Set_MixingAngle(unsigned int i, unsigned int j, double angle);
        /// \brief Set m_i - m_1
		void Set_SquareMassDifference(unsigned int i, double diff);
        /// \brief Set delta_cp_(i+1)_(j+1)
		void Set_CPPhase(unsigned int i, unsigned int j,double angle);

        /// \brief Set minimum integration step size
		void Set_h_min(double opt);
        /// \brief Set maximum integration step size
		void Set_h_max(double opt);
        /// \brief Set begin integration step size for adaptive stepsize integration
		void Set_h(double opt);
        /// \brief Set maximum relative integration error
		void Set_rel_error(double opt);
        /// \brief Set maximum absolute integration error
		void Set_abs_error(double opt);
        /// \brief Set number of steps for fixed stepsize integration
		void Set_NumSteps(unsigned int opt);

        /// \brief Get minimum integration step size
		double Get_h_min() const;
        /// \brief Get maximum integration step size
		double Get_h_max() const;
        /// \brief begin integration step size for adaptive stepsize integration
		double Get_h() const;
        /// \brief Set maximum relative integration error
		double Get_rel_error() const;
        /// \brief Set maximum absolute integration error
		double Get_abs_error() const;
        /// \brief Set number of steps for fixed stepsize integration
		unsigned int Get_NumSteps() const;

        /// \brief Get list of neutrino energies
		nusquids::marray<double,1> GetERange() const;
        /// \brief Get flavor projector. If rho == 1, get projector for anti-neutrino
		const squids::SU_vector& GetFlavorProj(unsigned int flv,unsigned int rho = 0) const;
        /// \brief Get mass projector. If rho == 1, get projector for anti-neutrino
		const squids::SU_vector& GetMassProj(unsigned int flv,unsigned int rho = 0) const;
        /// \brief Get object with squids constants
		const squids::Const& GetParams() const;
        /// \brief Get number of energy bins
		int GetNumE() const;
        /// \brief Get number of flavors
		int GetNumNeu() const;
        /// \brief Get number of neutrino types
		int GetNumRho() const;

        /// \brief Get number of paths
		int getNumPaths() const;

        /// \brief Check if interactions can be used
		bool Get_CanUseInteractions() const;

        /// \brief Enable / Disable neutrino oscillation
		void Set_IncludeOscillations(bool opt);
        /// \brief Check if neutrino oscillation is enabled
		bool Get_IncludeOscillations() const;

        /// \brief Enable / Disable non-coherent terms
		void Set_NonCoherentRhoTerms(bool opt);
        /// \brief Check if non-coherent terms are enabled
		bool Get_NonCoherentRhoTerms() const;

        /// \brief Enable / Disable interaction terms
		void Set_InteractionsRhoTerms(bool opt);
        /// \brief Check if interaction terms are enabled
		bool Get_InteractionsRhoTerms() const;

        /// \brief Enable / Disable neutral current interactions. Will only be calculated if Get_InteractionsRhoTerms() == true
		void Set_NCInteractions(bool opt);
        /// \brief Enable / Disable Tau regeneration. Will only be calculated if Get_InteractionsRhoTerms() == true
		void Set_TauRegeneration(bool opt);
        /// \brief Enable / Disable Glashow resonance. Will only be calculated if Get_InteractionsRhoTerms() == true
		void Set_GlashowResonance(bool opt);

        /// \brief Enable / Disable progress bar. A Progressbar should only be used with a single GPU.
        void Set_ProgressBar(bool opt);
        /// \brief Check if progress bar is enabled
        bool Get_ProgressBar() const;

        /// \brief Set device ids for GPUs that should be used for simulation
        void Set_DeviceIds(const std::vector<int>& ids);
        /// \brief Get device Ids
        const std::vector<int>& Get_DeviceIds() const;

        /// \brief Set solver type. Either Version1 or Version2.
        void Set_SolverType(cudanusquids::SolverType solverType_);
        /// \brief Get solver type.
        cudanusquids::SolverType Get_SolverType() const;

        /// \brief Set integration method. RK4 (4th order Runge-Kutta)
        void Set_StepperType(cudanusquids::ode::StepperType stepperType_);
        /// \brief Get integration method
        cudanusquids::ode::StepperType Get_StepperType() const;

        /// \brief Set initial neutrino flux from generator, specified in basis fluxbasis, either mass or flavor
        template<class Func>
    	void setInitialFlux(Func fluxGenerator, nusquids::Basis fluxbasis_){
    		initialFlux.resize(getNumPaths() * GetNumRho() * GetNumE() * GetNumNeu());

    		size_t k = 0;
    		for(int index_cosine = 0; index_cosine < getNumPaths(); index_cosine++){
    			for(int index_rho = 0; index_rho < GetNumRho(); index_rho++){
    				for(int index_energy = 0; index_energy < GetNumE(); index_energy++){
    					for(int flv = 0; flv < GetNumNeu(); flv++){
    						initialFlux[k++] = fluxGenerator(index_cosine, index_rho, index_energy, flv);
    					}
    				}
    			}
    		}

            fluxbasis = fluxbasis_;
    	}

        /// \brief Set initial neutrino flux from list, specified in basis fluxbasis, either mass or flavor.
        ///
        /// \details initialFlux is flat array of dimensions [number of paths][number of neutrino types][number of energies][number of flavors]
        void setInitialFlux(const std::vector<double>& initialFlux, nusquids::Basis fluxbasis);
        /// \brief Get initial neutrino flux list as it was set by setInitialFlux
        const std::vector<double>& Get_InitialFlux() const;
        /// \brief Get flux basis as it was set by setInitialFlux
        nusquids::Basis Get_FluxBasis() const;

        /// \brief Create a GPU array of size bytes which can be accessed by custom physics operations.
        /// The array contents are left uninitialized
        ///
        /// \details The additional GPU arrays are made available in member void** additionalData; of struct Physics
        /// additionalData[0] holds a pointer to the first registered GPU array,
        /// additionalData[1] holds a pointer to the second registered GPU array, and so on.
        /// The arrays are not exclusive to a specific path, but can be accessed by all neutrino paths.
        /// Thus, the arrays must be created large enough to fit the data for the maximum number of parallel simulated paths
        void registerAdditionalData(size_t size);
        /// \brief Create a GPU array of size bytes which can be accessed by custom physics operations
        /// size bytes are copied from data to the GPU array.
        ///
        /// \details The additional GPU arrays are made available in member void** additionalData; of struct Physics
        /// additionalData[0] holds a pointer to the first registered GPU array,
        /// additionalData[1] holds a pointer to the second registered GPU array, and so on.
        /// The arrays are not exclusive to a specific path, but can be accessed by all neutrino paths.
        /// Thus, the arrays must be created large enough to fit the data for the maximum number of parallel simulated paths
        void registerAdditionalData(size_t size, const char* data);
        /// \brief Delete every GPU array created by registerAdditionalData
        void clearAdditionalData();

	public:
		std::shared_ptr<cudanusquids::InteractionStructure> make_InteractionStructure();

        void initializeProjectors();
		void InitializeInteractions();

		std::shared_ptr<nusquids::nuSQUIDS::InteractionStructure> GetInteractionStructure();
		std::shared_ptr<const nusquids::nuSQUIDS::InteractionStructure> GetInteractionStructure() const;

		detail::Flags getFlags() const;

        squids::SU_vector getDM2() const;

		const std::vector<ExtraData>& Get_AdditionalDataCpu() const;
	private:
		NusquidsWrapper wrapper;

		int numPaths;

		detail::Flags flags;
        bool showProgress;
        std::vector<int> deviceIds;

        cudanusquids::SolverType solverType;
        cudanusquids::ode::StepperType stepperType;

        std::vector<double> initialFlux;
        nusquids::Basis fluxbasis;

        std::vector<ExtraData> additionalDataCPU;
    };

}

#endif
