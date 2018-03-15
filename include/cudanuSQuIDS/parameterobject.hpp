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

#ifndef CUDANUSQUIDS_NUSQUIDSCONVERSION_HPP
#define CUDANUSQUIDS_NUSQUIDSCONVERSION_HPP

#include <nuSQuIDS/nuSQuIDS.h>
#include <nuSQuIDS/xsections.h>

#include <cudanuSQuIDS/interaction_structure.hpp>
#include <cudanuSQuIDS/types.hpp>

#include <vector>
#include <memory>
#include <map>
#include <mutex>

namespace cudanusquids{

	//derived from nuSQUIDS to reuse its Getters and Setters
    struct ParameterObject : public nusquids::nuSQUIDS{
        nusquids::marray<double,1> pathParameterList;
		detail::Flags flags;
        bool showProgress;
        std::vector<int> deviceIds;

        cudanusquids::SolverType solverType;
        cudanusquids::ode::StepperType stepperType;

        std::vector<double> initialFlux;
        nusquids::Basis fluxbasis;

	std::mutex mutex;

        struct ExtraData{
            size_t size;
            std::vector<char> data;
        };

        std::vector<ExtraData> additionalDataCPU;

        ParameterObject(nusquids::marray<double,1> pathParameterList, nusquids::marray<double,1> energylist, unsigned int n_flvs,
            nusquids::NeutrinoType neutype, bool interactions,
            std::shared_ptr<nusquids::NeutrinoCrossSections> ncs = nullptr)
                : nusquids::nuSQUIDS(energylist, n_flvs, neutype, interactions, ncs),
                    pathParameterList(pathParameterList), showProgress(false){

				flags.useNCInteractions = interactions;
            }

        std::shared_ptr<cudanusquids::InteractionStructure> make_InteractionStructure();

        void initializeProjectors(){
	    std::lock_guard<std::mutex> g(mutex);
            iniProjectors();
        }

        squids::SU_vector getDM2(){
            squids::SU_vector vec = squids::SU_vector(nsun);
            for(unsigned int i = 1; i < nsun; i++){
                vec += (b0_proj[i])*params.GetEnergyDifference(i);
            }
            return vec;
        }

        nusquids::marray<double,1> getPathParameterList() const{
				return pathParameterList;
		}

		size_t getNumPaths() const{
			return pathParameterList.size();
		}

		nusquids::Basis getBasis() const{
				return basis;
		}

		nusquids::NeutrinoType getNeutrinoType() const{
				return NT;
		}

        //shadowing nuSQUIDS
		void Set_IncludeOscillations(bool opt){
			flags.useCoherentRhoTerms = opt;
		}
		//shadowing nuSQUIDS
		void Set_NonCoherentRhoTerms(bool opt){
			flags.useNonCoherentRhoTerms = opt;
		}

		void Set_NCInteractions(bool opt){
			flags.useNCInteractions = opt;
		}
		//shadowing nuSQUIDS
		void Set_TauRegeneration(bool opt){
			if(opt && getNeutrinoType() != nusquids::NeutrinoType::both)
				throw std::runtime_error("ParameterObject: Tau Regeneration is only available if neutrino type is nusquids::NeutrinoType::both");
			flags.useTauRegeneration = opt;
		}
		//shadowing nuSQUIDS
		void Set_GlashowResonance(bool opt){
			flags.useGlashowResonance = opt;
		}

		detail::Flags getFlags() const{
			return flags;
		}
        //shadowing nuSQUIDS
        void Set_ProgressBar(bool opt){
            showProgress = opt;
        }

        bool Get_ProgressBar() const{
            return showProgress;
        }

        void Set_DeviceIds(const std::vector<int>& ids){
            deviceIds = ids;
        };

        const std::vector<int>& Get_DeviceIds() const{
            return deviceIds;
        };

        void Set_SolverType(cudanusquids::SolverType solverType_){
            solverType = solverType_;
        }

        cudanusquids::SolverType Get_SolverType() const{
            return solverType;
        }

        void Set_StepperType(cudanusquids::ode::StepperType stepperType_){
            stepperType = stepperType_;
        }

        cudanusquids::ode::StepperType Get_StepperType() const{
            return stepperType;
        }

        template<class Func>
    	void setInitialFlux(Func fluxGenerator, nusquids::Basis fluxbasis_){
    		initialFlux.resize(getNumPaths() * GetNumRho() * GetNumE() * GetNumNeu());

    		size_t k = 0;
    		for(size_t index_cosine = 0; index_cosine < getNumPaths(); index_cosine++){
    			for(size_t index_rho = 0; index_rho < GetNumRho(); index_rho++){
    				for(size_t index_energy = 0; index_energy < GetNumE(); index_energy++){
    					for(size_t flv = 0; flv < GetNumNeu(); flv++){
    						initialFlux[k++] = fluxGenerator(index_cosine, index_rho, index_energy, flv);
    					}
    				}
    			}
    		}

            fluxbasis = fluxbasis_;
    	}

        //initialFlux_ is flat array of dimensions [number of paths][number of rhos][number of energies][number of flavors]
        void setInitialFlux(const std::vector<double>& initialFlux_, nusquids::Basis fluxbasis_){
            if(getNumPaths()  * GetNumE() * GetNumRho() * GetNumNeu() != initialFlux_.size())
                throw std::runtime_error("ParameterObject::setInitialFlux: ParameterObject was not created for this number of states");

            initialFlux = initialFlux_;

            fluxbasis = fluxbasis_;
        }

        const std::vector<double>& Get_InitialFlux() const{
            return initialFlux;
        }

        nusquids::Basis Get_FluxBasis() const{
            return fluxbasis;
        }

        void registerAdditionalData(size_t size){
            additionalDataCPU.push_back({size, {}});
        }

        void registerAdditionalData(size_t size, const char* data){
            additionalDataCPU.push_back({size, std::vector<char>(data, data + size)});
        }

        void clearAdditionalData(){
            additionalDataCPU.clear();
        }

        const std::vector<ExtraData>& Get_AdditionalDataCpu() const{
            return additionalDataCPU;
        }


    };

}

#endif
