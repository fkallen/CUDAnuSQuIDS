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

#include <cudanuSQuIDS/parameterobject.hpp>

#include <nuSQuIDS/xsections.h>
#include <nuSQuIDS/nuSQuIDS.h>
#include <nuSQuIDS/marray.h>

#include <cudanuSQuIDS/interaction_structure.hpp>

#include <memory>

namespace cudanusquids{
	
	ParameterObject::ParameterObject(int numPaths,
									nusquids::marray<double,1> energylist, 
									int n_flvs,
									nusquids::NeutrinoType neutype, 
									bool interactions,
									std::shared_ptr<nusquids::NeutrinoCrossSections> ncs)
			: 	wrapper(NusquidsWrapper(energylist, n_flvs, neutype, interactions, ncs)),
				numPaths(numPaths),
				showProgress(false){

		flags.canUseInteractions = interactions;
	}

	void ParameterObject::initializeProjectors(){
		wrapper.initializeProjectors();
	}
	
	void ParameterObject::InitializeInteractions(){
		wrapper.InitializeInteractions();
	}
	
	std::shared_ptr<nusquids::nuSQUIDS::InteractionStructure> ParameterObject::GetInteractionStructure(){
		return wrapper.GetInteractionStructure();
	}
	
	std::shared_ptr<const nusquids::nuSQUIDS::InteractionStructure> ParameterObject::GetInteractionStructure() const{
		return wrapper.GetInteractionStructure();
	}		

	squids::SU_vector ParameterObject::getDM2() const{
		return wrapper.getDM2();
	}
	
	void ParameterObject::Set_Basis(nusquids::Basis basis){
			wrapper.Set_Basis(basis);
	}
	
	nusquids::Basis ParameterObject::getBasis() const{
		return wrapper.getBasis();
	}

	nusquids::NeutrinoType ParameterObject::getNeutrinoType() const{
		return wrapper.getNeutrinoType();
	}
	
	void ParameterObject::Set_MixingAngle(unsigned int i, unsigned int j, double angle){
		wrapper.Set_MixingAngle(i,j,angle);
	}
	
	void ParameterObject::Set_SquareMassDifference(unsigned int i, double diff){
		wrapper.Set_SquareMassDifference(i, diff);
	}
	
	void ParameterObject::Set_CPPhase(unsigned int i, unsigned int j,double angle){
		wrapper.Set_CPPhase(i,j,angle);
	}
	
	void ParameterObject::Set_h_min(double opt){
		wrapper.Set_h_min(opt);
	}

	void ParameterObject::Set_h_max(double opt){
		wrapper.Set_h_max(opt);
	}

	void ParameterObject::Set_h(double opt){
		wrapper.Set_h(opt);
	}

	void ParameterObject::Set_rel_error(double opt){
		wrapper.Set_rel_error(opt);
	}

	void ParameterObject::Set_abs_error(double opt){
		wrapper.Set_abs_error(opt);
	}

	void ParameterObject::Set_NumSteps(unsigned int opt){
		wrapper.Set_NumSteps(opt);
	}

	double ParameterObject::Get_h_min() const{
		return wrapper.Get_h_min();
	}

	double ParameterObject::Get_h_max() const{
		return wrapper.Get_h_max();
	}

	double ParameterObject::Get_h() const{
		return wrapper.Get_h();
	}

	double ParameterObject::Get_rel_error() const{
		return wrapper.Get_rel_error();
	}

	double ParameterObject::Get_abs_error() const{
		return wrapper.Get_abs_error();
	}

	unsigned int ParameterObject::Get_NumSteps() const{
		return wrapper.Get_NumSteps();
	}
	
	nusquids::marray<double,1> ParameterObject::GetERange() const{
		return wrapper.GetERange();
	}
	
	const squids::SU_vector& ParameterObject::GetFlavorProj(unsigned int flv,unsigned int rho) const{
		return wrapper.GetFlavorProj(flv,rho);
	}

	const squids::SU_vector& ParameterObject::GetMassProj(unsigned int flv,unsigned int rho) const{
		return wrapper.GetMassProj(flv,rho);
	}
	
	const squids::Const& ParameterObject::GetParams() const{
		return wrapper.GetParams();
	}
	
	int ParameterObject::GetNumE() const{
		return wrapper.GetNumE();
	}

	int ParameterObject::GetNumNeu() const{
		return wrapper.GetNumNeu();
	}

	int ParameterObject::GetNumRho() const{
		return wrapper.GetNumRho();
	} 

	int ParameterObject::getNumPaths() const{
		return numPaths;
	}
	
	bool ParameterObject::Get_CanUseInteractions() const{
		return flags.canUseInteractions;
	}
	
	bool ParameterObject::Get_NonCoherentRhoTerms() const{
		return flags.useNonCoherentRhoTerms;
	}
	
	bool ParameterObject::Get_InteractionsRhoTerms() const{
		return flags.useInteractionsRhoTerms;
	}
	
	void ParameterObject::Set_IncludeOscillations(bool opt){
		flags.useCoherentRhoTerms = opt;
	}
	
	bool ParameterObject::Get_IncludeOscillations() const{
		return flags.useCoherentRhoTerms;
	}

	void ParameterObject::Set_NonCoherentRhoTerms(bool opt){
		flags.useNonCoherentRhoTerms = opt;
	}
	
	void ParameterObject::Set_InteractionsRhoTerms(bool opt){
		flags.useInteractionsRhoTerms = opt;
	}

	void ParameterObject::Set_NCInteractions(bool opt){
		flags.useNCInteractions = opt;
	}

	void ParameterObject::Set_TauRegeneration(bool opt){
		if(opt && getNeutrinoType() != nusquids::NeutrinoType::both)
			throw std::runtime_error("ParameterObject: Tau Regeneration is only available if neutrino type is nusquids::NeutrinoType::both");
		flags.useTauRegeneration = opt;
	}

	void ParameterObject::Set_GlashowResonance(bool opt){
		flags.useGlashowResonance = opt;
	}

	detail::Flags ParameterObject::getFlags() const{
		return flags;
	}

	void ParameterObject::Set_ProgressBar(bool opt){
		showProgress = opt;
	}

	bool ParameterObject::Get_ProgressBar() const{
		return showProgress;
	}

	void ParameterObject::Set_DeviceIds(const std::vector<int>& ids){
		deviceIds = ids;
	}

	const std::vector<int>& ParameterObject::Get_DeviceIds() const{
		return deviceIds;
	}

	void ParameterObject::Set_SolverType(cudanusquids::SolverType solverType_){
		solverType = solverType_;
	}

	cudanusquids::SolverType ParameterObject::Get_SolverType() const{
		return solverType;
	}

	void ParameterObject::Set_StepperType(cudanusquids::ode::StepperType stepperType_){
		stepperType = stepperType_;
	}

	cudanusquids::ode::StepperType ParameterObject::Get_StepperType() const{
		return stepperType;
	}

	//initialFlux_ is flat array of dimensions [number of paths][number of rhos][number of energies][number of flavors]
	void ParameterObject::setInitialFlux(const std::vector<double>& initialFlux_, nusquids::Basis fluxbasis_){
		if(size_t(getNumPaths()  * GetNumE() * GetNumRho() * GetNumNeu()) != initialFlux_.size())
			throw std::runtime_error("ParameterObject::setInitialFlux: ParameterObject was not created for this number of states");

		initialFlux = initialFlux_;

		fluxbasis = fluxbasis_;
	}

	const std::vector<double>& ParameterObject::Get_InitialFlux() const{
		return initialFlux;
	}

	nusquids::Basis ParameterObject::Get_FluxBasis() const{
		return fluxbasis;
	}

	void ParameterObject::registerAdditionalData(size_t size){
		additionalDataCPU.push_back({size, {}});
	}

	void ParameterObject::registerAdditionalData(size_t size, const char* data){
		additionalDataCPU.push_back({size, std::vector<char>(data, data + size)});
	}

	void ParameterObject::clearAdditionalData(){
		additionalDataCPU.clear();
	}

	const std::vector<ParameterObject::ExtraData>& ParameterObject::Get_AdditionalDataCpu() const{
		return additionalDataCPU;
	}	

    std::shared_ptr<cudanusquids::InteractionStructure> ParameterObject::make_InteractionStructure(){

        size_t n_energies_ = GetNumE();
        size_t n_rhos_ = GetNumRho();

        InitializeInteractions();
        auto nusint = GetInteractionStructure();

        std::shared_ptr<cudanusquids::InteractionStructure> intstruct = std::make_shared<cudanusquids::InteractionStructure>();

        intstruct->n_rhos = n_rhos_;
        intstruct->n_flvs = GetNumNeu();
        intstruct->n_energies = n_energies_;

        intstruct->s1 = intstruct->n_rhos * intstruct->n_flvs * intstruct->n_energies * intstruct->n_energies;
        intstruct->s2 = intstruct->n_energies * intstruct->n_energies;
        intstruct->s3 = intstruct->n_rhos * intstruct->n_flvs * intstruct->n_energies;
        intstruct->s4 = intstruct->n_energies;

        intstruct->dNdE_CC.resize(intstruct->s1, 0.0);
        intstruct->dNdE_NC.resize(intstruct->s1, 0.0);
        intstruct->dNdE_GR.resize(intstruct->s2, 0.0);
        intstruct->sigma_CC.resize(intstruct->s3, 0.0);
        intstruct->sigma_NC.resize(intstruct->s3, 0.0);
        intstruct->sigma_GR.resize(intstruct->s4, 0.0);
        intstruct->dNdE_tau_all.resize(intstruct->s2, 0.0);
        intstruct->dNdE_tau_lep.resize(intstruct->s2, 0.0);

        //copy marray data to contiguous memory for simple gpu transfer

        //copy sigma_CC, sigma_NC
        for(size_t neutype = 0; neutype < intstruct->n_rhos; neutype++){
            for(size_t flv = 0; flv < intstruct->n_flvs; flv++){
                for(size_t e1 = 0; e1 < intstruct->n_energies; e1++){
                    const auto index = neutype * intstruct->n_flvs * intstruct->n_energies + flv * intstruct->n_energies + e1;
                    intstruct->sigma_CC[index] = nusint->sigma_CC[neutype][flv][e1];
                    intstruct->sigma_NC[index] = nusint->sigma_NC[neutype][flv][e1];
                }
            }
        }

        // copy sigma_GR
        for(size_t e1 = 0; e1 < intstruct->n_energies; e1++){
            intstruct->sigma_GR[e1] = nusint->sigma_GR[e1];
        }

        //copy dNdE_tau_all, dNdE_tau_lep
        for(size_t e1 = 0; e1 < intstruct->n_energies; e1++){
            for(size_t e2 = 0; e2 < e1; e2++){
                intstruct->dNdE_tau_all[e1 * intstruct->n_energies + e2] = nusint->dNdE_tau_all[e1][e2];
                intstruct->dNdE_tau_lep[e1 * intstruct->n_energies + e2] = nusint->dNdE_tau_lep[e1][e2];
            }
        }

        //copy  dNdE_CC, dNdE_NC,
        for(size_t neutype = 0; neutype < intstruct->n_rhos; neutype++){
            for(size_t flv = 0; flv < intstruct->n_flvs; flv++){
                for(size_t e1 = 0; e1 < intstruct->n_energies; e1++){
                    for(size_t e2 = 0; e2 < e1; e2++){
                        const auto index = neutype * intstruct->n_flvs * intstruct->n_energies * intstruct->n_energies
                                    + flv * intstruct->n_energies * intstruct->n_energies
                                    + e1 * intstruct->n_energies
                                    + e2;

                        intstruct->dNdE_NC[index] = nusint->dNdE_NC[neutype][flv][e1][e2];
                        intstruct->dNdE_CC[index] = nusint->dNdE_CC[neutype][flv][e1][e2];
                    }
                }
            }
        }

        //copy dNdE_GR
        for(size_t e1 = 0; e1 < intstruct->n_energies; e1++){
            for(size_t e2 = 0; e2 < e1; e2++){
                intstruct->dNdE_GR[e1 * intstruct->n_energies + e2] = nusint->dNdE_GR[e1][e2];
            }
        }

        return intstruct;
    }


}
