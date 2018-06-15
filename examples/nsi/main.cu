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


/*
    This example shows:
    How to set neutrino cross section information,
    How to use custom physic operators,
    How to make additional GPU memory available for custom physic operators
*/

#include "../path.h"
#include "customphysics.cuh"

#include <cudanuSQuIDS/cudanusquids.cuh>
#include <cudanuSQuIDS/cudahelpers.cuh>
#include <cudanuSQuIDS/hpc_helpers.hpp>

#include <nuSQuIDS/tools.h>
#include <nuSQuIDS/xsections.h>
#include <SQuIDS/SUNalg.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>


template<unsigned int NFLVS>
void run(std::shared_ptr<cudanusquids::ParameterObject>& params, const nusquids::marray<double,1>& cosineList, int batchsizeLimit){

    using Bodytype = cudanusquids::EarthAtm; // choose Earth with atmosphere as body
    using Ops = CustomPhysicsOps;

    // At most BatchsizeLimit paths will be evolved in parallel per GPU. This can control resource usage such as GPU memory consumption.
    cudanusquids::CudaNusquids<NFLVS, Bodytype, Ops> cudanus(params, batchsizeLimit);

    std::vector<Bodytype> gpuBodies(params->Get_DeviceIds().size());
    std::vector<typename Bodytype::Track> tracks;
	tracks.reserve(params->getNumPaths());

    //it is required to create a separate Body for each GPU
    for(size_t i = 0; i < params->Get_DeviceIds().size(); i++){
        /*
            Expects file to be a table with at least 3 columns.
            First column contains radii, second column contains densities, third column contains hydrogenfractions.
            Like nuSQuIDS'/data/astro/EARTH_MODEL_PREM.dat
        */
        gpuBodies[i] = cudanusquids::EarthAtm::make_body_gpu(params->Get_DeviceIds()[i], EXAMPLE_DATA_PATH + "/astro/EARTH_MODEL_PREM.dat"); //this file is identical to the file shipped with nuSQuIDS
        cudanus.setBody(gpuBodies[i], params->Get_DeviceIds()[i]);
    }

    //create tracks with cosine
	for(int i = 0; i < params->getNumPaths(); i++){
		tracks.emplace_back(cosineList[i]);
	}

	cudanus.setTracks(tracks);

	/*
		Create additional data for use with custom physics
	*/

	double epsilon_mutau = 2e-2;

	gsl_matrix_complex * M = gsl_matrix_complex_calloc(3,3);
    gsl_complex c {{ epsilon_mutau , 0.0 }};
    gsl_matrix_complex_set(M,2,1,c);
    gsl_matrix_complex_set(M,1,2,gsl_complex_conjugate(c));
    squids::SU_vector NSI = squids::SU_vector(M);
    NSI.RotateToB1(params->GetParams());
    gsl_matrix_complex_free(M);

	std::vector<double> nsicomponents = NSI.GetComponents();

	//creates a gpu buffer of nsicomponents.size() doubles and copies contents of nsicomponents
	params->registerAdditionalData(sizeof(double) * nsicomponents.size(), (const char*)nsicomponents.data());

	// create buffer to store evolved NSI, asumming same hamiltonian for neutrinos/antineutrinos.
    // The buffer is left uninitialized.
    // since additionalData is shared between all paths, we need to allocate enough
    // memory for the maximum number of path which can be calculated simultaneously.
	params->registerAdditionalData(sizeof(double) * nsicomponents.size() * params->GetNumE()
                                    * std::min(params->getNumPaths(), batchsizeLimit));

	//since params changed after CudaNusquids construction, it is required to notify about certain changes
	cudanus.additionalDataChanged();

	/* other changes which require notification:
	cudanus.mixingParametersChanged(); Set_MixingAngle, etc.
	cudanus.simulationFlagsChanged();  Set_IncludeOscillations, etc.
    cudanus.initialFluxChanged();
	*/


    TIMERSTARTCPU(evolve);

    cudanus.evolve();

    TIMERSTOPCPU(evolve);

    // check that every path was solved successfully and print some stats;
    for(int i = 0; i < params->getNumPaths(); i++){
		/*
		   members of RKstats:
            unsigned int steps; // number of required steps
            unsigned int repeats; // number of repeated steps
            Status status; // success or failure
		 */
        const cudanusquids::ode::RKstats& stats = cudanus.getRKstats(i); //get Runge Kutta stats of path i

        //check if successfully evolved, if not print message
        if(stats.status != cudanusquids::ode::Status::success)
            std::cout << "cosine " << cosineList[i] << " failed after " << stats.steps << " steps." << std::endl;
    }

    TIMERSTARTCPU(fileoutput);

    std::ofstream out("out.txt");
    out << params->getNumPaths() << " " << params->GetNumE() << " " << params->GetNumRho() << " " << params->GetNumNeu() << '\n';
    for(int flv = 0; flv < params->GetNumNeu(); flv++){
        out << "Flv " << flv << '\n';
        for(int i = 0; i <params->getNumPaths(); i++){
            out << "cos(th) = " << cosineList[i] << "\n";
            for(int j = 0; j < params->GetNumE(); j++){
                for(int k = 0; k < params->GetNumRho(); k++){
					// the return value is invalid if evolution was not successful.
                    const double val = cudanus.EvalFlavorAtNode(flv, i, k, j); CUERR;
                    out << std::setprecision(20) << val << " ";
                }
            }
            out << '\n';
        }
        out << '\n';
    }

    TIMERSTOPCPU(fileoutput);

    //free gpu bodies
    for(auto& body : gpuBodies){
    	Bodytype::destroy_body_gpu(body);
    }
}

int nsi(int argc, char** argv){
    using Const = cudanusquids::Const;

	double Emin=1.e2 * Const::TeV();
	double Emax=1.e4 * Const::TeV();
	int n_energies = 200;
	double czmin = -1;
	double czmax = 0;
	int n_cosines = 200;
	nusquids::NeutrinoType neutrinoType = nusquids::NeutrinoType::both;
	int n_neutrinos = 3;
	unsigned int nSteps = 0; // nSteps 0 means adaptive step size mode

	std::vector<int> usableDeviceIds{0};

	if(argc > 1){
		usableDeviceIds.clear();
		for(int i = 1; i < argc; i++)
			usableDeviceIds.push_back(std::atoi(argv[i]));
	}

	auto energyList = nusquids::logspace(Emin, Emax, n_energies);
	auto cosineList = nusquids::linspace(czmin, czmax, n_cosines);

	assert(usableDeviceIds.size() > 0);

	bool useOscillation = true;
	bool useNonCoherentRhoTerms = true;
	bool useNCInteractions = true;
	bool useTauRegeneration = true;
	bool useGlashowResonance = true;

	bool useInteractionsRhoTerms = useNCInteractions
				|| useTauRegeneration || useGlashowResonance;

	bool anyInteractions = useNonCoherentRhoTerms || useInteractionsRhoTerms;

	auto typeToName = [](nusquids::NeutrinoType t){
			switch(t){
				case nusquids::NeutrinoType::both: return "Neutrino+Antrineutrino";
				case nusquids::NeutrinoType::neutrino: return "Neutrino";
				case nusquids::NeutrinoType::antineutrino: return "Antrineutrino";
				default: return "";
			}
	};

	std::cout << "cz min : " << czmin << '\n'
		<< "cz max : " << czmax << '\n'
		<< "cosine bins : " << n_cosines << '\n'
		<< "E min : " << Emin / Const::TeV()<< " TeV" << '\n'
		<< "E max : " << Emax / Const::TeV()<< " TeV" << '\n'
		<< "energy bins : " << n_energies << '\n'
		<< "neutrino type : " << typeToName(neutrinoType) << '\n'
		<< "flavors : " <<  n_neutrinos << '\n';


	std::cout << "useOscillation : " << useOscillation
		<< "\nuseNonCoherentRhoTerms : " << useNonCoherentRhoTerms
		<< "\nuseNCInteractions : " << useNCInteractions
		<< "\nuseTauRegeneration : " << useTauRegeneration
		<< "\nuseGlashowResonance : " << useGlashowResonance << '\n';

	std::cout << "usable device ids : [";
	for( const auto& id : usableDeviceIds)
		std::cout << ' ' << id;
	std::cout << " ]" << std::endl;

    // use nusquids::NeutrinoDISCrossSectionsFromTables to gather neutrino cross-section information
    // this is the default and does not need to be explicitly specified
	std::shared_ptr<nusquids::NeutrinoCrossSections> crossSections
            = std::make_shared<nusquids::NeutrinoDISCrossSectionsFromTables>();
	std::shared_ptr<cudanusquids::ParameterObject> params
        = std::make_shared<cudanusquids::ParameterObject>(cosineList.size(), energyList,
                                                        n_neutrinos, neutrinoType,
                                                        anyInteractions, crossSections);

	params->Set_Basis(nusquids::Basis::interaction);

	//enable / disable parts of the simulated physics
	params->Set_IncludeOscillations(useOscillation);
	params->Set_NonCoherentRhoTerms(useNonCoherentRhoTerms);
	params->Set_InteractionsRhoTerms(useInteractionsRhoTerms);
	params->Set_NCInteractions(useNCInteractions);
	params->Set_TauRegeneration(useTauRegeneration);
	params->Set_GlashowResonance(useGlashowResonance);

	//set neutrino parameters
	params->Set_MixingAngle(0,1,0.563942);
	params->Set_MixingAngle(0,2,0.154085);
	params->Set_MixingAngle(1,2,0.785398);
	params->Set_SquareMassDifference(1,7.65e-05);
	params->Set_SquareMassDifference(2,0.00247);
	params->Set_CPPhase(0,2,0);

	if(n_neutrinos > 3){
		params->Set_SquareMassDifference(3,1.);
		params->Set_MixingAngle(1,3,0.5);
	}

	// Runge-Kutta settings
    params->Set_NumSteps(nSteps);
    params->Set_rel_error(1.e-6);
    params->Set_abs_error(1.e-6);
	//Type of Runge-Kutta method. Currently, only RK4 is available
    params->Set_StepperType(cudanusquids::ode::StepperType::RK4);
	//In Version1 the Runge-Kutta algorithm is executed on the CPU, but works on GPU data
	//In Version2 the Runge-Kutta algorithm is executed completely on the GPU
    params->Set_SolverType(cudanusquids::SolverType::Version1);

    params->Set_DeviceIds(usableDeviceIds);
    params->Set_ProgressBar(usableDeviceIds.size() == 1);

    auto fluxfunc = [&](int iPath, int irho, int ien, int iflv){
        if(iflv == 1){ // muon only flux on each path
            return 1.0;
        }else
            return 0.0;
    };

    params->setInitialFlux(fluxfunc, nusquids::Basis::flavor);

    switch(n_neutrinos){
        case 3:{
            run<3>(params, cosineList, 2000);
            break;}
        case 4:{
            run<4>(params, cosineList, 2000);
            break;}
		default: printf("error\n");
    }

	return 0;
}


int main(int argc, char** argv){
	int r = nsi(argc, argv);
	cudaDeviceReset(); CUERR;

    return r;
}
