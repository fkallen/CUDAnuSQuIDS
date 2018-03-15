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
    This example program shows the basic usage of cudanusquids.
*/

#include "../path.h"

#include <cudanuSQuIDS/cudanusquids.cuh>
#include <cudanuSQuIDS/hpc_helpers.hpp>

#include <nuSQuIDS/tools.h>
#include <nuSQuIDS/xsections.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

template<unsigned int NFLVS, unsigned int BATCH_SIZE_LIMIT>
void run(std::shared_ptr<cudanusquids::ParameterObject>& params){

    using Bodytype = cudanusquids::EarthAtm; // choose Earth with atmosphere as body

    // At most BATCH_SIZE_LIMIT paths will be evolved in parallel per GPU. This can control resource usage such as GPU memory consumption.
    cudanusquids::CudaNusquids<NFLVS, Bodytype, BATCH_SIZE_LIMIT> cudanus(params);

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
	for(size_t i = 0; i < params->getNumPaths(); i++){
		tracks.emplace_back(params->getPathParameterList()[i]);
	}

	cudanus.setTracks(tracks);

    TIMERSTARTCPU(evolve);

    cudanus.evolve();

    TIMERSTOPCPU(evolve);

    // check that every path was solved successfully and print some stats;
    for(size_t i = 0; i < params->getNumPaths(); i++){
		/*
		   members of RKstats:
            unsigned int steps; // number of required steps
            unsigned int repeats; // number of repeated steps
            Status status; // success or failure
		 */
        const cudanusquids::ode::RKstats& stats = cudanus.getRKstats(i); //get Runge Kutta stats of path i

        //check if successfully evolved, if not print message
        if(stats.status != cudanusquids::ode::Status::success)
            std::cout << "cosine " << params->getPathParameterList()[i] << " failed after " << stats.steps << " steps." << std::endl;
    }

    TIMERSTARTCPU(fileoutput);

    std::ofstream out("out.txt");
    out << params->getNumPaths() << " " << params->GetNumE() << " " << params->GetNumRho() << " " << params->GetNumNeu() << '\n';
    for(size_t flv = 0; flv < params->GetNumNeu(); flv++){
        out << "Flv " << flv << '\n';
        for(size_t i = 0; i <params->getNumPaths(); i++){
            out << "cos(th) = " << params->getPathParameterList()[i] << "\n";
            for(size_t j = 0; j < params->GetNumE(); j++){
                for(size_t k = 0; k < params->GetNumRho(); k++){
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

int atmospheric(int argc, char** argv){
    constexpr unsigned int BATCH_SIZE_LIMIT=2000;

    using Const = cudanusquids::Const;

	double Emin=1.e0 * Const::GeV();
	double Emax=1.e2 * Const::GeV();
	unsigned int n_energies = 200;
	double czmin = -1;
	double czmax = 0;
	unsigned int n_cosines = 200;
	nusquids::NeutrinoType neutrinoType = nusquids::NeutrinoType::both;
	unsigned int n_neutrinos = 3;
	unsigned int nSteps = 2000; // nSteps=0 means adaptive step size mode

	std::vector<int> usableDeviceIds{0}; //use device with id 0

	if(argc > 1){
		usableDeviceIds.clear();
		for(int i = 1; i < argc; i++)
			usableDeviceIds.push_back(std::atoi(argv[i]));
	}

	auto energyList = nusquids::logspace(Emin, Emax, n_energies);
	auto cosineList = nusquids::linspace(czmin, czmax, n_cosines);

	assert(usableDeviceIds.size() > 0);

	bool useOscillation = true;
	bool useNonCoherentRhoTerms = false;
	bool useNCInteractions = false;
	bool useTauRegeneration = false;
	bool useGlashowResonance = false;

	bool anyInteractions = useNonCoherentRhoTerms || useNCInteractions
				|| useTauRegeneration || useGlashowResonance;

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
		<< "E min : " << Emin / Const::GeV()<< " GeV" << '\n'
		<< "E max : " << Emax / Const::GeV()<< " GeV" << '\n'
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

	std::shared_ptr<cudanusquids::ParameterObject> params
        = std::make_shared<cudanusquids::ParameterObject>(cosineList, energyList,
                                                        n_neutrinos, neutrinoType,
                                                        anyInteractions);

	params->Set_Basis(nusquids::Basis::interaction);

	//enable / disable parts of the simulated physics
	params->Set_IncludeOscillations(useOscillation);
	params->Set_NonCoherentRhoTerms(useNonCoherentRhoTerms);
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

	/*
	//Its also possible to initialize flux from a 4D array of dimensions [number of paths][number of neutrino types][number of energies][number of flavors] embedded into a 1D std::vector
	unsigned int neutypes = (neutrinoType == nusquids::NeutrinoType::both ? 2 : 1);
	std::vector<double> fluxvector(n_cosines * neutypes * n_energies * n_neutrinos);

	size_t k = 0;
	for(size_t index_cosine = 0; index_cosine < n_cosines; index_cosine++){
		for(size_t index_rho = 0; index_rho < neutypes; index_rho++){
			for(size_t index_energy = 0; index_energy < n_energies; index_energy++){
				for(size_t flv = 0; flv < n_neutrinos; flv++){
					initialFlux[k++] = fluxfunc(index_cosine, index_rho, index_energy, flv);
				}
			}
		}
	}

	params->setInitialFlux(fluxvector, nusquids::Basis::flavor);
	*/


    switch(n_neutrinos){
        case 3:{
            run<3, BATCH_SIZE_LIMIT>(params);
            break;}
        case 4:{
            run<4, BATCH_SIZE_LIMIT>(params);
            break;}
		default: printf("error\n");
    }

	return 0;
}


int main(int argc, char** argv){
	int r = atmospheric(argc, argv);
	cudaDeviceReset(); CUERR;

    return r;
}
