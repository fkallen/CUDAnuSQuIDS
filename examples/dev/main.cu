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
#include "../nsi_atmospheric/customphysics.cuh"
#include <cudanuSQuIDS/cudanusquids.cuh>
#include <cudanuSQuIDS/hpc_helpers.hpp>

#include <nuSQuIDS/xsections.h>
#include <nuSQuIDS/tools.h>

#include <vector>
#include <iostream>

#include <memory>
#include <algorithm>
#include <iterator>
#include <iomanip>
#include <cstdint>
#include <fstream>

#define DO_OUTPUT


template<unsigned int NFLVS, unsigned int BATCH_SIZE_LIMIT>
void run(std::shared_ptr<cudanusquids::ParameterObject>& params){

    using Bodytype = cudanusquids::EarthAtm;
    //using Bodytype = cudanusquids::SunASnu;

    cudanusquids::CudaNusquids<NFLVS, Bodytype, BATCH_SIZE_LIMIT, CustomPhysicsOps<NFLVS, Bodytype>> nus(params);

    std::vector<Bodytype> gpuBodies(params->Get_DeviceIds().size());
    std::vector<typename Bodytype::Track> tracks;
	tracks.reserve(params->getNumPaths());

    for(size_t i = 0; i < params->Get_DeviceIds().size(); i++){
        gpuBodies[i] = cudanusquids::EarthAtm::make_body_gpu(params->Get_DeviceIds()[i], "../data/astro/EARTH_MODEL_PREM.dat");
        //gpuBodies[i] = cudanusquids::SunASnu::make_body_gpu(params->Get_DeviceIds()[i], "../data/astro/bs05_agsop.dat");
        nus.setBody(gpuBodies[i], i);
    }

	for(size_t i = 0; i < params->getNumPaths(); i++){
		tracks.emplace_back(params->getPathParameterList()[i]);
	}

	nus.setTracks(tracks);

    double epsilon_mutau = 2e-2;

	gsl_matrix_complex * M = gsl_matrix_complex_calloc(3,3);
    gsl_complex c {{ epsilon_mutau , 0.0 }};
    gsl_matrix_complex_set(M,2,1,c);
    gsl_matrix_complex_set(M,1,2,gsl_complex_conjugate(c));
    squids::SU_vector NSI = squids::SU_vector(M);
    NSI.RotateToB1(params->GetParams());
    gsl_matrix_complex_free(M);

	std::vector<double> nsicomponents = NSI.GetComponents();
	params->registerAdditionalData(sizeof(double) * nsicomponents.size(), (const char*)nsicomponents.data());
	params->registerAdditionalData(sizeof(double) * nsicomponents.size() * params->GetNumE()
                                    * std::min(params->getNumPaths(), size_t(BATCH_SIZE_LIMIT)));
	nus.additionalDataChanged();

TIMERSTARTCPU(evolve);

    nus.evolve();

TIMERSTOPCPU(evolve);
    CUERR;

    // check that every path was solved successfully and print some stats;

    for(size_t i = 0; i < params->getNumPaths(); i++){
        const auto& stats = nus.getRKstats(i);

        if(stats.status != cudanusquids::ode::Status::success)
            std::cout << "cosine " << params->getPathParameterList()[i] << " failed after " << stats.steps << " steps." << std::endl;

        //std::cout << i << " " << stats.steps << " " << stats.repeats << " " << stats.status << std::endl;
    }

#ifdef DO_OUTPUT
    TIMERSTARTCPU(fileoutput);

    std::ofstream out("out.txt");
    out << params->getNumPaths() << " " << params->GetNumE() << " " << params->GetNumRho() << " " << params->GetNumNeu() << '\n';
    for(size_t flv = 0; flv < params->GetNumNeu(); flv++){
        out << "Flv " << flv << '\n';
        for(size_t i = 0; i <params->getNumPaths(); i++){
            out << "cos(th) = " << params->getPathParameterList()[i] << "\n";
            for(size_t j = 0; j < params->GetNumE(); j++){
                for(size_t k = 0; k < params->GetNumRho(); k++){
                    const double val = nus.EvalFlavorAtNode(flv, i, k, j); CUERR;
                    out << std::setprecision(20) << val << " ";
                }
            }
            out << '\n';
        }
        out << '\n';
    }

    TIMERSTOPCPU(fileoutput);
#endif

    //free gpu data
    for(auto& body : gpuBodies){
    	Bodytype::destroy_body_gpu(body);
    }
}

int multigpu(int argc, char** argv){

    using Const = cudanusquids::Const;

    if(argc < 2) return -1;

    std::string mode(argv[1]);
    if(mode != "c" && mode != "g") return -1;

	double Emin=1.e1 * Const::GeV();
	double Emax=1.e6 * Const::GeV();
	unsigned int n_energies = 1;

	double czmin = -1;
	double czmax = 0;
	unsigned int n_cosines = 1;

	nusquids::NeutrinoType neutrinoType = nusquids::NeutrinoType::neutrino;

	unsigned int n_neutrinos = 3;

	unsigned int flags = 1;
	unsigned int nSteps = 2000;

	std::vector<int> usableDeviceIds;

	if(argc >= 12){
		czmin= atof(argv[2]);
		czmax= atof(argv[3]);
		n_cosines = atoi(argv[4]);
		Emin= atof(argv[5]) * Const::GeV();
		Emax= atof(argv[6]) * Const::GeV();
		n_energies = atoi(argv[7]);
		int type = atoi(argv[8]);
		if( type > 0 && type < 4 )
			neutrinoType = nusquids::NeutrinoType(type);
		n_neutrinos = atoi(argv[9]);
		flags = atoi(argv[10]);
		nSteps = atoi(argv[11]);
	}

	if(argc > 12){
		for(int i = 12; i < argc; i++)
			usableDeviceIds.push_back(atoi(argv[i]));
	}

	assert(usableDeviceIds.size() > 0);


	bool useOscillation = ((flags >> 0) & 0x1);  // 1
	bool useNonCoherentRhoTerms = ((flags >> 1) & 0x1);// 2
	bool useNCInteractions = ((flags >> 2) & 0x1);// 4
	bool useTauRegeneration = ((flags >> 3) & 0x1);// 8
	bool useGlashowResonance = ((flags >> 4) & 0x1);// 16

	bool anyInteractions = useNonCoherentRhoTerms || useNCInteractions
				|| useTauRegeneration || useGlashowResonance;


/*
	std::cout << "cz min : " << czmin << '\n'
		<< "cz max : " << czmax << '\n'
		<< "cosine bins : " << n_cosines << '\n'
		<< "E min : " << Emin / Const::GeV()<< " GeV" << '\n'
		<< "E max : " << Emax / Const::GeV()<< " GeV" << '\n'
		<< "energy bins : " << n_energies << '\n'
		<< "neutrino type : " << neutrinoType << '\n'
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
*/


	auto energyList = nusquids::logspace(Emin, Emax, n_energies);
	auto cosineList = nusquids::linspace(czmin, czmax, n_cosines);

	std::shared_ptr<nusquids::NeutrinoCrossSections> crossSections = std::make_shared<nusquids::NeutrinoDISCrossSectionsFromTables>();
	std::shared_ptr<cudanusquids::ParameterObject> params
        = std::make_shared<cudanusquids::ParameterObject>(cosineList, energyList,
                                                        n_neutrinos, neutrinoType,
                                                        anyInteractions, crossSections);

	params->Set_Basis(nusquids::Basis::interaction);

	params->Set_IncludeOscillations(useOscillation);
	params->Set_NonCoherentRhoTerms(useNonCoherentRhoTerms);
	params->Set_NCInteractions(useNCInteractions);
	params->Set_TauRegeneration(useTauRegeneration);
	params->Set_GlashowResonance(useGlashowResonance);

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

    params->Set_NumSteps(nSteps);
    params->Set_rel_error(1.e-6);
    params->Set_abs_error(1.e-6);
    params->Set_StepperType(cudanusquids::ode::StepperType::RK4);


    if(mode == "c")
        params->Set_SolverType(cudanusquids::SolverType::Version1);
    else if(mode == "g")
        params->Set_SolverType(cudanusquids::SolverType::Version2);
    else
        std::cout << "invalid mode" << std::endl;

    params->Set_DeviceIds(usableDeviceIds);
    params->Set_ProgressBar(usableDeviceIds.size() == 1);

    auto flux = [&](int icos, int irho, int ien, int iflv){
        if(iflv == 1){ // muon only flux
            return 1.0;
        }else
            return 0.0;
    };

    params->setInitialFlux(flux, nusquids::Basis::flavor);

    switch(n_neutrinos){
        case 3:{
            run<3, 2000>(params);
            break;}
        case 4:{
            run<4, 2000>(params);
            break;}
            default: printf("error\n");
    }

	return 0;
}


int main(int argc, char** argv){
	int r = multigpu(argc, argv);
	cudaDeviceReset(); CUERR;

    return r;
}
