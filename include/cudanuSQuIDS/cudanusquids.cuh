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

#ifndef CUDANUSQUIDS_CUDANUSQUIDS_CUH
#define CUDANUSQUIDS_CUDANUSQUIDS_CUH

#include <cudanuSQuIDS/cudahelpers.cuh>
#include <cudanuSQuIDS/parameterobject.hpp>
#include <cudanuSQuIDS/propagator.cuh>
#include <cudanuSQuIDS/bodygpu.cuh>
#include <cudanuSQuIDS/ode.cuh>
#include <cudanuSQuIDS/physics.cuh>
#include <cudanuSQuIDS/types.hpp>
#include <cudanuSQuIDS/interaction_structure.hpp>

#include <nuSQuIDS/xsections.h>
#include <nuSQuIDS/nuSQuIDS.h>


#include <vector>
#include <string>

#include <stdexcept>
#include <thread>
#include <memory>
#include <algorithm>

#include <assert.h>

namespace cudanusquids{
	
/*	
	This is main class of CudaNusquids.
	
	NFLVS: number of neutrino flavors
	body_t: body type
	BATCH_SIZE_LIMIT: Upper limit to the maximum number of paths which are evolved in parallel. Can be used to control GPU memory usage.
	Ops: The function object to use for calculation of the operators
*/

template<unsigned int NFLVS, class body_t, unsigned int BATCH_SIZE_LIMIT = 400, class Ops = PhysicsOps<NFLVS, body_t>>
class CudaNusquids{

	std::vector<body_t> bodies;
	std::vector<typename body_t::Track> tracks;
	std::vector<Propagator<NFLVS, body_t, BATCH_SIZE_LIMIT, Ops>> propagators;
	std::vector<std::vector<size_t>> cosineIndices;
	std::vector<size_t> localCosineIndices;	
	
    std::shared_ptr<InteractionStructure> intstruct;
	std::shared_ptr<ParameterObject> parameterObject;

	int nGpus;

	CudaNusquids(const CudaNusquids& other) = delete;
	CudaNusquids& operator=(const CudaNusquids& other) = delete;
public:
	~CudaNusquids(){};
	
	//Construct CudaNusquids from ParameterObject
	CudaNusquids(std::shared_ptr<ParameterObject>& params)
		:  parameterObject(params){

		if(!params) throw std::runtime_error("CudaNusquids::CudaNusquids: params are null");

		if(parameterObject->Get_DeviceIds().empty())
			throw std::runtime_error("CudaNusquids::CudaNusquids : no device specified");

		cudaGetDeviceCount(&nGpus); CUERR;

		for(int id : parameterObject->Get_DeviceIds()){
			if(id < 0 || id >= nGpus)
				throw std::runtime_error("CudaNusquids::CudaNusquids : invalid device id " + std::to_string(id));
		}

		bodies.resize(parameterObject->Get_DeviceIds().size());
		cosineIndices.resize(parameterObject->Get_DeviceIds().size());
		localCosineIndices.resize(parameterObject->getNumPaths());

		for(size_t icos = 0; icos < parameterObject->getNumPaths(); icos++){

			size_t deviceIndex = getCosineDeviceIndex(icos);

			cosineIndices[deviceIndex].push_back(icos);
			localCosineIndices[icos] = cosineIndices[deviceIndex].size() - 1;
		}

		for(size_t i = 0; i < parameterObject->Get_DeviceIds().size(); i++){
			propagators.emplace_back(parameterObject->Get_DeviceIds()[i], cosineIndices[i].size(), parameterObject);
		}

		parameterObject->initializeProjectors();

        mixingParametersChanged();
        simulationFlagsChanged();
        initialFluxChanged();
        additionalDataChanged();
	}
	
	CudaNusquids(CudaNusquids&& other){
			*this = other;
	}
	
	CudaNusquids& operator=(CudaNusquids&& other){
		bodies = std::move(other.bodies);
		tracks = std::move(other.tracks);
		propagators = std::move(other.propagators);
		cosineIndices = std::move(other.cosineIndices);
		localCosineIndices = std::move(other.localCosineIndices);
		intstruct = std::move(other.intstruct);
		parameterObject = std::move(other.parameterObject);

		nGpus = other.nGpus;
	}	

	// map cosine / track to gpu
	size_t getCosineDeviceIndex(size_t index_path) const{
		size_t id = 0;

		// cyclic distribution. this creates a balanced work load per gpu if path lengths decrease / increase with index_path

		id = index_path % parameterObject->Get_DeviceIds().size();

		return id;
	}

	//Get expectation value
	double EvalMassAtNode(size_t flavor, size_t index_path, size_t index_rho, size_t index_energy){

		const size_t deviceIndex = getCosineDeviceIndex(index_path);

		const size_t local_index_path = localCosineIndices[index_path];

		const double flux = propagators[deviceIndex].EvalMassAtNode(flavor, local_index_path, index_rho, index_energy);

		return flux;
	}
	
	//Get expectation value
	double EvalFlavorAtNode(size_t flavor, size_t index_path, size_t index_rho, size_t index_energy){

		const size_t deviceIndex = getCosineDeviceIndex(index_path);

		const size_t local_index_path = localCosineIndices[index_path];

		const double flux = propagators[deviceIndex].EvalFlavorAtNode(flavor, local_index_path, index_rho, index_energy);

		return flux;
	}
	
	
	//Evolve all bins until end of path is reached
	void evolve(){
		// distribute cosineList, tracks among gpus
		for(size_t i = 0; i < parameterObject->Get_DeviceIds().size(); i++){
			std::vector<double> myCos(cosineIndices[i].size());
			std::vector<typename body_t::Track> myTracks(cosineIndices[i].size());

			// make list of cosine values for this gpu
			std::transform(cosineIndices[i].begin(), cosineIndices[i].end(),
							myCos.begin(),
							[&](size_t icos){ return parameterObject->getPathParameterList()[icos]; });

			for(size_t c = 0; c < cosineIndices[i].size(); c++){
				myTracks[c] = tracks[cosineIndices[i][c]];
			}

			propagators[i].setCosineList(myCos);
			propagators[i].setTracks(myTracks);
		}

		// use one thread per gpu
		std::vector<std::thread> threads;

		auto evolvefunc = [&](int i){
			cudaSetDevice(parameterObject->Get_DeviceIds()[i]); CUERR;
			propagators[i].evolve();
		};

		for(size_t i = 0; i < parameterObject->Get_DeviceIds().size(); i++){
			threads.emplace_back(evolvefunc, i);
		}

		for(auto& thread : threads)
			thread.join();
	}

	//notify CudaNusquids that mixing parameters in parameter object changed
	void mixingParametersChanged(){
		for(auto& nus : propagators)
			nus.mixingParametersChanged();
	}

	//notify CudaNusquids that simulation parameters which enable / disable physics in parameter object changed
	void simulationFlagsChanged(){
		if(parameterObject->GetUseInteractions() && !intstruct){
            intstruct = parameterObject->make_InteractionStructure();

    		for(size_t i = 0; i < parameterObject->Get_DeviceIds().size(); i++){
    			propagators[i].intstructcpu = intstruct;
    		}
		}
		for(auto& nus : propagators)
			nus.simulationFlagsChanged();
	}

	//notify CudaNusquids that initial flux in parameter object changed
    void initialFluxChanged(){
        const size_t fluxesPerCosine = parameterObject->GetNumRho() * parameterObject->GetNumE() * NFLVS;

		// distribute flux, cosineList, tracks among gpus
		for(size_t i = 0; i < parameterObject->Get_DeviceIds().size(); i++){
			std::vector<double> fluxForGpu(cosineIndices[i].size() * fluxesPerCosine);

			auto iter = fluxForGpu.begin();

			for(size_t icos = 0; icos < parameterObject->getNumPaths(); icos++){
				size_t deviceIndex = getCosineDeviceIndex(icos);
				if(deviceIndex == i){
					iter = std::copy(parameterObject->Get_InitialFlux().begin() + icos * fluxesPerCosine,
								parameterObject->Get_InitialFlux().begin() + (icos + 1) * fluxesPerCosine,
								iter);
				}
			}

			propagators[i].setInitialFlux(fluxForGpu);
        }
    }

    //notify CudaNusquids that additional data in parameter object changed
    void additionalDataChanged(){
        for(auto& nus : propagators)
			nus.additionalDataChanged();
    }

    //Set body for gpu deviceId. The deviceId must be identical to the one for which the body was created
	void setBody(const body_t& body_, int deviceId){
		auto it = std::find(parameterObject->Get_DeviceIds().begin(), parameterObject->Get_DeviceIds().end(), deviceId);
		if(it == parameterObject->Get_DeviceIds().end())
			throw std::runtime_error("CudaNusquids::setBody: deviceId " + std::to_string(deviceId) + " not specified in parameter object");
		auto dist = std::distance(parameterObject->Get_DeviceIds().begin(), it);
		bodies[dist] = body_;
		propagators[dist].setBody(body_);
	}

	//set neutrino tracks. must be one track per path
	void setTracks(const std::vector<typename body_t::Track>& tracks_){
		if(tracks_.size() != parameterObject->getNumPaths())
			throw std::runtime_error("CudaNusquids::setTracks error, must provide one track per cosine bin.");
		tracks = tracks_;
	}

	//get Runge-Kutta stats after evolution
    ode::RKstats getRKstats(size_t index_path) const{
        const size_t deviceIndex = getCosineDeviceIndex(index_path);
		const size_t local_index_path = localCosineIndices[index_path];
        return propagators[deviceIndex].getRKstats(local_index_path);
    }

};


}








#endif
