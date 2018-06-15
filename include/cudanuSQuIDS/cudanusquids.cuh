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



///\class CudaNusquids
///\brief This is the main class of of the library.
/// @param NFLVS_ Number of neutrino flavors
/// @param body_t The body type which provides density lookup
/// @param Op_t The operators for physics simulation. (optional, default behaviour provided by struct PhysicsOps)
template<int NFLVS_, class body_t, class Op_t = PhysicsOps>
class CudaNusquids{
    template<int, class, class> friend class CudaNusquids;
public:
    static constexpr int NFLVS = NFLVS_;
    using Body = body_t;
    using Ops = Op_t;
private:
    using Track = typename Body::Track;
    using Propagator_t = Propagator<NFLVS, Body, Ops>;

	std::vector<Body> bodies;
	std::vector<Track> tracks;
	std::vector<Propagator_t> propagators;
	std::vector<std::vector<size_t>> cosineIndices;
	std::vector<size_t> localCosineIndices;

    std::shared_ptr<InteractionStructure> intstruct;
	std::shared_ptr<ParameterObject> parameterObject;

	int nGpus;
    bool bodiesAreSet = false;
    bool tracksAreSet = false;

	CudaNusquids(const CudaNusquids& other) = delete;
	CudaNusquids& operator=(const CudaNusquids& other) = delete;
public:

    /// \brief Constructor to change body type.
    ///
    /// \details other will be moved to *this. other must not be accessed afterwards.
    template<class Nus_t>
    CudaNusquids(Nus_t& other){
        static_assert(NFLVS == other.NFLVS, "Body conversion: NFLVS does not match");

        for(auto& otherprop : other.propagators){
            propagators.emplace_back(otherprop);
        }

		cosineIndices = std::move(other.cosineIndices);
		localCosineIndices = std::move(other.localCosineIndices);
		intstruct = std::move(other.intstruct);
		parameterObject = std::move(other.parameterObject);

        bodies.resize(parameterObject->Get_DeviceIds().size());

		nGpus = other.nGpus;

        bodiesAreSet = false;
        tracksAreSet = false;
    }

    /// \brief Construct CudaNusquids from ParameterObject
	///
    /// \details batchsizeLimit determines the maximum number of neutrino paths, which are processed simultaneously per GPU.
    /// If the number of paths is greater than batchsizeLimit, paths are processed in chunks of size batchsizeLimit
    /// This effectively controls the GPU memory usage, since there only has to be enough memory to process batchsizeLimit paths
	CudaNusquids(std::shared_ptr<ParameterObject>& params, int batchsizeLimit)
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

		for(int icos = 0; icos < parameterObject->getNumPaths(); icos++){

			int deviceIndex = getCosineDeviceIndex(icos);

			cosineIndices[deviceIndex].push_back(icos);
			localCosineIndices[icos] = cosineIndices[deviceIndex].size() - 1;
		}

		for(size_t i = 0; i < parameterObject->Get_DeviceIds().size(); i++){
			propagators.emplace_back(parameterObject->Get_DeviceIds()[i],
                                    cosineIndices[i].size(),
                                    batchsizeLimit,
                                    parameterObject);
		}

		parameterObject->initializeProjectors();

        mixingParametersChanged();
        simulationFlagsChanged();
        initialFluxChanged();
        additionalDataChanged();
	}

    /// \brief Move constructor
	CudaNusquids(CudaNusquids&& other){
			*this = other;
	}

    /// \brief Move assignment operator
	CudaNusquids& operator=(CudaNusquids&& other){
		bodies = std::move(other.bodies);
		tracks = std::move(other.tracks);
		propagators = std::move(other.propagators);
		cosineIndices = std::move(other.cosineIndices);
		localCosineIndices = std::move(other.localCosineIndices);
		intstruct = std::move(other.intstruct);
		parameterObject = std::move(other.parameterObject);

        bodiesAreSet = other.bodiesAreSet;
        tracksAreSet = other.tracksAreSet;
		nGpus = other.nGpus;

        other.bodiesAreSet = false;
        other.tracksAreSet = false;
	}

	/// \brief Return device Id of GPU which processes the index_path-th path
	int getCosineDeviceIndex(int index_path) const{
		int id = 0;

		// cyclic distribution. this creates a balanced work load per gpu if path lengths decrease / increase with index_path

		id = index_path % parameterObject->Get_DeviceIds().size();

		return id;
	}

	/// \brief Get expectation value in mass Basis
	double EvalMassAtNode(int flavor, int index_path, int index_rho, int index_energy){

		const int deviceIndex = getCosineDeviceIndex(index_path);

		const int local_index_path = localCosineIndices[index_path];

		const double flux = propagators[deviceIndex].EvalMassAtNode(flavor, local_index_path, index_rho, index_energy);

		return flux;
	}

	/// \brief Get expectation value in flavor Basis
	double EvalFlavorAtNode(int flavor, int index_path, int index_rho, int index_energy){

		const int deviceIndex = getCosineDeviceIndex(index_path);

		const int local_index_path = localCosineIndices[index_path];

		const double flux = propagators[deviceIndex].EvalFlavorAtNode(flavor, local_index_path, index_rho, index_energy);

		return flux;
	}


    /// \brief Evolve neutrinos along the specified paths from path begin to path end
	void evolve(){
        if(!bodiesAreSet || !tracksAreSet){
            throw std::runtime_error("tracks or bodies not set!");
        }
		// distribute cosineList, tracks among gpus
		for(size_t i = 0; i < parameterObject->Get_DeviceIds().size(); i++){
			std::vector<Track> myTracks(cosineIndices[i].size());

			for(size_t c = 0; c < cosineIndices[i].size(); c++){
				myTracks[c] = tracks[cosineIndices[i][c]];
			}

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

    /// \brief Notify a CudaNusquids instance that mixing parameters in parameter object changed
	///
    /// \details Needs to be called before CudaNusquids::evolve() after calls to
    /// ParameterObject::Set_MixingAngle,
    /// ParameterObject::Set_SquareMassDifference,
    /// ParameterObject::Set_CPPhase
	void mixingParametersChanged(){
		for(auto& nus : propagators)
			nus.mixingParametersChanged();
	}

    /// \brief Notify a CudaNusquids instance that simulation parameters which enable / disable physics in parameter object changed
	///
    /// \details Needs to be called before CudaNusquids::evolve() after calls to
    /// ParameterObject::Set_IncludeOscillations,
    /// ParameterObject::Set_NonCoherentRhoTerms,
    /// ParameterObject::Set_InteractionsRhoTerms
    /// ParameterObject::Set_NCInteractions
    /// ParameterObject::Set_TauRegeneration
    /// ParameterObject::Set_GlashowResonance
	void simulationFlagsChanged(){
		if(parameterObject->Get_CanUseInteractions() && !intstruct){
            intstruct = parameterObject->make_InteractionStructure();

    		for(size_t i = 0; i < parameterObject->Get_DeviceIds().size(); i++){
    			propagators[i].setInteractionStructure(intstruct);
    		}
		}
		for(auto& nus : propagators)
			nus.simulationFlagsChanged();
	}

    /// \brief Notify a CudaNusquids instance that initial flux in parameter object changed
	///
    /// \details Needs to be called before CudaNusquids::evolve() after calls to
    /// ParameterObject::setInitialFlux
    void initialFluxChanged(){
        const size_t fluxesPerCosine = parameterObject->GetNumRho() * parameterObject->GetNumE() * NFLVS;

		// distribute flux, cosineList, tracks among gpus
		for(int i = 0; i < int(parameterObject->Get_DeviceIds().size()); i++){
			std::vector<double> fluxForGpu(cosineIndices[i].size() * fluxesPerCosine);

			auto iter = fluxForGpu.begin();

			for(int icos = 0; icos < parameterObject->getNumPaths(); icos++){
				int deviceIndex = getCosineDeviceIndex(icos);
				if(deviceIndex == i){
					iter = std::copy(parameterObject->Get_InitialFlux().begin() + icos * fluxesPerCosine,
								parameterObject->Get_InitialFlux().begin() + (icos + 1) * fluxesPerCosine,
								iter);
				}
			}

			propagators[i].setInitialFlux(fluxForGpu);
        }
    }

    /// \brief Notify a CudaNusquids instance that additional data in parameter object changed
	///
    /// \details Needs to be called before CudaNusquids::evolve() after calls to
    /// ParameterObject::registerAdditionalData
    /// ParameterObject::clearAdditionalData
    void additionalDataChanged(){
        for(auto& nus : propagators)
			nus.additionalDataChanged();
    }

    /// \brief Set body for GPU with device id deviceId
	///
    /// \details The deviceId must be identical to the one for which the body was created
	void setBody(const Body& body_, int deviceId){
		auto it = std::find(parameterObject->Get_DeviceIds().begin(), parameterObject->Get_DeviceIds().end(), deviceId);
		if(it == parameterObject->Get_DeviceIds().end())
			throw std::runtime_error("CudaNusquids::setBody: deviceId " + std::to_string(deviceId) + " not specified in parameter object");
		auto dist = std::distance(parameterObject->Get_DeviceIds().begin(), it);
		bodies[dist] = body_;
		propagators[dist].setBody(body_);

        bodiesAreSet = true;
	}

    /// \brief Set neutrino tracks. Must be one track per path
	void setTracks(const std::vector<Track>& tracks_){
		if(tracks_.size() != size_t(parameterObject->getNumPaths()))
			throw std::runtime_error("CudaNusquids::setTracks error, must provide one track per cosine bin.");
		tracks = tracks_;
        tracksAreSet = true;
	}

	/// \brief Get Runge-Kutta stats after evolution
    ///
    /// \details members of RKstats:
    ///    unsigned int steps; // number of required steps
    ///    unsigned int repeats; // number of repeated steps
    ///    Status status; // success or failure
    ode::RKstats getRKstats(int index_path) const{
        const int deviceIndex = getCosineDeviceIndex(index_path);
		const int local_index_path = localCosineIndices[index_path];
        return propagators[deviceIndex].getRKstats(local_index_path);
    }

};


}








#endif
