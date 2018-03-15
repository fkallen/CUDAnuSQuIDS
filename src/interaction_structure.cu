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

#include <cudanuSQuIDS/interaction_structure.hpp>
#include <cudanuSQuIDS/cudahelpers.cuh>
#include <cudanuSQuIDS/const.cuh>

namespace cudanusquids{

    InteractionStructureGpu make_InteractionStructureGpu(int deviceId, const InteractionStructure& intstruct){
        int nGpus;
        cudaGetDeviceCount(&nGpus); CUERR;

        if(deviceId < 0 || deviceId >= nGpus)
            throw std::runtime_error("make_InteractionStructureGpu : invalid device id");

        cudaSetDevice(deviceId); CUERR;

        InteractionStructureGpu intstructgpu;
        intstructgpu.deviceId = deviceId;

        cudaMalloc((void**)&intstructgpu.dNdE_CC, sizeof(double) * intstruct.dNdE_CC.size()); CUERR;
        cudaMalloc((void**)&intstructgpu.dNdE_NC, sizeof(double) * intstruct.dNdE_NC.size()); CUERR;
        cudaMalloc((void**)&intstructgpu.dNdE_GR, sizeof(double) * intstruct.dNdE_GR.size()); CUERR;
        cudaMalloc((void**)&intstructgpu.dNdE_tau_all, sizeof(double) * intstruct.dNdE_tau_all.size()); CUERR;
        cudaMalloc((void**)&intstructgpu.dNdE_tau_lep, sizeof(double) * intstruct.dNdE_tau_lep.size()); CUERR;
        cudaMalloc((void**)&intstructgpu.sigma_CC, sizeof(double) * intstruct.sigma_CC.size()); CUERR;
        cudaMalloc((void**)&intstructgpu.sigma_NC, sizeof(double) * intstruct.sigma_NC.size()); CUERR;
        cudaMalloc((void**)&intstructgpu.sigma_GR, sizeof(double) * intstruct.sigma_GR.size()); CUERR;

        cudaMemcpy(intstructgpu.dNdE_CC, intstruct.dNdE_CC.data(), sizeof(double) * intstruct.dNdE_CC.size(), H2D); CUERR;
        cudaMemcpy(intstructgpu.dNdE_NC, intstruct.dNdE_NC.data(), sizeof(double) * intstruct.dNdE_NC.size(), H2D); CUERR;
        cudaMemcpy(intstructgpu.dNdE_GR, intstruct.dNdE_GR.data(), sizeof(double) * intstruct.dNdE_GR.size(), H2D); CUERR;
        cudaMemcpy(intstructgpu.dNdE_tau_all, intstruct.dNdE_tau_all.data(), sizeof(double) * intstruct.dNdE_tau_all.size(), H2D); CUERR;
        cudaMemcpy(intstructgpu.dNdE_tau_lep, intstruct.dNdE_tau_lep.data(), sizeof(double) * intstruct.dNdE_tau_lep.size(), H2D); CUERR;
        cudaMemcpy(intstructgpu.sigma_CC, intstruct.sigma_CC.data(), sizeof(double) * intstruct.sigma_CC.size(), H2D); CUERR;
        cudaMemcpy(intstructgpu.sigma_NC, intstruct.sigma_NC.data(), sizeof(double) * intstruct.sigma_NC.size(), H2D); CUERR;
        cudaMemcpy(intstructgpu.sigma_GR, intstruct.sigma_GR.data(), sizeof(double) * intstruct.sigma_GR.size(), H2D); CUERR;

        return intstructgpu;
    }

    void destroy_InteractionStructureGpu(InteractionStructureGpu& intstructgpu){
        cudaSetDevice(intstructgpu.deviceId); CUERR;

        cudaFree(intstructgpu.dNdE_CC); CUERR;
        cudaFree(intstructgpu.dNdE_NC); CUERR;
        cudaFree(intstructgpu.dNdE_GR); CUERR;
        cudaFree(intstructgpu.dNdE_tau_all); CUERR;
        cudaFree(intstructgpu.dNdE_tau_lep); CUERR;
        cudaFree(intstructgpu.sigma_CC); CUERR;
        cudaFree(intstructgpu.sigma_NC); CUERR;
        cudaFree(intstructgpu.sigma_GR); CUERR;
    }

    InteractionStateBufferGpu make_InteractionStateBufferGpu(int deviceId, size_t n_rhos, size_t n_flvs, size_t n_energies, int npaths){
        int nGpus;
        cudaGetDeviceCount(&nGpus); CUERR;

        if(deviceId < 0 || deviceId >= nGpus)
            throw std::runtime_error("make_InteractionStructureGpu : invalid device id");

        cudaSetDevice(deviceId); CUERR;

        InteractionStateBufferGpu retval(npaths);
		for(auto& s : retval)
			s.deviceId = deviceId;

        size_t s3 = n_rhos * n_flvs * n_energies;
        size_t s4 = n_energies;

        double* invlen_CC;
        double* invlen_NC;
        double* invlen_GR;
        double* invlen_INT;

        cudaMalloc((void**)&invlen_CC, sizeof(double) * s3 * npaths); CUERR;
        cudaMalloc((void**)&invlen_NC, sizeof(double) * s3 * npaths); CUERR;
        cudaMalloc((void**)&invlen_GR, sizeof(double) * s4 * npaths); CUERR;
        cudaMalloc((void**)&invlen_INT, sizeof(double) * s3 * npaths); CUERR;

        for(int i = 0; i < npaths; i++){
                retval[i].invlen_CC = invlen_CC + i * s3;
                retval[i].invlen_NC = invlen_NC + i * s3;
                retval[i].invlen_GR = invlen_GR + i * s4;
                retval[i].invlen_INT = invlen_INT + i * s3;
        }

        return retval;
    }

    void destroy_InteractionStateBufferGpu(InteractionStateBufferGpu& buffergpu){
            cudaFree(buffergpu[0].invlen_CC); CUERR;
            cudaFree(buffergpu[0].invlen_NC); CUERR;
            cudaFree(buffergpu[0].invlen_GR); CUERR;
            cudaFree(buffergpu[0].invlen_INT); CUERR;
    }

}
