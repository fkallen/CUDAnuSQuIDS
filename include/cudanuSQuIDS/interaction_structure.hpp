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

#ifndef CUDANUSQUIDS_INTERACTION_STRUCTURE_CUH
#define CUDANUSQUIDS_INTERACTION_STRUCTURE_CUH

#include <cudanuSQuIDS/cudahelpers.cuh>

#include <vector>
#include <cstddef>

namespace cudanusquids{

    /*
        GPU cross sections.
    */

    struct InteractionStructure{
        std::vector<double> dNdE_CC; // size s1
        std::vector<double> dNdE_NC; // size s1
        std::vector<double> dNdE_GR; // size s2
        std::vector<double> dNdE_tau_all; // size s2
        std::vector<double> dNdE_tau_lep; // size s2
        std::vector<double> sigma_CC; // size s3
        std::vector<double> sigma_NC; // size s3
        std::vector<double> sigma_GR; // size s4

        size_t s1;
        size_t s2;
        size_t s3;
        size_t s4;

        size_t n_rhos;
        size_t n_flvs;
        size_t n_energies;
    };

    struct InteractionStructureGpu{

        double* dNdE_CC = nullptr; // [rho, flavor, energy_in, energy_out]
        double* dNdE_NC = nullptr; // [rho, flavor, energy_in, energy_out]
        double* dNdE_GR = nullptr; // [energy_in, energy_out]
        double* dNdE_tau_all = nullptr; // [energy_in, energy_out]
        double* dNdE_tau_lep = nullptr; // [energy_in, energy_out]

        double* sigma_CC = nullptr; // [rho, flavor, energy]
        double* sigma_NC = nullptr; // [rho, flavor, energy]
        double* sigma_GR = nullptr; // [energy]

        int deviceId = -1;
    };

    struct InteractionStateGpu{

        double* invlen_CC = nullptr; // [rho, flavor, energy]
        double* invlen_NC = nullptr; // [rho, flavor, energy]
        double* invlen_GR = nullptr; // [energy]
        double* invlen_INT = nullptr; // [rho, flavor, energy]

        int deviceId = -1;
    };

    typedef std::vector<InteractionStateGpu> InteractionStateBufferGpu;

    InteractionStructureGpu make_InteractionStructureGpu(int deviceId, const InteractionStructure& intstruct);
    void destroy_InteractionStructureGpu(InteractionStructureGpu& intstructgpu);

    InteractionStateBufferGpu make_InteractionStateBufferGpu(int deviceId, size_t n_rhos, size_t n_flvs, size_t n_energies, int npaths);
    void destroy_InteractionStateBufferGpu(InteractionStateBufferGpu& buffergpu);


}



#endif
