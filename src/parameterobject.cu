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
