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

#ifndef CUDANUSQUIDS_SOLVER2D_KERNELS_HPP
#define CUDANUSQUIDS_SOLVER2D_KERNELS_HPP

namespace cudanusquids{

    namespace ode{

        void callSolver2dHadjustKernelAsync(int* retVals, unsigned int ord,
                    const size_t* activeIndices, size_t nIndices,
                    const double* y, const double* yerr, const double* yp,
                    double* h, size_t systemsize, double eps_abs, double eps_rel, size_t dimx, size_t dimy, size_t pitchx,
                    dim3 grid, dim3 block, cudaStream_t stream);

    }

}

#endif
