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

#ifndef CUDANUSQUIDS_ODE_CUH
#define CUDANUSQUIDS_ODE_CUH

/*
    Every Runge-Kutta related include file
*/


#include <cudanuSQuIDS/solver2d.hpp>
#include <cudanuSQuIDS/solvergpu.cuh>

#include <cudanuSQuIDS/rk42d.hpp>
#include <cudanuSQuIDS/rk4.cuh>

#include <cudanuSQuIDS/types.hpp>


/* How to integrate more Runge Kutta steppers into CudaNusquids:

   - include both the stepper for Version1 (2d) and the stepper for Version 2 in ode.cuh.
   - in types.hpp, add new enumeration value to cudanusquids::ode::StepperType
   - in propagator.cuh in struct PropagatorImpl, add case to switch statements in functions evolveVersion1 and evolveVersion2
     to select the new enumeration value for the new Runge Kutta stepper
*/






#endif
