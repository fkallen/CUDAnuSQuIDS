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

#ifndef CUDANUSQUIDS_HPC_HELPERS_HPP
#define CUDANUSQUIDS_HPC_HELPERS_HPP

#include <iostream>
#include <chrono>

#define TIMERSTARTCPU(label)                                                  \
	std::chrono::time_point<std::chrono::system_clock> a##label, b##label; \
	a##label = std::chrono::system_clock::now();

#define TIMERSTOPCPU(label)                                                   \
	b##label = std::chrono::system_clock::now();                           \
	std::chrono::duration<double> delta##label = b##label-a##label;        \
	std::cout << "# elapsed time ("<< #label <<"): "                       \
				<< delta##label.count()  << " s" << std::endl;


#endif
