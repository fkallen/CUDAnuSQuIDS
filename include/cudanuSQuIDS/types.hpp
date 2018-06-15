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

#ifndef CUDANUSQUIDS_TYPES_HPP
#define CUDANUSQUIDS_TYPES_HPP

namespace cudanusquids{

    enum class SolverType {Version1, Version2};

	namespace detail{

		struct Flags{
			bool useCoherentRhoTerms = false;
			bool useNonCoherentRhoTerms = false;
			bool useInteractionsRhoTerms = false;
			bool canUseInteractions = false;
			bool useNCInteractions = false;
			bool useTauRegeneration = false;
			bool useGlashowResonance = false;
		};

        enum class EvalType {None, NodeFlavor, NodeMass};
	}

	namespace ode{

        //add additional implementations here
		enum class StepperType{
            RK4
        };

        enum class Status{
            success, failure
        };

        struct RKstats{
            unsigned int steps = 0;
            unsigned int repeats = 0;
            Status status = Status::success;
        };

	}
}


#endif
