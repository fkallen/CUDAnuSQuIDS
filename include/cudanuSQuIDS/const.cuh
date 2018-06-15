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

#ifndef CUDANUSQUIDS_CONST_CUH
#define CUDANUSQUIDS_CONST_CUH

#include <cudanuSQuIDS/cudahelpers.cuh>

namespace cudanusquids{



//Contains physical and mathematical constants
struct Const{

	// mathematical constants //

	HOSTDEVICEQUALIFIER
	static constexpr double pi(){
        return 3.14159265358979;	    // Pi
	}

    HOSTDEVICEQUALIFIER
	static constexpr double piby2(){
        return 1.57079632679490;         // Pi/2
	}

	HOSTDEVICEQUALIFIER
	static constexpr double sqrt2(){
        return 1.41421356237310;
    }

	//double ln2;
	// astronomical constants //

    HOSTDEVICEQUALIFIER
	static constexpr double earthradius(){
        return 6371.0;
	}

	HOSTDEVICEQUALIFIER
	static constexpr double sunradius(){
        return 109.0*earthradius();
	}

	///// physics constants/////

	HOSTDEVICEQUALIFIER
	static constexpr double GF(){
        return 1.16639e-23;	            // [eV^-2] Fermi Constant
    }

	HOSTDEVICEQUALIFIER
	static constexpr double Na(){
        return 6.0221415e+23;		    // [mol cm^-3] Avogadro Number
    }

    HOSTDEVICEQUALIFIER
	static constexpr double sw_sq(){
        return 0.2312;                 // [dimensionless] sin(th_weinberg) ^2
	}

	HOSTDEVICEQUALIFIER
	static constexpr double G(){
        return 6.67300e-11;               // [m^3 kg^-1 s^-2]
	}

	HOSTDEVICEQUALIFIER
	static constexpr double alpha(){
	    return 1.0/137.0;              // [dimensionless] fine-structure constant
	}

    //double e_charge;

	/////////// units //////////
	// energy

	HOSTDEVICEQUALIFIER
	static constexpr double TeV(){
	    return 1.0e12;                   // [eV/TeV]
	}

	HOSTDEVICEQUALIFIER
	static constexpr double GeV(){
	    return 1.0e9;                    // [eV/GeV]
	}

	HOSTDEVICEQUALIFIER
	static constexpr double MeV(){
	    return 1.0e6;                    // [eV/MeV]
	}

	HOSTDEVICEQUALIFIER
	static constexpr double keV(){
	    return 1.0e3;                    // [eV/keV]
	}

	HOSTDEVICEQUALIFIER
	static constexpr double eV(){
	    return 1.0;                      // [eV/eV]
	}

	HOSTDEVICEQUALIFIER
	static constexpr double Joule(){
	    return 1/1.60225e-19;          // [eV/J]
	}

	// mass

	HOSTDEVICEQUALIFIER
	static constexpr double kg(){
	    return 5.62e35;                   // [eV/kg]
	}

	HOSTDEVICEQUALIFIER
	static constexpr double gr(){
	    return 1e-3*kg();                   // [eV/g]
	}

	// time

	HOSTDEVICEQUALIFIER
	static constexpr double sec(){
	    return 1.523e15;                 // [eV^-1/s]
	}

	HOSTDEVICEQUALIFIER
	static constexpr double hour(){
	    return 3600.0*sec();              // [eV^-1/h]
	}

	HOSTDEVICEQUALIFIER
	static constexpr double day(){
	    return 24.0*hour();                // [eV^-1/d]
	}

	HOSTDEVICEQUALIFIER
	static constexpr double year(){
	    return 365.0*day();               // [eV^-1/yr]
	}

	// distance

	HOSTDEVICEQUALIFIER
	static constexpr double meter(){
        return 5.06773093741e6;        // [eV^-1/m]
    }

    HOSTDEVICEQUALIFIER
    static constexpr double cm(){
        return 1.0e-2*meter();              // [eV^-1/cm]
    }

    HOSTDEVICEQUALIFIER
	static constexpr double km(){
	    return 1.0e3*meter();               // [eV^-1/km]
	}

	HOSTDEVICEQUALIFIER
	static constexpr double fermi(){
	    return 1.0e-15*meter();          // [eV^-1/fm]
	}

	HOSTDEVICEQUALIFIER
	static constexpr double angstrom(){
        return 1.0e-10*meter();       // [eV^-1/A]
	}

	HOSTDEVICEQUALIFIER
	static constexpr double AU(){
	    return 149.60e9*meter();            // [eV^-1/AU]
	}

	HOSTDEVICEQUALIFIER
	static constexpr double ly(){
	    return 9.4605284e15*meter();        // [eV^-1/ly]
	}

	HOSTDEVICEQUALIFIER
	static constexpr double parsec(){
	    return 3.08568025e16*meter();   // [eV^-1/parsec]
	}

	// luminocity

	HOSTDEVICEQUALIFIER
	static constexpr double picobarn(){
	    return 1.0e-36 * cm() * cm();       // [eV^-2/pb]
	}

	HOSTDEVICEQUALIFIER
	static constexpr double femtobarn(){
	    return 1.0e-39 * cm() * cm();      // [eV^-2/fb]
	}

	// presure

	HOSTDEVICEQUALIFIER
	static constexpr double Pascal(){
	    return Joule()/ meter() / meter() / meter();        // [eV^4/Pa]
	}

	HOSTDEVICEQUALIFIER
	static constexpr double atm(){
	    return 101325.0*Pascal();          // [eV^4/atm]
	}

	// temperature

	HOSTDEVICEQUALIFIER
	static constexpr double Kelvin(){
	    return 1/1.1604505e4;         // [eV/K]
	}

	// electromagnetic units

	/*HOSTDEVICEQUALIFIER
	static constexpr double C(){
	    return 6.24150965e18*e_charge();
	}

	HOSTDEVICEQUALIFIER
	static constexpr double A(){
	    return C()/sec();
	}

	HOSTDEVICEQUALIFIER
	static constexpr double T(){
	    return kg()/(A()*sec()*sec());
	}*/

	// angle
	HOSTDEVICEQUALIFIER
	static constexpr double degree(){
	    return pi()/180.0;              // [rad/degree]
	}

    HOSTDEVICEQUALIFIER
	static constexpr double tau_lifetime(){
	    return 2.906e-13*sec();
	}

	HOSTDEVICEQUALIFIER
	static constexpr double tau_mass(){
	    return 1776.82*MeV();
	}

	HOSTDEVICEQUALIFIER
	static constexpr double muon_lifetime(){
	    return 2.196e-6*sec();
	}

	HOSTDEVICEQUALIFIER
	static constexpr double muon_mass(){
	    return 105.658*MeV();
	}

	HOSTDEVICEQUALIFIER
	static constexpr double electron_mass(){
	    return 0.5109*MeV();
	}

	HOSTDEVICEQUALIFIER
	static constexpr double proton_mass(){
	    return 938.272*MeV();
	}

    HOSTDEVICEQUALIFIER
	static constexpr double neutron_mass(){
	    return 939.565*MeV();
	}

	//double HI_constants;
	HOSTDEVICEQUALIFIER
	static constexpr double HI_constants(){
        return sqrt2() * GF() * Na() / cm() / cm() / cm();
    }

    HOSTDEVICEQUALIFIER
    Const()
    {
        //ln2 = log(2.0);                 // log[2]
        //e_charge = sqrt(4.*pi()*alpha());
    }
};



}

#endif
