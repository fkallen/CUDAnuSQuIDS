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

#ifndef CUDANUSQUIDS_BODYGPU_CUH
#define CUDANUSQUIDS_BODYGPU_CUH

#include <cudanuSQuIDS/cudahelpers.cuh>
#include <cudanuSQuIDS/akima_interpolation.cuh>
#include <cudanuSQuIDS/const.cuh>
#include <cudanuSQuIDS/cudautils.cuh>

#include <nuSQuIDS/tools.h>

#include <string>
#include <memory>

namespace cudanusquids{


    /*
        CudaNusquids uses static polymorphism to support different body types (Body type is a template parameter).

        A body has to declare a nested class named Track. Track has to provide the following functions:

        __host__ __device__
        double getXBegin() const; // Return the begin of the trajectory.

        __host__ __device__
        double getXEnd() const; // Return the end of the trajectory

        __host__ __device__
        double getCurrentX() const; // Return the current position along the trajectory. getXBegin() <= getCurrentX() <= getXEnd()

        __host__ __device__
        void setCurrentX(double x); // Set current position along the trajectory

        A body has to provide the following functions:

        __device__
        double getDensity(const Track& track) const; // Return density at position track.getCurrentX()

        __device__
        double getYe(const Track& track) const; // Return Ye at position track.getCurrentX()

        __device__
        bool isConstantDensity() const; // Return whether or not Body has a constant density
    */


    /*
        Constant density
    */

    struct ConstantDensity{

		struct Track{
			double x_cur; //natural units
			double x_begin; //natural units
			double x_end; //natural units

			HOSTDEVICEQUALIFIER
			Track() : Track(double(0.0)){}

			HOSTDEVICEQUALIFIER
			Track(double baselineInKilometers)
                : x_cur(0.0),
                  x_begin(0.0),
                  x_end(baselineInKilometers * Const::km()){

			}

			HOSTDEVICEQUALIFIER
			double getXBegin() const{ return x_begin;}

			HOSTDEVICEQUALIFIER
			double getXEnd() const{ return x_end;}

			HOSTDEVICEQUALIFIER
			double getCurrentX() const{ return x_cur;}

			HOSTDEVICEQUALIFIER
			void setCurrentX(double x){x_cur = x;}
		};

		int deviceId;

        double density;
        double ye;

		HOSTDEVICEQUALIFIER
		ConstantDensity() : ConstantDensity(0.0, 0.0) {}

		HOSTDEVICEQUALIFIER
		ConstantDensity(double density, double ye)
            : density(density), ye(ye){}

		HOSTDEVICEQUALIFIER
		double getDensity(const Track& track_earthatm) const{
			return density;
		}

		HOSTDEVICEQUALIFIER
		double getYe(const Track& track_earthatm) const{
			return ye;
		}

		HOSTDEVICEQUALIFIER
		bool isConstantDensity() const{
			return true;
		}

        /*
            Expects file to be a table with at least 3 columns.
            First column contains radii, second column contains densities, third column contains electron fractions.
            Like nuSQuIDS'/data/astro/EARTH_MODEL_PREM.dat
        */
		static ConstantDensity make_body_gpu(int deviceId, double density, double ye){
			return ConstantDensity(density, ye);
		}

		static void destroy_body_gpu(ConstantDensity& constantDensityGpu){
            //nothing to do here.
		}
	};


    /*
        Earth with atmosphere
    */

	struct EarthAtm{

		struct Track{
			double x_cur; //natural units
			double x_begin; //natural units
			double x_end; //natural units

			double cos_zenith = 0.0;
			double earth_radius = 0.0; //natural units
			double athmosphere_height = 0.0; //natural units
			double baseline = 0.0; //natural units

			mutable size_t intercacheD = 0;
			mutable size_t intercacheE = 0;

			HOSTDEVICEQUALIFIER
			Track() : Track(double(0.0)){}

			HOSTDEVICEQUALIFIER
			Track(double coszenith) : cos_zenith(coszenith), earth_radius(6371.0*Const::km()), athmosphere_height(22.*Const::km()){
				const double sinsqphi = 1 - (cos_zenith * cos_zenith);
				const double R = earth_radius;
				const double r = athmosphere_height;
				const double radiusPlusAtm = R + r;

				baseline = sqrt((radiusPlusAtm * radiusPlusAtm) - R * R * sinsqphi) - R * cos_zenith;

				x_cur = 0.0;
				x_begin = 0.0;
				x_end = baseline;
			}

			HOSTDEVICEQUALIFIER
			double getXBegin() const{ return x_begin;}

			HOSTDEVICEQUALIFIER
			double getXEnd() const{ return x_end;}

			HOSTDEVICEQUALIFIER
			double getCurrentX() const{ return x_cur;}

			HOSTDEVICEQUALIFIER
			void setCurrentX(double x){x_cur = x;}
		};

		int deviceId;

		double* radii = nullptr;
		double* densities = nullptr;
		double* electronFractions = nullptr;
		size_t entries = 0;

		double earth_radius = 6371.0; //km SI units
		double atmosphere_height = 22.0; //km SI units

		double radius_min = 0.0;
		double radius_max = 0.0;
		double density_min = 0.0;
		double density_max = 0.0;
		double electronFraction_min = 0.0;
		double electronFraction_max = 0.0;

		Akima::AkimaInterpolationData akimaDensity;
		Akima::AkimaInterpolationData akimaElectronFraction;

		HOSTDEVICEQUALIFIER
		EarthAtm() : EarthAtm(6371.0, 22.0) {}

		HOSTDEVICEQUALIFIER
		EarthAtm(double earthradius, double atmosphereheight):earth_radius(earthradius), atmosphere_height(atmosphereheight){}

		HOSTDEVICEQUALIFIER
		double getDensity(const Track& track_earthatm) const{
			const double xkm = track_earthatm.getCurrentX() / Const::km();
			const double sinsqphi = 1 - track_earthatm.cos_zenith * track_earthatm.cos_zenith;
			const double radiusPlusAtm = earth_radius + atmosphere_height;
			const double dL = sqrt(radiusPlusAtm * radiusPlusAtm - earth_radius * earth_radius * sinsqphi) + earth_radius * track_earthatm.cos_zenith;
			const double r = sqrt(radiusPlusAtm * radiusPlusAtm + (xkm * xkm) - (track_earthatm.baseline / Const::km() + dL) * xkm);
			const double rel_r = r / radiusPlusAtm;

			if ( rel_r < radius_min ){
				return density_min;
			}
			else if ( rel_r > radius_max && rel_r < earth_radius/radiusPlusAtm) {
				return density_max;
			}
			else if ( rel_r > earth_radius/radiusPlusAtm ) {
				double h = atmosphere_height * (rel_r - earth_radius / radiusPlusAtm);
				double h0 = 25.0;
				return 1.05 * exp(-h/h0);
			} else {
				return densityInterpolation(r / earth_radius, &(track_earthatm.intercacheD));
			}
		}

		HOSTDEVICEQUALIFIER
		double getYe(const Track& track_earthatm) const{
			const double xkm = track_earthatm.getCurrentX() / Const::km();
			const double sinsqphi = 1 - track_earthatm.cos_zenith * track_earthatm.cos_zenith;
			const double radiusPlusAtm = earth_radius + atmosphere_height;
			const double dL = sqrt(radiusPlusAtm * radiusPlusAtm - earth_radius * earth_radius * sinsqphi) + earth_radius * track_earthatm.cos_zenith;
			const double r = sqrt(radiusPlusAtm * radiusPlusAtm + (xkm * xkm) - (track_earthatm.baseline / Const::km() + dL) * xkm);
			const double rel_r = r / radiusPlusAtm;

			if ( rel_r < radius_min ){
				return electronFraction_min;
			}
			else if ( rel_r > radius_max && rel_r < earth_radius/radiusPlusAtm) {
				return electronFraction_max;
			}
			else if ( rel_r > earth_radius/radiusPlusAtm ) {
				return 0.494;
			} else {
				return electronFractionInterpolation(rel_r, &(track_earthatm.intercacheE));
			}
		}

		HOSTDEVICEQUALIFIER
		double densityInterpolation(double x, size_t* cache) const{
			return akimaDensity.interpolation(x, cache);
		}

		HOSTDEVICEQUALIFIER
		double electronFractionInterpolation(double x, size_t* cache) const{
			return akimaElectronFraction.interpolation(x, cache);
		}

		HOSTDEVICEQUALIFIER
		bool isConstantDensity() const{
			return false;
		}

        /*
            Expects file to be a table with at least 3 columns.
            First column contains radii, second column contains densities, third column contains electron fractions.
            Like nuSQuIDS'/data/astro/EARTH_MODEL_PREM.dat
        */
		static EarthAtm make_body_gpu(int deviceId, double atm_height, const std::string& filename){
			cudaSetDevice(deviceId); CUERR;

			EarthAtm earthAtmGpu(6371.0, atm_height);
			earthAtmGpu.deviceId = deviceId;

			nusquids::marray<double,2> earth_model = nusquids::quickread(filename);
			size_t rows = earth_model.extent(0);

			std::vector<double> radii(rows);
			std::vector<double> densities(rows);
			std::vector<double> electronFractions(rows);

			for(size_t row = 0; row < rows; row++){
				radii[row] = earth_model[row][0];
				densities[row] = earth_model[row][1];
				electronFractions[row] = earth_model[row][2];
			}

			earthAtmGpu.entries = rows;
			earthAtmGpu.radius_min = radii[0];
			earthAtmGpu.radius_max = radii[rows - 1];
			earthAtmGpu.density_min = densities[0];
			earthAtmGpu.density_max = densities[rows - 1];
			earthAtmGpu.electronFraction_min = electronFractions[0];
			earthAtmGpu.electronFraction_max = electronFractions[rows - 1];

			cudaMalloc(&earthAtmGpu.radii, sizeof(double) * rows); CUERR;
			cudaMalloc(&earthAtmGpu.densities, sizeof(double) * rows); CUERR;
			cudaMalloc(&earthAtmGpu.electronFractions, sizeof(double) * rows); CUERR;

			cudaMemcpy(earthAtmGpu.radii, radii.data(), sizeof(double) * rows, H2D); CUERR;
			cudaMemcpy(earthAtmGpu.densities, densities.data(), sizeof(double) * rows, H2D); CUERR;
			cudaMemcpy(earthAtmGpu.electronFractions, electronFractions.data(), sizeof(double) * rows, H2D); CUERR;

			Akima::init_akima_interpolation_gpu_data_from_gpu_arrays(earthAtmGpu.akimaDensity, earthAtmGpu.radii, earthAtmGpu.densities, rows);
			Akima::init_akima_interpolation_gpu_data_from_gpu_arrays(earthAtmGpu.akimaElectronFraction, earthAtmGpu.radii, earthAtmGpu.electronFractions, rows);

			return earthAtmGpu;
		}

		static EarthAtm make_body_gpu(int deviceId, const std::string& filename){
			return make_body_gpu(deviceId, 22.0, filename);
		}

		static void destroy_body_gpu(EarthAtm& earthAtmGpu){
			cudaSetDevice(earthAtmGpu.deviceId); CUERR;

			cudaFree(earthAtmGpu.radii); CUERR;
			cudaFree(earthAtmGpu.densities); CUERR;
			cudaFree(earthAtmGpu.electronFractions); CUERR;

			destroy_akima_interpolation_gpu_data(earthAtmGpu.akimaDensity);
			destroy_akima_interpolation_gpu_data(earthAtmGpu.akimaElectronFraction);
		}
	};




	struct SunASnu{

		struct Track{
			double x_cur; //natural units
			double x_begin; //natural units
			double x_end; //natural units

			double b_impact;
			double radius_nu; //natural units

			mutable size_t intercacheD = 0;
			mutable size_t intercacheH = 0;

			HOSTDEVICEQUALIFIER
			Track() : Track(double(0.0)){}

			HOSTDEVICEQUALIFIER
			Track(double b_impact_) : b_impact(b_impact_), radius_nu(694439.0*Const::km()){
				x_end = 2.0*sqrt(radius_nu * radius_nu - b_impact * b_impact);
			}

			HOSTDEVICEQUALIFIER
			double getXBegin() const{ return x_begin;}

			HOSTDEVICEQUALIFIER
			double getXEnd() const{ return x_end;}

			HOSTDEVICEQUALIFIER
			double getCurrentX() const{ return x_cur;}

			HOSTDEVICEQUALIFIER
			void setCurrentX(double x){x_cur = x;}
		};

		int deviceId;

		double* radii = nullptr;
		double* densities = nullptr;
		double* hydrogenfractions = nullptr;
		size_t entries = 0;

		double sun_radius = 694439.0*Const::km(); //natural units

		double radius_min = 0.0;
		double radius_max = 0.0;
		double density_min = 0.0;
		double density_max = 0.0;
		double hydrogenFraction_min = 0.0;
		double hydrogenFraction_max = 0.0;

		Akima::AkimaInterpolationData akimaDensity;
		Akima::AkimaInterpolationData akimahydrogenFraction;

		HOSTDEVICEQUALIFIER
		SunASnu() : sun_radius(694439.0*Const::km()) {}

		HOSTDEVICEQUALIFIER
		double rdensity(double x, size_t* cache) const{
			// x is adimentional radius : x = 0 : center, x = 1 : radius
			if (x < radii[0]){
				return densities[0];
			} else if ( x > radii[entries - 1]){
				return 0;
			} else {
				return akimaDensity.interpolation(x, cache);
			}
		}

		HOSTDEVICEQUALIFIER
		double rxh(double x, size_t* cache) const{
			// x is adimentional radius : x = 0 : center, x = 1 : radius
			if (x < radii[0]){
				return hydrogenfractions[0];
			} else if ( x > radii[entries-1]){
				return 0;
			} else {
				return akimahydrogenFraction.interpolation(x, cache);
			}
		}

		HOSTDEVICEQUALIFIER
		double getDensity(const Track& track) const{
				const double x = track.getCurrentX();
				const double b = track.b_impact;
				const double r = sqrt(sun_radius * sun_radius + x * x - 2.0 * x * sqrt(sun_radius * sun_radius - b * b))/sun_radius;

				return rdensity(r, &(track.intercacheD));
		}

		HOSTDEVICEQUALIFIER
		double getYe(const Track& track) const{
			const double x = track.getCurrentX();
			const double b = track.b_impact;
			const double r = sqrt(sun_radius * sun_radius + x * x - 2.0 * x * sqrt(sun_radius * sun_radius - b * b))/sun_radius;

			return 0.5*(1.0+rxh(r, &(track.intercacheH)));
		}

		HOSTDEVICEQUALIFIER
		bool isConstantDensity() const{
			return false;
		}

        /*
            Expects file to be a table with at least 7 columns.
            Second column contains radii, fourth column contains densities, seventh column contains hydrogenfractions.
            Like nuSQuIDS'/data/astro/bs05_agsop.dat
        */
		static SunASnu make_body_gpu(int deviceId, const std::string& filename){
			cudaSetDevice(deviceId); CUERR;

			SunASnu bodyGpu;
			bodyGpu.deviceId = deviceId;

			nusquids::marray<double,2> sunasnu_model = nusquids::quickread(filename);
			size_t rows = sunasnu_model.extent(0);

			std::vector<double> radii(rows);
			std::vector<double> densities(rows);
			std::vector<double> hydrogenfractions(rows);

			for(size_t row = 0; row < rows; row++){
				radii[row] = sunasnu_model[row][1];
				densities[row] = sunasnu_model[row][3];
				hydrogenfractions[row] = sunasnu_model[row][6];
			}

			bodyGpu.entries = rows;
			bodyGpu.radius_min = radii[0];
			bodyGpu.radius_max = radii[rows - 1];
			bodyGpu.density_min = densities[0];
			bodyGpu.density_max = densities[rows - 1];
			bodyGpu.hydrogenFraction_min = hydrogenfractions[0];
			bodyGpu.hydrogenFraction_max = hydrogenfractions[rows - 1];

			cudaMalloc(&bodyGpu.radii, sizeof(double) * rows); CUERR;
			cudaMalloc(&bodyGpu.densities, sizeof(double) * rows); CUERR;
			cudaMalloc(&bodyGpu.hydrogenfractions, sizeof(double) * rows); CUERR;

			cudaMemcpy(bodyGpu.radii, radii.data(), sizeof(double) * rows, H2D); CUERR;
			cudaMemcpy(bodyGpu.densities, densities.data(), sizeof(double) * rows, H2D); CUERR;
			cudaMemcpy(bodyGpu.hydrogenfractions, hydrogenfractions.data(), sizeof(double) * rows, H2D); CUERR;

			Akima::init_akima_interpolation_gpu_data_from_gpu_arrays(bodyGpu.akimaDensity, bodyGpu.radii, bodyGpu.densities, rows);
			Akima::init_akima_interpolation_gpu_data_from_gpu_arrays(bodyGpu.akimahydrogenFraction, bodyGpu.radii, bodyGpu.hydrogenfractions, rows);

			return bodyGpu;
	    }

		static void destroy_body_gpu(SunASnu& bodyGpu){
			cudaSetDevice(bodyGpu.deviceId); CUERR;

			cudaFree(bodyGpu.radii); CUERR;
			cudaFree(bodyGpu.densities); CUERR;
			cudaFree(bodyGpu.hydrogenfractions); CUERR;

			destroy_akima_interpolation_gpu_data(bodyGpu.akimaDensity);
			destroy_akima_interpolation_gpu_data(bodyGpu.akimahydrogenFraction);
		}
	};


}




#endif
