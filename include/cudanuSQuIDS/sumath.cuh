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

#ifndef CUDANUSQUIDS_SUMATH_CUH
#define CUDANUSQUIDS_SUMATH_CUH

#include <cudanuSQuIDS/cudahelpers.cuh>

namespace cudanusquids{

namespace sumath{

	//constexpr square root of integers
	constexpr unsigned int ct_sqrt(unsigned int n, unsigned int i = 1){
	    return n == i ? n : (i * i < n ? ct_sqrt(n, i + 1) : i);
	}

	//constexpr find that i for which i * (i - 1) == n
	constexpr unsigned int findEvolBufSUN(unsigned int n, unsigned int i = 1){
	    return (i * (i - 1) == n ? i : findEvolBufSUN(n, i + 1));
	}

    template <unsigned int N>
    DEVICEQUALIFIER
    double sutrace(double (&left)[N], double (&right)[N]){

	constexpr unsigned int SUN = ct_sqrt(N);

        double trace = SUN * left[0] * right[0];

        UNROLLQUALIFIER
        for(int i = 1; i < SUN * SUN; i++){
            trace += 2 * left[i] * right[i];
        }

        return trace;
    }


    // ################## prepare projector evolution

    template <unsigned int N>
    DEVICEQUALIFIER
    void prepareEvolution(double (&evolbuf)[N], const double time, const double* const h0, const size_t h0offset){

	       constexpr unsigned int SUN = findEvolBufSUN(N);

        static_assert(2 <= SUN && SUN <=  4, "prepareEvolution requires 2 <= SUN <= 4");

        switch(SUN){
        case 2:{
            const double h03 = h0[3 * h0offset];

            sincos(2 * time * h03, &evolbuf[0], &evolbuf[1]);

            break;
            }
        case 3:{
            const double h04 = h0[4 * h0offset];
            const double h08 = h0[8 * h0offset];

            sincos(2.*time*h04, &evolbuf[0], &evolbuf[3]);
            sincos(time*(h04 + sqrt((double)3)*h08), &evolbuf[1], &evolbuf[4]);
            sincos(time*(h04 - sqrt((double)3)*h08), &evolbuf[2], &evolbuf[5]);

            break;
            }
        case 4:{
            const double h05 = h0[5 * h0offset];
            const double h010 = h0[10 * h0offset];
            const double h015 = h0[15 * h0offset];

            sincos(2*time*h05, &evolbuf[0], &evolbuf[6]);
            sincos(time*(h05 + sqrt((double)3)*h010), &evolbuf[1], &evolbuf[7]);
            sincos(time*h05 + (time*(h010 + 2*sqrt((double)2)*h015))/sqrt((double)3), &evolbuf[2], &evolbuf[8]);
            sincos(time*(h05 - sqrt((double)3)*h010), &evolbuf[3], &evolbuf[9]);
            sincos(time*h05 - (time*(h010 + 2*sqrt((double)2)*h015))/sqrt((double)3), &evolbuf[4], &evolbuf[10]);
            sincos((2*time*(h010 - sqrt((double)2)*h015))/sqrt((double)3), &evolbuf[5], &evolbuf[11]);

            break;
            }
        }
    }


    // ################## evolve projector

    template <unsigned int N, unsigned int M>
    DEVICEQUALIFIER
    void evolve(double (&evolproj)[N], double (&proj)[N], double (&evolbuf)[M]){

    	constexpr unsigned int SUN = ct_sqrt(N);
    	constexpr unsigned int SUN2 = findEvolBufSUN(M);

    	static_assert(SUN == SUN2, "evolve: evolbuf dim and projector dim do not match");
        static_assert(2 <= SUN && SUN <=  4, "evolve requires 2 <= SUN <= 4");

        switch(SUN){
        case 2:
            evolproj[0] = proj[0];
            evolproj[1] = evolbuf[1]*proj[1] + proj[2]*evolbuf[0];
            evolproj[2] = evolbuf[1]*proj[2] - proj[1]*evolbuf[0];
            evolproj[3] = proj[3];
            break;
        case 3:
            evolproj[0] = proj[0];
            evolproj[4] = proj[4];
            evolproj[8] = proj[8];
            evolproj[1] = evolbuf[3]*proj[1] + evolbuf[0]*proj[3];
            evolproj[3] = -(evolbuf[0]*proj[1]) + evolbuf[3]*proj[3];
            evolproj[2] = evolbuf[4]*proj[2] + evolbuf[1]*proj[6];
            evolproj[6] = -(evolbuf[1]*proj[2]) + evolbuf[4]*proj[6];
            evolproj[7] = evolbuf[2]*proj[5] + evolbuf[5]*proj[7];
            evolproj[5] = evolbuf[5]*proj[5] - evolbuf[2]*proj[7];
            break;
        case 4:
            evolproj[0] = proj[0];
            evolproj[1] = evolbuf[6]*proj[1] + proj[4]*evolbuf[0];
            evolproj[2] = evolbuf[7]*proj[2] + proj[8]*evolbuf[1];
            evolproj[3] = evolbuf[8]*proj[3] + proj[12]*evolbuf[2];
            evolproj[4] = evolbuf[6]*proj[4] - proj[1]*evolbuf[0];
            evolproj[5] = proj[5];
            evolproj[6] = evolbuf[9]*proj[6] - proj[9]*evolbuf[3];
            evolproj[7] = evolbuf[10]*proj[7] - proj[13]*evolbuf[4];
            evolproj[8] = evolbuf[7]*proj[8] - proj[2]*evolbuf[1];
            evolproj[9] = evolbuf[9]*proj[9] + proj[6]*evolbuf[3];
            evolproj[10] = proj[10];
            evolproj[11] = evolbuf[11]*proj[11] - proj[14]*evolbuf[5];
            evolproj[12] = evolbuf[8]*proj[12] - proj[3]*evolbuf[2];
            evolproj[13] = evolbuf[10]*proj[13] + proj[7]*evolbuf[4];
            evolproj[14] = evolbuf[11]*proj[14] + proj[11]*evolbuf[5];
            evolproj[15] = proj[15];
            break;
        }
    }

    template <unsigned int N>
    DEVICEQUALIFIER
    void evolve(double (&evolved)[N], double (&op)[N], const double time, const double* const h0, const size_t h0offset){

	constexpr unsigned int SUN = ct_sqrt(N);

        static_assert(2 <= SUN && SUN <=  4, "evolve requires 2 <= SUN <= 4");

        switch(SUN){
        case 2:{
            const double h03 = h0[3 * h0offset];

            double tmp0, tmp1;
            sincos(2.0 * time * h03, &tmp0, &tmp1);

            evolved[0] = op[0];
            evolved[1] = tmp1*op[1] + op[2]*tmp0;
            evolved[2] = tmp1*op[2] - op[1]*tmp0;
            evolved[3] = op[3];
            break;
            }
        case 3:{
            const double h04 = h0[4 * h0offset];
            const double h08 = h0[8 * h0offset];

            double tmpa0, tmpa1, tmpa2, tmpa3, tmpa4, tmpa5;

            sincos(2.0 * time * h04, &tmpa0, &tmpa3);
            sincos(time * (h04 + sqrt(3.0) * h08), &tmpa1, &tmpa4);
            sincos(time * (h04 - sqrt(3.0) * h08), &tmpa2, &tmpa5);

            evolved[0] = op[0];
            evolved[4] = op[4];
            evolved[8] = op[8];
            evolved[1] = tmpa3*op[1] + tmpa0*op[3];
            evolved[3] = -(tmpa0*op[1]) + tmpa3*op[3];
            evolved[2] = tmpa4*op[2] + tmpa1*op[6];
            evolved[6] = -(tmpa1*op[2]) + tmpa4*op[6];
            evolved[7] = tmpa2*op[5] + tmpa5*op[7];
            evolved[5] = tmpa5*op[5] - tmpa2*op[7];
            break;
            }
        case 4:{
            const double h05 = h0[5 * h0offset];
            const double h010 = h0[10 * h0offset];
            const double h015 = h0[15 * h0offset];

            double tmpb0, tmpb1, tmpb2, tmpb3, tmpb4, tmpb5, tmpb6, tmpb7, tmpb8, tmpb9, tmpb10, tmpb11;

            sincos(2.0 * time * h05, &tmpb0, &tmpb6);
            sincos(time * (h05 + sqrt(3.0) * h010), &tmpb1, &tmpb7);
            sincos(time * h05 + (time * (h010 + 2.0 * sqrt(2.0) * h015))/sqrt(3.0), &tmpb2, &tmpb8);
            sincos(time * (h05 - sqrt(3.0) * h010), &tmpb3, &tmpb9);
            sincos(time * h05 - (time * (h010 + 2.0 * sqrt(2.0) * h015))/sqrt(3.0), &tmpb4, &tmpb10);
            sincos((2.0 * time * (h010 - sqrt(2.0) * h015))/sqrt(3.0), &tmpb5, &tmpb11);

            evolved[0] = op[0];
            evolved[1] = tmpb6*op[1] + op[4]*tmpb0;
            evolved[2] = tmpb7*op[2] + op[8]*tmpb1;
            evolved[3] = tmpb8*op[3] + op[12]*tmpb2;
            evolved[4] = tmpb6*op[4] - op[1]*tmpb0;
            evolved[5] = op[5];
            evolved[6] = tmpb9*op[6] - op[9]*tmpb3;
            evolved[7] = tmpb10*op[7] - op[13]*tmpb4;
            evolved[8] = tmpb7*op[8] - op[2]*tmpb1;
            evolved[9] = tmpb9*op[9] + op[6]*tmpb3;
            evolved[10] = op[10];
            evolved[11] = tmpb11*op[11] - op[14]*tmpb5;
            evolved[12] = tmpb8*op[12] - op[3]*tmpb2;
            evolved[13] = tmpb10*op[13] + op[7]*tmpb4;
            evolved[14] = tmpb11*op[14] + op[11]*tmpb5;
            evolved[15] = op[15];
            break;
            }
        }
    }



    // ################### iCommutator

    template <unsigned int N>
    DEVICEQUALIFIER
    void iCommutator(double (&result)[N], double (&left)[N], double (&right)[N]){
	constexpr unsigned int SUN = ct_sqrt(N);

        static_assert(2 <= SUN && SUN <=  4, "iCommutator requires 2 <= SUN <= 4");

        switch(SUN){
        case 2:
            result[0] = 0;
            result[1] = 2*left[3]*right[2] - 2*left[2]*right[3];
            result[2] = -2*left[3]*right[1] + 2*left[1]*right[3];
            result[3] = 2*left[2]*right[1] - 2*left[1]*right[2];
            break;
        case 3:
            result[0] = 0;
            result[1] = left[7]*right[2] + 2.*left[4]*right[3] - 2.*left[3]*right[4] + left[6]*right[5] - left[5]*right[6] - left[2]*right[7];
            result[2] = -(left[7]*right[1]) - left[5]*right[3] + left[3]*right[5] + (left[4] + sqrt((double)3)*left[8])*right[6] + left[1]*right[7] - left[6]*(right[4] + sqrt((double)3)*right[8]);
            result[3] = -2.*left[4]*right[1] + left[5]*right[2] + 2*left[1]*right[4] - left[2]*right[5] + left[7]*right[6] - left[6]*right[7];
            result[4] = 2.*left[3]*right[1] + left[6]*right[2] - 2.*left[1]*right[3] - left[7]*right[5] - left[2]*right[6] + left[5]*right[7];
            result[5] = -(left[6]*right[1]) - left[3]*right[2] + left[2]*right[3] + left[1]*right[6] + (-left[4] + sqrt((double)3)*left[8])*right[7] + left[7]*(right[4] - sqrt((double)3)*right[8]);
            result[6] = left[5]*right[1] - (left[4] + sqrt((double)3)*left[8])*right[2] - left[7]*right[3] - left[1]*right[5] + left[3]*right[7] + left[2]*(right[4] + sqrt((double)3)*right[8]);
            result[7] = left[2]*right[1] - left[1]*right[2] + left[6]*right[3] + (left[4] - sqrt((double)3)*left[8])*right[5] - left[3]*right[6] + left[5]*(-right[4] + sqrt((double)3)*right[8]);
            result[8] = sqrt((double)3)*(left[6]*right[2] + left[7]*right[5] - left[2]*right[6] - left[5]*right[7]);
            break;
        case 4:
            result[0] = 0;
            result[1] = left[9]*right[2] + left[13]*right[3] + 2*left[5]*right[4] - 2*left[4]*right[5] + left[8]*right[6] + left[12]*right[7] - left[6]*right[8] - left[2]*right[9] - left[7]*right[12] - left[3]*right[13];
            result[2] = -(left[9]*right[1]) + left[14]*right[3] - left[6]*right[4] + left[4]*right[6] + (left[5] + sqrt((double)3)*left[10])*right[8] + left[1]*right[9] - left[8]*(right[5] + sqrt((double)3)*right[10]) + left[12]*right[11] - left[11]*right[12] - left[3]*right[14];
            result[3] = -(left[13]*right[1]) - left[14]*right[2] - left[7]*right[4] + left[4]*right[7] - left[11]*right[8] + left[8]*right[11] + ((3*left[5] + sqrt((double)3)*left[10] + 2*sqrt((double)6)*left[15])*right[12])/3. + left[1]*right[13] + left[2]*right[14] - (left[12]*(3*right[5] + sqrt((double)3)*right[10] + 2*sqrt((double)6)*right[15]))/3.;
            result[4] = -2*left[5]*right[1] + left[6]*right[2] + left[7]*right[3] + 2*left[1]*right[5] - left[2]*right[6] - left[3]*right[7] + left[9]*right[8] - left[8]*right[9] + left[13]*right[12] - left[12]*right[13];
            result[5] = 2*left[4]*right[1] + left[8]*right[2] + left[12]*right[3] - 2*left[1]*right[4] - left[9]*right[6] - left[13]*right[7] - left[2]*right[8] + left[6]*right[9] - left[3]*right[12] + left[7]*right[13];
            result[6] = -(left[8]*right[1]) - left[4]*right[2] + left[2]*right[4] + left[14]*right[7] + left[1]*right[8] + (-left[5] + sqrt((double)3)*left[10])*right[9] + left[9]*(right[5] - sqrt((double)3)*right[10]) + left[13]*right[11] - left[11]*right[13] - left[7]*right[14];
            result[7] = -(left[12]*right[1]) - left[4]*right[3] + left[3]*right[4] - left[14]*right[6] - left[11]*right[9] + left[9]*right[11] + left[1]*right[12] + ((-3*left[5] + sqrt((double)3)*left[10] + 2*sqrt((double)6)*left[15])*right[13])/3. + left[6]*right[14] + left[13]*(right[5] - (right[10] + 2*sqrt((double)2)*right[15])/sqrt((double)3));
            result[8] = left[6]*right[1] - (left[5] + sqrt((double)3)*left[10])*right[2] + left[11]*right[3] - left[9]*right[4] - left[1]*right[6] + left[4]*right[9] + left[2]*(right[5] + sqrt((double)3)*right[10]) - left[3]*right[11] + left[14]*right[12] - left[12]*right[14];
            result[9] = left[2]*right[1] - left[1]*right[2] + left[8]*right[4] + (left[5] - sqrt((double)3)*left[10])*right[6] + left[11]*right[7] - left[4]*right[8] + left[6]*(-right[5] + sqrt((double)3)*right[10]) - left[7]*right[11] + left[14]*right[13] - left[13]*right[14];
            result[10] = (3*left[8]*right[2] + left[12]*right[3] + 3*left[9]*right[6] + left[13]*right[7] - 3*left[2]*right[8] - 3*left[6]*right[9] - 2*left[14]*right[11] - left[3]*right[12] - left[7]*right[13] + 2*left[11]*right[14])/sqrt((double)3);
            result[11] = -(left[12]*right[2]) - left[8]*right[3] - left[13]*right[6] - left[9]*right[7] + left[3]*right[8] + left[7]*right[9] + (3*left[2]*right[12] + 3*left[6]*right[13] - 2*sqrt((double)3)*(left[10] - sqrt((double)2)*left[15])*right[14] + 2*sqrt((double)3)*left[14]*(right[10] - sqrt((double)2)*right[15]))/3.;
            result[12] = left[7]*right[1] + (3*left[11]*right[2] - (3*left[5] + sqrt((double)3)*left[10] + 2*sqrt((double)6)*left[15])*right[3] - 3*left[13]*right[4] - 3*(left[1]*right[7] + left[14]*right[8] + left[2]*right[11] - left[4]*right[13] - left[8]*right[14]) + left[3]*(3*right[5] + sqrt((double)3)*right[10] + 2*sqrt((double)6)*right[15]))/3.;
            result[13] = left[3]*right[1] - left[1]*right[3] + left[12]*right[4] + left[11]*right[6] + left[5]*right[7] - ((left[10] + 2*sqrt((double)2)*left[15])*right[7])/sqrt((double)3) - left[14]*right[9] - left[6]*right[11] - left[4]*right[12] + left[9]*right[14] + left[7]*(-right[5] + (right[10] + 2*sqrt((double)2)*right[15])/sqrt((double)3));
            result[14] = left[3]*right[2] - left[2]*right[3] + left[7]*right[6] + (-3*left[6]*right[7] + 3*left[12]*right[8] + 3*left[13]*right[9] + 2*sqrt((double)3)*(left[10] - sqrt((double)2)*left[15])*right[11] - 3*(left[8]*right[12] + left[9]*right[13]) - 2*sqrt((double)3)*left[11]*(right[10] - sqrt((double)2)*right[15]))/3.;
            result[15] = 2*sqrt((double)0.6666666666666666)*(left[12]*right[3] + left[13]*right[7] + left[14]*right[11] - left[3]*right[12] - left[7]*right[13] - left[11]*right[14]);
            break;
        }
    }


    // #################### anticommutator


    template <unsigned int N>
    DEVICEQUALIFIER
    void anticommutator(double (&result)[N], double (&left)[N], double (&right)[N]){
	constexpr unsigned int SUN = ct_sqrt(N);
	static_assert(2 <= SUN && SUN <=  4, "anticommutator requires 2 <= SUN <= 4");

        switch(SUN){
        case 2:
            result[0] = 2*(left[0]*right[0] + left[1]*right[1] + left[2]*right[2] + left[3]*right[3]);
            result[1] = 2*(left[1]*right[0] + left[0]*right[1]);
            result[2] = 2*(left[2]*right[0] + left[0]*right[2]);
            result[3] = 2*(left[3]*right[0] + left[0]*right[3]);
            break;
        case 3:
            result[0] = (2*(3*left[0]*right[0] + 2*(left[1]*right[1] + left[2]*right[2] + left[3]*right[3] + left[4]*right[4] + left[5]*right[5] + left[6]*right[6] + left[7]*right[7] + left[8]*right[8])))/3.;
            result[1] = 2*left[0]*right[1] + (2*left[8]*right[1])/sqrt((double)3) + left[5]*right[2] + left[2]*right[5] + left[7]*right[6] + left[6]*right[7] + left[1]*(2*right[0] + (2*right[8])/sqrt((double)3));
            result[2] = left[5]*right[1] + ((6*left[0] + 3*left[4] - sqrt((double)3)*left[8])*right[2])/3. - left[7]*right[3] + left[1]*right[5] - left[3]*right[7] + left[2]*(2*right[0] + right[4] - right[8]/sqrt((double)3));
            result[3] = -(left[7]*right[2]) + 2*left[0]*right[3] + (2*left[8]*right[3])/sqrt((double)3) + left[6]*right[5] + left[5]*right[6] - left[2]*right[7] + left[3]*(2*right[0] + (2*right[8])/sqrt((double)3));
            result[4] = left[2]*right[2] + 2*left[0]*right[4] + (2*left[8]*right[4])/sqrt((double)3) - left[5]*right[5] + left[6]*right[6] - left[7]*right[7] + left[4]*(2*right[0] + (2*right[8])/sqrt((double)3));
            result[5] = left[2]*right[1] + left[1]*right[2] + left[6]*right[3] + ((6*left[0] - 3*left[4] - sqrt((double)3)*left[8])*right[5])/3. + left[3]*right[6] + left[5]*(2*right[0] - right[4] - right[8]/sqrt((double)3));
            result[6] = left[7]*right[1] + left[5]*right[3] + left[3]*right[5] + ((6*left[0] + 3*left[4] - sqrt((double)3)*left[8])*right[6])/3. + left[1]*right[7] + left[6]*(2*right[0] + right[4] - right[8]/sqrt((double)3));
            result[7] = left[6]*right[1] - left[3]*right[2] - left[2]*right[3] + left[1]*right[6] + ((6*left[0] - 3*left[4] - sqrt((double)3)*left[8])*right[7])/3. + left[7]*(2*right[0] - right[4] - right[8]/sqrt((double)3));
            result[8] = (2*left[1]*right[1] - left[2]*right[2] + 2*left[3]*right[3] + 2*left[4]*right[4] - left[5]*right[5] - left[6]*right[6] - left[7]*right[7] + 2*left[8]*(sqrt((double)3)*right[0] - right[8]) + 2*sqrt((double)3)*left[0]*right[8])/sqrt((double)3);
            break;
        case 4:
            result[0] = 2*left[0]*right[0] + left[1]*right[1] + left[2]*right[2] + left[3]*right[3] + left[4]*right[4] + left[5]*right[5] + left[6]*right[6] + left[7]*right[7] + left[8]*right[8] + left[9]*right[9] + left[10]*right[10] + left[11]*right[11] + left[12]*right[12] + left[13]*right[13] + left[14]*right[14] + left[15]*right[15];
            result[1] = ((6*left[0] + 2*sqrt((double)3)*left[10] + sqrt((double)6)*left[15])*right[1])/3. + left[6]*right[2] + left[7]*right[3] + left[2]*right[6] + left[3]*right[7] + left[9]*right[8] + left[8]*right[9] + left[13]*right[12] + left[12]*right[13] + (left[1]*(6*right[0] + 2*sqrt((double)3)*right[10] + sqrt((double)6)*right[15]))/3.;
            result[2] = left[6]*right[1] + ((6*left[0] + 3*left[5] - sqrt((double)3)*left[10] + sqrt((double)6)*left[15])*right[2])/3. + left[11]*right[3] - left[9]*right[4] + left[1]*right[6] - left[4]*right[9] + left[3]*right[11] + left[14]*right[12] + left[12]*right[14] + left[2]*(2*right[0] + right[5] - right[10]/sqrt((double)3) + sqrt((double)0.6666666666666666)*right[15]);
            result[3] = left[7]*right[1] + left[11]*right[2] + ((6*left[0] + 3*left[5] + sqrt((double)3)*left[10] - sqrt((double)6)*left[15])*right[3])/3. - left[13]*right[4] + left[1]*right[7] - left[14]*right[8] + left[2]*right[11] - left[4]*right[13] - left[8]*right[14] + left[3]*(2*right[0] + right[5] + (right[10] - sqrt((double)2)*right[15])/sqrt((double)3));
            result[4] = -(left[9]*right[2]) - left[13]*right[3] + ((6*left[0] + 2*sqrt((double)3)*left[10] + sqrt((double)6)*left[15])*right[4])/3. + left[8]*right[6] + left[12]*right[7] + left[6]*right[8] - left[2]*right[9] + left[7]*right[12] - left[3]*right[13] + (left[4]*(6*right[0] + 2*sqrt((double)3)*right[10] + sqrt((double)6)*right[15]))/3.;
            result[5] = left[2]*right[2] + left[3]*right[3] + ((6*left[0] + 2*sqrt((double)3)*left[10] + sqrt((double)6)*left[15])*right[5])/3. - left[6]*right[6] - left[7]*right[7] + left[8]*right[8] - left[9]*right[9] + left[12]*right[12] - left[13]*right[13] + (left[5]*(6*right[0] + 2*sqrt((double)3)*right[10] + sqrt((double)6)*right[15]))/3.;
            result[6] = left[2]*right[1] + left[1]*right[2] + left[8]*right[4] + ((6*left[0] - 3*left[5] - sqrt((double)3)*left[10] + sqrt((double)6)*left[15])*right[6])/3. + left[11]*right[7] + left[4]*right[8] + left[7]*right[11] + left[14]*right[13] + left[13]*right[14] + (left[6]*(6*right[0] - 3*right[5] - sqrt((double)3)*right[10] + sqrt((double)6)*right[15]))/3.;
            result[7] = left[3]*right[1] + left[1]*right[3] + left[12]*right[4] + left[11]*right[6] + ((6*left[0] - 3*left[5] + sqrt((double)3)*left[10] - sqrt((double)6)*left[15])*right[7])/3. - left[14]*right[9] + left[6]*right[11] + left[4]*right[12] - left[9]*right[14] + (left[7]*(6*right[0] - 3*right[5] + sqrt((double)3)*right[10] - sqrt((double)6)*right[15]))/3.;
            result[8] = left[9]*right[1] - left[14]*right[3] + left[6]*right[4] + left[4]*right[6] + ((6*left[0] + 3*left[5] - sqrt((double)3)*left[10] + sqrt((double)6)*left[15])*right[8])/3. + left[1]*right[9] + left[12]*right[11] + left[11]*right[12] - left[3]*right[14] + left[8]*(2*right[0] + right[5] - right[10]/sqrt((double)3) + sqrt((double)0.6666666666666666)*right[15]);
            result[9] = left[8]*right[1] - left[4]*right[2] - left[2]*right[4] - left[14]*right[7] + left[1]*right[8] + ((6*left[0] - 3*left[5] - sqrt((double)3)*left[10] + sqrt((double)6)*left[15])*right[9])/3. + left[13]*right[11] + left[11]*right[13] - left[7]*right[14] + (left[9]*(6*right[0] - 3*right[5] - sqrt((double)3)*right[10] + sqrt((double)6)*right[15]))/3.;
            result[10] = (2*left[1]*right[1] - left[2]*right[2] + left[3]*right[3] + 2*left[4]*right[4] + 2*left[5]*right[5] - left[6]*right[6] + left[7]*right[7] - left[8]*right[8] - left[9]*right[9] + (2*sqrt((double)3)*left[0] + sqrt((double)2)*left[15])*right[10] - 2*left[11]*right[11] + left[12]*right[12] + left[13]*right[13] - 2*left[14]*right[14] + left[10]*(2*sqrt((double)3)*right[0] - 2*right[10] + sqrt((double)2)*right[15]))/sqrt((double)3);
            result[11] = left[3]*right[2] + left[2]*right[3] + left[7]*right[6] + left[6]*right[7] + left[12]*right[8] + left[13]*right[9] + ((6*left[0] - 2*sqrt((double)3)*left[10] - sqrt((double)6)*left[15])*right[11])/3. + left[8]*right[12] + left[9]*right[13] + (left[11]*(6*right[0] - 2*sqrt((double)3)*right[10] - sqrt((double)6)*right[15]))/3.;
            result[12] = left[13]*right[1] + left[14]*right[2] + left[7]*right[4] + left[4]*right[7] + left[11]*right[8] + left[8]*right[11] + ((6*left[0] + 3*left[5] + sqrt((double)3)*left[10] - sqrt((double)6)*left[15])*right[12])/3. + left[1]*right[13] + left[2]*right[14] + left[12]*(2*right[0] + right[5] + (right[10] - sqrt((double)2)*right[15])/sqrt((double)3));
            result[13] = left[12]*right[1] - left[4]*right[3] - left[3]*right[4] + left[14]*right[6] + left[11]*right[9] + left[9]*right[11] + left[1]*right[12] + ((6*left[0] - 3*left[5] + sqrt((double)3)*left[10] - sqrt((double)6)*left[15])*right[13])/3. + left[6]*right[14] + (left[13]*(6*right[0] - 3*right[5] + sqrt((double)3)*right[10] - sqrt((double)6)*right[15]))/3.;
            result[14] = left[12]*right[2] - left[8]*right[3] + left[13]*right[6] - left[9]*right[7] - left[3]*right[8] - left[7]*right[9] + left[2]*right[12] + left[6]*right[13] + ((6*left[0] - 2*sqrt((double)3)*left[10] - sqrt((double)6)*left[15])*right[14])/3. + (left[14]*(6*right[0] - 2*sqrt((double)3)*right[10] - sqrt((double)6)*right[15]))/3.;
            result[15] = sqrt((double)0.6666666666666666)*(left[1]*right[1] + left[2]*right[2] - left[3]*right[3] + left[4]*right[4] + left[5]*right[5] + left[6]*right[6] - left[7]*right[7] + left[8]*right[8] + left[9]*right[9] + left[10]*right[10] - left[11]*right[11] - left[12]*right[12] - left[13]*right[13] - left[14]*right[14] + left[15]*(sqrt((double)6)*right[0] - 2*right[15]) + sqrt((double)6)*left[0]*right[15]);
            break;
        }
    }

	// ##################### make projector
	template<unsigned int N>
	HOSTDEVICEQUALIFIER
	void toProjector(double (&components)[N], unsigned int n){
		constexpr unsigned int SUN = ct_sqrt(N);
		static_assert(2 <= SUN && SUN <=  4, "rotate requires 2 <= SUN <= 4");

		assert(n < SUN);

		double m_real[SUN][SUN];
		double m_imag[SUN][SUN];

		auto kronecker = [](unsigned int i, unsigned int j){ return ( i==j ? 1 : 0 ); };

		for(unsigned int i=0; i<SUN; i++){
			for(unsigned int j=0; j<SUN; j++){
				m_real[i][j] = kronecker(i,j)*kronecker(i,n);
				m_imag[i][j] = 0.0;
			}
		}

		matrix_to_su(components, m_real, m_imag);
	}

	// ##################### SU to matrix
	template<unsigned int SUN, unsigned int N>
	HOSTDEVICEQUALIFIER
	void su_to_matrix(double (&re)[SUN][SUN], double (&im)[SUN][SUN], const double (&components)[N]){
		static_assert(2 <= SUN && SUN <=  4, "su_to_matrix requires 2 <= SUN <= 4");
		static_assert(SUN * SUN == N, "su_to_matrix: array dimensions do not match");

		switch (SUN){
		case 2:
			re[0][0] = components[0] + components[3];
			re[0][1] = components[1];
			re[1][0] = components[1];
			re[1][1] = components[0] - components[3];
			im[0][0] = 0;
			im[0][1] = -components[2];
			im[1][0] = components[2];
			im[1][1] = 0;
			break;
		case 3:
			re[0][0] = components[0] + components[4] + components[8]/sqrt((double)3);
			re[0][1] = components[1];
			re[0][2] = components[2];
			re[1][0] = components[1];
			re[1][1] = components[0] - components[4] + components[8]/sqrt((double)3);
			re[1][2] = components[5];
			re[2][0] = components[2];
			re[2][1] = components[5];
			re[2][2] = components[0] - (2*components[8])/sqrt((double)3);
			im[0][0] = 0;
			im[0][1] = -components[3];
			im[0][2] = -components[6];
			im[1][0] = components[3];
			im[1][1] = 0;
			im[1][2] = -components[7];
			im[2][0] = components[6];
			im[2][1] = components[7];
			im[2][2] = 0;
			break;
		case 4:
			re[0][0] = components[0] + components[5] + components[10]/sqrt((double)3) + components[15]/sqrt((double)6);
			re[0][1] = components[1];
			re[0][2] = components[2];
			re[0][3] = components[3];
			re[1][0] = components[1];
			re[1][1] = components[0] - components[5] + components[10]/sqrt((double)3) + components[15]/sqrt((double)6);
			re[1][2] = components[6];
			re[1][3] = components[7];
			re[2][0] = components[2];
			re[2][1] = components[6];
			re[2][2] = components[0] - (2*components[10])/sqrt((double)3) + components[15]/sqrt((double)6);
			re[2][3] = components[11];
			re[3][0] = components[3];
			re[3][1] = components[7];
			re[3][2] = components[11];
			re[3][3] = components[0] - sqrt((double)1.5)*components[15];
			im[0][0] = 0;
			im[0][1] = -components[4];
			im[0][2] = -components[8];
			im[0][3] = -components[12];
			im[1][0] = components[4];
			im[1][1] = 0;
			im[1][2] = -components[9];
			im[1][3] = -components[13];
			im[2][0] = components[8];
			im[2][1] = components[9];
			im[2][2] = 0;
			im[2][3] = -components[14];
			im[3][0] = components[12];
			im[3][1] = components[13];
			im[3][2] = components[14];
			im[3][3] = 0;
			break;
		}
	}





	// ##################### matrix to SU


    template <unsigned int SUN, unsigned int N>
    HOSTDEVICEQUALIFIER
    void matrix_to_su(double (&components)[N], const double (&m_real)[SUN][SUN], const double (&m_imag)[SUN][SUN]){

	static_assert(2 <= SUN && SUN <=  4, "matrix_to_su requires 2 <= SUN <= 4");
	static_assert(SUN * SUN == N, "matrix_to_su: array dimensions do not match");

	for(unsigned int i = 0; i < N; i++)
		components[i] = 0;

	switch(SUN){
	case 2:
		components[0]+=0.5*m_real[0][0];
		components[3]+=0.5*m_real[0][0];
		components[1]+=0.5*m_real[0][1];
		components[2]+=-0.5*m_imag[0][1];
		components[1]+=0.5*m_real[1][0];
		components[2]+=0.5*m_imag[1][0];
		components[0]+=0.5*m_real[1][1];
		components[3]+=-0.5*m_real[1][1];
		break;
	case 3:
		components[0]+=0.3333333333333333*m_real[0][0];
		components[4]+=0.5*m_real[0][0];
		components[8]+=1/(2.*sqrt((double)3))*m_real[0][0];
		components[1]+=0.5*m_real[0][1];
		components[3]+=-0.5*m_imag[0][1];
		components[2]+=0.5*m_real[0][2];
		components[6]+=-0.5*m_imag[0][2];
		components[1]+=0.5*m_real[1][0];
		components[3]+=0.5*m_imag[1][0];
		components[0]+=0.3333333333333333*m_real[1][1];
		components[4]+=-0.5*m_real[1][1];
		components[8]+=1/(2.*sqrt((double)3))*m_real[1][1];
		components[5]+=0.5*m_real[1][2];
		components[7]+=-0.5*m_imag[1][2];
		components[2]+=0.5*m_real[2][0];
		components[6]+=0.5*m_imag[2][0];
		components[5]+=0.5*m_real[2][1];
		components[7]+=0.5*m_imag[2][1];
		components[0]+=0.3333333333333333*m_real[2][2];
		components[8]+=-(1/sqrt((double)3))*m_real[2][2];
		break;
	case 4:
		components[0]+=0.25*m_real[0][0];
		components[5]+=0.5*m_real[0][0];
		components[10]+=1/(2.*sqrt((double)3))*m_real[0][0];
		components[15]+=1/(2.*sqrt((double)6))*m_real[0][0];
		components[1]+=0.5*m_real[0][1];
		components[4]+=-0.5*m_imag[0][1];
		components[2]+=0.5*m_real[0][2];
		components[8]+=-0.5*m_imag[0][2];
		components[3]+=0.5*m_real[0][3];
		components[12]+=-0.5*m_imag[0][3];
		components[1]+=0.5*m_real[1][0];
		components[4]+=0.5*m_imag[1][0];
		components[0]+=0.25*m_real[1][1];
		components[5]+=-0.5*m_real[1][1];
		components[10]+=1/(2.*sqrt((double)3))*m_real[1][1];
		components[15]+=1/(2.*sqrt((double)6))*m_real[1][1];
		components[6]+=0.5*m_real[1][2];
		components[9]+=-0.5*m_imag[1][2];
		components[7]+=0.5*m_real[1][3];
		components[13]+=-0.5*m_imag[1][3];
		components[2]+=0.5*m_real[2][0];
		components[8]+=0.5*m_imag[2][0];
		components[6]+=0.5*m_real[2][1];
		components[9]+=0.5*m_imag[2][1];
		components[0]+=0.25*m_real[2][2];
		components[10]+=-(1/sqrt((double)3))*m_real[2][2];
		components[15]+=1/(2.*sqrt((double)6))*m_real[2][2];
		components[11]+=0.5*m_real[2][3];
		components[14]+=-0.5*m_imag[2][3];
		components[3]+=0.5*m_real[3][0];
		components[12]+=0.5*m_imag[3][0];
		components[7]+=0.5*m_real[3][1];
		components[13]+=0.5*m_imag[3][1];
		components[11]+=0.5*m_real[3][2];
		components[14]+=0.5*m_imag[3][2];
		components[0]+=0.25*m_real[3][3];
		components[15]+=-sqrt((double)1.5)/2.*m_real[3][3];
		break;
	}
    }


	//#############################  rotation

	template <unsigned int N>
	HOSTDEVICEQUALIFIER
	void rotate(int i, int j, double (&rotcomponents)[N], const double (&components)[N], double th, double del){

		constexpr unsigned int SUN = ct_sqrt(N);
		static_assert(2 <= SUN && SUN <=  4, "rotate requires 2 <= SUN <= 4");

		assert(i<SUN);
		assert(j<SUN);
		assert(i<j);
		assert(i >= 0);
		assert(j >= 0);

		for(int i = 0; i < N; i++)
			rotcomponents[i] = 0;

		switch (SUN){
		case 2:
			switch (i){
			case 0:
				switch (j){
				case 1:
					rotcomponents[0]+=components[0];
					rotcomponents[1]+=(cos(th)*cos(th) - cos(2*del)*sin(th)*sin(th))*components[1] + cos(del)*(-2*sin(del)*sin(th)*sin(th)*components[2] + sin(2*th)*components[3]);
					rotcomponents[2]+=-(sin(2*del)*sin(th)*sin(th)*components[1]) + (cos(th)*cos(th) + cos(2*del)*sin(th)*sin(th))*components[2] + sin(del)*sin(2*th)*components[3];
					rotcomponents[3]+=-(sin(2*th)*(cos(del)*components[1] + sin(del)*components[2])) + cos(2*th)*components[3];
					break;
				default:
					printf("SUN_rotation error. \n");
					break;
				}
				break;
			default:
				printf("SUN_rotation error. \n");
				break;
			}
			break;
		case 3:
			switch (i){
			case 0:
				switch (j){
				case 1:
					rotcomponents[0]+=components[0];
					rotcomponents[1]+=(cos(th)*cos(th) - cos(2*del)*sin(th)*sin(th))*components[1] + cos(del)*(-2*sin(del)*sin(th)*sin(th)*components[3] + sin(2*th)*components[4]);
					rotcomponents[2]+=cos(th)*components[2] + sin(th)*(-(cos(del)*components[5]) + sin(del)*components[7]);
					rotcomponents[3]+=-(sin(2*del)*sin(th)*sin(th)*components[1]) + (cos(th)*cos(th) + cos(2*del)*sin(th)*sin(th))*components[3] + sin(del)*sin(2*th)*components[4];
					rotcomponents[4]+=-(sin(2*th)*(cos(del)*components[1] + sin(del)*components[3])) + cos(2*th)*components[4];
					rotcomponents[5]+=cos(del)*sin(th)*components[2] + cos(th)*components[5] + sin(del)*sin(th)*components[6];
					rotcomponents[6]+=cos(th)*components[6] - sin(th)*(sin(del)*components[5] + cos(del)*components[7]);
					rotcomponents[7]+=-(sin(del)*sin(th)*components[2]) + cos(del)*sin(th)*components[6] + cos(th)*components[7];
					rotcomponents[8]+=components[8];
					break;
				case 2:
					rotcomponents[0]+=components[0];
					rotcomponents[1]+=cos(th)*components[1] - sin(th)*(cos(del)*components[5] + sin(del)*components[7]);
					rotcomponents[2]+=(cos(th)*cos(th) - cos(2*del)*sin(th)*sin(th))*components[2] + cos(del)*sin(th)*(-2*sin(del)*sin(th)*components[6] + cos(th)*(components[4] + sqrt((double)3)*components[8]));
					rotcomponents[3]+=cos(th)*components[3] + sin(th)*(-(sin(del)*components[5]) + cos(del)*components[7]);
					rotcomponents[4]+=(-2*cos(del)*sin(2*th)*components[2] + (3 + cos(2*th))*components[4] - 2*sin(th)*(2*cos(th)*sin(del)*components[6] + sqrt((double)3)*sin(th)*components[8]))/4.;
					rotcomponents[5]+=cos(del)*sin(th)*components[1] + sin(del)*sin(th)*components[3] + cos(th)*components[5];
					rotcomponents[6]+=-(sin(2*del)*sin(th)*sin(th)*components[2]) + (cos(th)*cos(th) + cos(2*del)*sin(th)*sin(th))*components[6] + cos(th)*sin(del)*sin(th)*(components[4] + sqrt((double)3)*components[8]);
					rotcomponents[7]+=sin(del)*sin(th)*components[1] - cos(del)*sin(th)*components[3] + cos(th)*components[7];
					rotcomponents[8]+=(-2*sqrt((double)3)*sin(th)*sin(th)*components[4] - 2*sqrt((double)3)*sin(2*th)*(cos(del)*components[2] + sin(del)*components[6]) + (1 + 3*cos(2*th))*components[8])/4.;
					break;
				default:
					printf("SUN_rotation error. \n");
				break;
				}
				break;
			case 1:
				switch (j){
				case 2:
					rotcomponents[0]+=components[0];
					rotcomponents[1]+=cos(th)*components[1] - sin(th)*(cos(del)*components[2] + sin(del)*components[6]);
					rotcomponents[2]+=cos(del)*sin(th)*components[1] + cos(th)*components[2] - sin(del)*sin(th)*components[3];
					rotcomponents[3]+=sin(del)*sin(th)*components[2] + cos(th)*components[3] - cos(del)*sin(th)*components[6];
					rotcomponents[4]+=((3 + cos(2*th))*components[4] + 2*sin(th)*(2*cos(th)*(cos(del)*components[5] + sin(del)*components[7]) + sqrt((double)3)*sin(th)*components[8]))/4.;
					rotcomponents[5]+=-(cos(del)*cos(th)*sin(th)*components[4]) + (cos(th)*cos(th) - cos(2*del)*sin(th)*sin(th))*components[5] + cos(del)*sin(th)*(-2*sin(del)*sin(th)*components[7] + sqrt((double)3)*cos(th)*components[8]);
					rotcomponents[6]+=sin(del)*sin(th)*components[1] + cos(del)*sin(th)*components[3] + cos(th)*components[6];
					rotcomponents[7]+=(cos(th)*cos(th) + cos(2*del)*sin(th)*sin(th))*components[7] - sin(th)*(sin(2*del)*sin(th)*components[5] + cos(th)*sin(del)*(components[4] - sqrt((double)3)*components[8]));
					rotcomponents[8]+=(6*sin(th)*sin(th)*components[4] - 6*sin(2*th)*(cos(del)*components[5] + sin(del)*components[7]) + sqrt((double)3)*(1 + 3*cos(2*th))*components[8])/(4.*sqrt((double)3));
					break;
				default:
					printf("SUN_rotation error. \n");
					break;
				}
				break;
			default:
				printf("SUN_rotation error. \n");
				break;
			}
			break;
		case 4:
			switch (i){
			case 0:
				switch (j){
				case 1:
					rotcomponents[0]+=components[0];
					rotcomponents[1]+=(cos(th)*cos(th) - cos(2*del)*sin(th)*sin(th))*components[1] + cos(del)*(-2*sin(del)*sin(th)*sin(th)*components[4] + sin(2*th)*components[5]);
					rotcomponents[2]+=cos(th)*components[2] + sin(th)*(-(cos(del)*components[6]) + sin(del)*components[9]);
					rotcomponents[3]+=cos(th)*components[3] + sin(th)*(-(cos(del)*components[7]) + sin(del)*components[13]);
					rotcomponents[4]+=-(sin(2*del)*sin(th)*sin(th)*components[1]) + (cos(th)*cos(th) + cos(2*del)*sin(th)*sin(th))*components[4] + sin(del)*sin(2*th)*components[5];
					rotcomponents[5]+=-(sin(2*th)*(cos(del)*components[1] + sin(del)*components[4])) + cos(2*th)*components[5];
					rotcomponents[6]+=cos(del)*sin(th)*components[2] + cos(th)*components[6] + sin(del)*sin(th)*components[8];
					rotcomponents[7]+=cos(del)*sin(th)*components[3] + cos(th)*components[7] + sin(del)*sin(th)*components[12];
					rotcomponents[8]+=cos(th)*components[8] - sin(th)*(sin(del)*components[6] + cos(del)*components[9]);
					rotcomponents[9]+=-(sin(del)*sin(th)*components[2]) + cos(del)*sin(th)*components[8] + cos(th)*components[9];
					rotcomponents[10]+=components[10];
					rotcomponents[11]+=components[11];
					rotcomponents[12]+=cos(th)*components[12] - sin(th)*(sin(del)*components[7] + cos(del)*components[13]);
					rotcomponents[13]+=-(sin(del)*sin(th)*components[3]) + cos(del)*sin(th)*components[12] + cos(th)*components[13];
					rotcomponents[14]+=components[14];
					rotcomponents[15]+=components[15];
					break;
				case 2:
					rotcomponents[0]+=components[0];
					rotcomponents[1]+=cos(th)*components[1] - sin(th)*(cos(del)*components[6] + sin(del)*components[9]);
					rotcomponents[2]+=(cos(th)*cos(th) - cos(2*del)*sin(th)*sin(th))*components[2] + cos(del)*sin(th)*(-2*sin(del)*sin(th)*components[8] + cos(th)*(components[5] + sqrt((double)3)*components[10]));
					rotcomponents[3]+=cos(th)*components[3] + sin(th)*(-(cos(del)*components[11]) + sin(del)*components[14]);
					rotcomponents[4]+=cos(th)*components[4] + sin(th)*(-(sin(del)*components[6]) + cos(del)*components[9]);
					rotcomponents[5]+=(-2*cos(del)*sin(2*th)*components[2] + (3 + cos(2*th))*components[5] - 2*sin(th)*(2*cos(th)*sin(del)*components[8] + sqrt((double)3)*sin(th)*components[10]))/4.;
					rotcomponents[6]+=cos(del)*sin(th)*components[1] + sin(del)*sin(th)*components[4] + cos(th)*components[6];
					rotcomponents[7]+=components[7];
					rotcomponents[8]+=-(sin(2*del)*sin(th)*sin(th)*components[2]) + (cos(th)*cos(th) + cos(2*del)*sin(th)*sin(th))*components[8] + cos(th)*sin(del)*sin(th)*(components[5] + sqrt((double)3)*components[10]);
					rotcomponents[9]+=sin(del)*sin(th)*components[1] - cos(del)*sin(th)*components[4] + cos(th)*components[9];
					rotcomponents[10]+=(-2*sqrt((double)3)*sin(th)*sin(th)*components[5] - 2*sqrt((double)3)*sin(2*th)*(cos(del)*components[2] + sin(del)*components[8]) + (1 + 3*cos(2*th))*components[10])/4.;
					rotcomponents[11]+=cos(del)*sin(th)*components[3] + cos(th)*components[11] + sin(del)*sin(th)*components[12];
					rotcomponents[12]+=cos(th)*components[12] - sin(th)*(sin(del)*components[11] + cos(del)*components[14]);
					rotcomponents[13]+=components[13];
					rotcomponents[14]+=-(sin(del)*sin(th)*components[3]) + cos(del)*sin(th)*components[12] + cos(th)*components[14];
					rotcomponents[15]+=components[15];
					break;
				case 3:
					rotcomponents[0]+=components[0];
					rotcomponents[1]+=cos(th)*components[1] - sin(th)*(cos(del)*components[7] + sin(del)*components[13]);
					rotcomponents[2]+=cos(th)*components[2] - sin(th)*(cos(del)*components[11] + sin(del)*components[14]);
					rotcomponents[3]+=(cos(th)*cos(th) - cos(2*del)*sin(th)*sin(th))*components[3] - sin(2*del)*sin(th)*sin(th)*components[12] + (cos(del)*sin(2*th)*(3*components[5] + sqrt((double)3)*components[10] + 2*sqrt((double)6)*components[15]))/6.;
					rotcomponents[4]+=cos(th)*components[4] + sin(th)*(-(sin(del)*components[7]) + cos(del)*components[13]);
					rotcomponents[5]+=((3 + cos(2*th))*components[5])/4. - (sin(th)*(6*cos(th)*(cos(del)*components[3] + sin(del)*components[12]) + sqrt((double)3)*sin(th)*(components[10] + 2*sqrt((double)2)*components[15])))/6.;
					rotcomponents[6]+=components[6];
					rotcomponents[7]+=cos(del)*sin(th)*components[1] + sin(del)*sin(th)*components[4] + cos(th)*components[7];
					rotcomponents[8]+=cos(th)*components[8] + sin(th)*(-(sin(del)*components[11]) + cos(del)*components[14]);
					rotcomponents[9]+=components[9];
					rotcomponents[10]+=((11 + cos(2*th))*components[10] - 2*sqrt((double)3)*sin(2*th)*(cos(del)*components[3] + sin(del)*components[12]) - 2*sin(th)*sin(th)*(sqrt((double)3)*components[5] + 2*sqrt((double)2)*components[15]))/12.;
					rotcomponents[11]+=cos(del)*sin(th)*components[2] + sin(del)*sin(th)*components[8] + cos(th)*components[11];
					rotcomponents[12]+=-2*cos(del)*sin(del)*sin(th)*sin(th)*components[3] + (cos(del)*cos(del) + cos(2*th)*sin(del)*sin(del))*components[12] + (sin(del)*sin(2*th)*(3*components[5] + sqrt((double)3)*components[10] + 2*sqrt((double)6)*components[15]))/6.;
					rotcomponents[13]+=sin(del)*sin(th)*components[1] - cos(del)*sin(th)*components[4] + cos(th)*components[13];
					rotcomponents[14]+=sin(del)*sin(th)*components[2] - cos(del)*sin(th)*components[8] + cos(th)*components[14];
					rotcomponents[15]+=(-(sqrt((double)2)*sin(th)*sin(th)*(sqrt((double)3)*components[5] + components[10])) - sqrt((double)6)*sin(2*th)*(cos(del)*components[3] + sin(del)*components[12]) + (1 + 2*cos(2*th))*components[15])/3.;
					break;
				default:
					printf("SUN_rotation error. \n");
					break;
				}
				break;
			case 1:
				switch (j){
				case 2:
					rotcomponents[0]+=components[0];
					rotcomponents[1]+=cos(th)*components[1] - sin(th)*(cos(del)*components[2] + sin(del)*components[8]);
					rotcomponents[2]+=cos(del)*sin(th)*components[1] + cos(th)*components[2] - sin(del)*sin(th)*components[4];
					rotcomponents[3]+=components[3];
					rotcomponents[4]+=sin(del)*sin(th)*components[2] + cos(th)*components[4] - cos(del)*sin(th)*components[8];
					rotcomponents[5]+=((3 + cos(2*th))*components[5] + 2*sin(th)*(2*cos(th)*(cos(del)*components[6] + sin(del)*components[9]) + sqrt((double)3)*sin(th)*components[10]))/4.;
					rotcomponents[6]+=-(cos(del)*cos(th)*sin(th)*components[5]) + (cos(th)*cos(th) - cos(2*del)*sin(th)*sin(th))*components[6] + cos(del)*sin(th)*(-2*sin(del)*sin(th)*components[9] + sqrt((double)3)*cos(th)*components[10]);
					rotcomponents[7]+=cos(th)*components[7] + sin(th)*(-(cos(del)*components[11]) + sin(del)*components[14]);
					rotcomponents[8]+=sin(del)*sin(th)*components[1] + cos(del)*sin(th)*components[4] + cos(th)*components[8];
					rotcomponents[9]+=(-2*sin(2*del)*sin(th)*sin(th)*components[6] + 2*(cos(th)*cos(th) + cos(2*del)*sin(th)*sin(th))*components[9] + sin(del)*sin(2*th)*(-components[5] + sqrt((double)3)*components[10]))/2.;
					rotcomponents[10]+=(2*sqrt((double)3)*sin(th)*sin(th)*components[5] - 2*sqrt((double)3)*sin(2*th)*(cos(del)*components[6] + sin(del)*components[9]) + (1 + 3*cos(2*th))*components[10])/4.;
					rotcomponents[11]+=cos(del)*sin(th)*components[7] + cos(th)*components[11] + sin(del)*sin(th)*components[13];
					rotcomponents[12]+=components[12];
					rotcomponents[13]+=cos(th)*components[13] - sin(th)*(sin(del)*components[11] + cos(del)*components[14]);
					rotcomponents[14]+=-(sin(del)*sin(th)*components[7]) + cos(del)*sin(th)*components[13] + cos(th)*components[14];
					rotcomponents[15]+=components[15];
					break;
				case 3:
					rotcomponents[0]+=components[0];
					rotcomponents[1]+=cos(th)*components[1] - sin(th)*(cos(del)*components[3] + sin(del)*components[12]);
					rotcomponents[2]+=components[2];
					rotcomponents[3]+=cos(del)*sin(th)*components[1] + cos(th)*components[3] - sin(del)*sin(th)*components[4];
					rotcomponents[4]+=sin(del)*sin(th)*components[3] + cos(th)*components[4] - cos(del)*sin(th)*components[12];
					rotcomponents[5]+=(3*(3 + cos(2*th))*components[5] + 2*sin(th)*(6*cos(th)*(cos(del)*components[7] + sin(del)*components[13]) + sqrt((double)3)*sin(th)*(components[10] + 2*sqrt((double)2)*components[15])))/12.;
					rotcomponents[6]+=cos(th)*components[6] - sin(th)*(cos(del)*components[11] + sin(del)*components[14]);
					rotcomponents[7]+=(cos(th)*cos(th) - cos(2*del)*sin(th)*sin(th))*components[7] - sin(2*del)*sin(th)*sin(th)*components[13] + (cos(del)*sin(2*th)*(-3*components[5] + sqrt((double)3)*components[10] + 2*sqrt((double)6)*components[15]))/6.;
					rotcomponents[8]+=components[8];
					rotcomponents[9]+=cos(th)*components[9] + sin(th)*(-(sin(del)*components[11]) + cos(del)*components[14]);
					rotcomponents[10]+=((11 + cos(2*th))*components[10] - 2*sqrt((double)3)*sin(2*th)*(cos(del)*components[7] + sin(del)*components[13]) + 2*sin(th)*sin(th)*(sqrt((double)3)*components[5] - 2*sqrt((double)2)*components[15]))/12.;
					rotcomponents[11]+=cos(del)*sin(th)*components[6] + sin(del)*sin(th)*components[9] + cos(th)*components[11];
					rotcomponents[12]+=sin(del)*sin(th)*components[1] + cos(del)*sin(th)*components[4] + cos(th)*components[12];
					rotcomponents[13]+=(-6*sin(2*del)*sin(th)*sin(th)*components[7] + 6*cos(del)*cos(del)*components[13] + sin(del)*(6*cos(2*th)*sin(del)*components[13] + sin(2*th)*(-3*components[5] + sqrt((double)3)*components[10] + 2*sqrt((double)6)*components[15])))/6.;
					rotcomponents[14]+=sin(del)*sin(th)*components[6] - cos(del)*sin(th)*components[9] + cos(th)*components[14];
					rotcomponents[15]+=(sqrt((double)2)*sin(th)*sin(th)*(sqrt((double)3)*components[5] - components[10]) - sqrt((double)6)*sin(2*th)*(cos(del)*components[7] + sin(del)*components[13]) + (1 + 2*cos(2*th))*components[15])/3.;
					break;
				default:
					printf("SUN_rotation error. \n");
					break;
				}
				break;
			case 2:
				switch (j){
				case 3:
					rotcomponents[0]+=components[0];
					rotcomponents[1]+=components[1];
					rotcomponents[2]+=cos(th)*components[2] - sin(th)*(cos(del)*components[3] + sin(del)*components[12]);
					rotcomponents[3]+=cos(del)*sin(th)*components[2] + cos(th)*components[3] - sin(del)*sin(th)*components[8];
					rotcomponents[4]+=components[4];
					rotcomponents[5]+=components[5];
					rotcomponents[6]+=cos(th)*components[6] - sin(th)*(cos(del)*components[7] + sin(del)*components[13]);
					rotcomponents[7]+=cos(del)*sin(th)*components[6] + cos(th)*components[7] - sin(del)*sin(th)*components[9];
					rotcomponents[8]+=sin(del)*sin(th)*components[3] + cos(th)*components[8] - cos(del)*sin(th)*components[12];
					rotcomponents[9]+=sin(del)*sin(th)*components[7] + cos(th)*components[9] - cos(del)*sin(th)*components[13];
					rotcomponents[10]+=(sqrt((double)3)*(2 + cos(2*th))*components[10] + 2*sin(th)*(3*cos(th)*(cos(del)*components[11] + sin(del)*components[14]) + sqrt((double)6)*sin(th)*components[15]))/(3.*sqrt((double)3));
					rotcomponents[11]+=(cos(th)*cos(th) - cos(2*del)*sin(th)*sin(th))*components[11] - (cos(del)*(6*sin(del)*sin(th)*sin(th)*components[14] + sqrt((double)3)*sin(2*th)*(components[10] - sqrt((double)2)*components[15])))/3.;
					rotcomponents[12]+=sin(del)*sin(th)*components[2] + cos(del)*sin(th)*components[8] + cos(th)*components[12];
					rotcomponents[13]+=sin(del)*sin(th)*components[6] + cos(del)*sin(th)*components[9] + cos(th)*components[13];
					rotcomponents[14]+=-(sin(2*del)*sin(th)*sin(th)*components[11]) + (cos(th)*cos(th) + cos(2*del)*sin(th)*sin(th))*components[14] + (sin(del)*sin(2*th)*(-components[10] + sqrt((double)2)*components[15]))/sqrt((double)3);
					rotcomponents[15]+=(2*sqrt((double)2)*sin(th)*sin(th)*components[10] - sqrt((double)6)*sin(2*th)*(cos(del)*components[11] + sin(del)*components[14]) + (1 + 2*cos(2*th))*components[15])/3.;
					break;
				default:
					printf("SUN_rotation error. \n");
					break;
				}
				break;
			default:
				printf("SUN_rotation error. \n");
				break;
			}
			break;

		default:
			printf("SUN_rotation error. \n");
			break;
		}
	}









} //namespace sumath


}









#endif
