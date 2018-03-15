
# CUDAnuSQuIDS

CUDAnuSQuIDS is a CUDA implementation of nuSQuIDS https://github.com/arguelles/nuSQuIDS.

## Prerequisites
Cuda compilation tools 9.1 or higher
A CUDA capable Pascal or Volta card. Other cards may work, but have not been tested and thus are not included in the Makefile

* nuSQuIDS (>= 1.10.0): https://github.com/arguelles/nuSQuIDS
* SQuIDS (>= 1.2): https://github.com/jsalvado/SQuIDS/
and their respective dependencies, which are:
* gsl (>= 1.15): http://www.gnu.org/software/gsl/
* hdf5 with c bindings: http://www.hdfgroup.org/HDF5/

## Build instructions
./configure
The configuration script tries to automatically detect the dependencies. 
If this is not possible, the libraries can be specified manually.
Run ./configure -h for more information.

make
make install
make examples

The code was built and tested on Ubuntu 16.04 LTS with kernel 4.13.0-36 and 4.4.0-116 and Cuda compilation tools V9.1.85 on GPUs with sm_60, sm_61 and sm_70.
