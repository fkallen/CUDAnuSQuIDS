
# CUDAnuSQuIDS

CUDAnuSQuIDS is a CUDA library to solve neutrino propagation in the presence of matter
with respect to neutrino oscillation and neutrino interactions.
Given a matter profile (body), a list of trajectories through this body, and a list of neutrino energies,
CUDAnuSQuIDS can propagate the neutrino states for all pairs of (trajectory, neutrino energy) in parallel
on multiple CUDA-capable GPUs

CUDAnuSQuIDS is a CUDA implementation of nuSQuIDS https://github.com/arguelles/nuSQuIDS.

CUDAnuSQuIDS supports three and four neutrino flavors, where the fourth flavor is assumed to be a sterile neutrino.

## Prerequisites
Cuda compilation tools 9.1 or higher
A CUDA capable Pascal or Volta card. Other cards may work, but have not been tested and thus are not included in the Makefile

* nuSQuIDS (>= 1.10.0): https://github.com/arguelles/nuSQuIDS
* SQuIDS (>= 1.2): https://github.com/jsalvado/SQuIDS/
and their respective dependencies, which are:
* gsl (>= 1.15): http://www.gnu.org/software/gsl/
* hdf5 with c bindings: http://www.hdfgroup.org/HDF5/


The code was built and tested on Ubuntu 16.04 LTS with kernel 4.13.0-36 and 4.4.0-116 and Cuda compilation tools V9.1.85 on GPUs with sm_60, sm_61 and sm_70.

## Build instructions
./configure
The configuration script tries to automatically detect the dependencies.
If this is not possible, the libraries can be specified manually.
Run ./configure -h for more information.

make
make install

To build examples:
make examples

## Basic usage
Please refer to examples/atmospheric/main.cu to see a full example how to use this library


1. Create a shared pointer to cudanusquids::ParameterObject
2. Set neutrino parameters in parameter object
3. Set simulation parameters in parameter object
4. Set device ids of GPUs to be used
5. Set initial neutrino flux
6. Use parameter object to create a CudaNusquids instance
7. For each device id, create a matter body and assign it to CudaNusquids
8. Create list of paths through this body type and assign it to CudaNusquids
9. Call CudaNusquids::evolve()
10. For each path check if propagation was successful and retrieve results


## Advanced usage - Customize neutrino properties

### Cross sections
CUDAnuSQuIDS uses the cross section information of nuSQuIDS provided by class nusquids::NeutrinoCrossSections.
A derived type of nusquids::NeutrinoCrossSections can be passed to the constructor of cudanusquids::ParameterObject as sixth parameter.

```
std::shared_ptr<nusquids::NeutrinoCrossSections> crossSections
        = std::make_shared<nusquids::NeutrinoDISCrossSectionsFromTables>();
std::shared_ptr<cudanusquids::ParameterObject> params
    = std::make_shared<cudanusquids::ParameterObject>(..., crossSections);
```

### Custom neutrino physics

The core of the physics simulation is defined in file include/cudanuSQuIDS/physics.cuh

    template<int n_flavors, class body_t, class Op_t>
    struct Physics

manages the neutrino data and is responsible the derivation of all energy bins along a specific neutrino path.
Template parameter class Op_t provides the functionality to calculate the actual operators like the Hamiltonian.
CUDAnuSQuIDS uses static polymorphism via templates to allow custom neutrino physics.

    struct PhysicsOps

is used as Op_t by default.

Class CudaNusquids can take a third template parameter CustomPhysicsOps which must provide the same interface as PhysicsOps.
If defined, struct Physics will use CustomPhysicsOps instead of PhysicsOps.

User defined GPU arrays can be attached to Physics via ParameterObject::registerAdditionalData(num_bytes) which can then be accessed by CustomPhysicsOps.
Pointers to arrays created via registerAdditionalData are stored in member variable void** additionalData of struct Physics, in the order of creation,
i.e. additionalData[0] points to array created by first call to registerAdditionalData, additionalData[1] points to second array and so on.

The method CustomPhysicsOps::addToPrederive can be implemented to update user defined non constant GPU data before a derivation step.

Please refer to examples/nsi/ to see a full example how to use custom physics
