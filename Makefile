NAME = cudanusquids
DYNAMIC_LIBRARY = lib/lib$(NAME).so
STATIC_LIBRARY = lib/lib$(NAME).a

NVCC=nvcc

PREFIX=$(shell cat .PREFIX)
NUSQUIDS_INCDIR=$(shell cat .NUSQUIDS_INCDIR)
NUSQUIDS_LIBDIR=$(shell cat .NUSQUIDS_LIBDIR)
SQUIDS_INCDIR=$(shell cat .SQUIDS_INCDIR)
SQUIDS_LIBDIR=$(shell cat .SQUIDS_LIBDIR)
HDF5_INCDIR=$(shell cat .HDF5_INCDIR)
HDF5_LIBDIR=$(shell cat .HDF5_LIBDIR)
GSL_INCDIR=$(shell cat .GSL_INCDIR)
GSL_LIBDIR=$(shell cat .GSL_LIBDIR)

GSL_CFLAGS=-I$(GSL_INCDIR)
GSL_LDFLAGS=-L$(GSL_LIBDIR) -lgsl -lgslcblas -lm

HDF5_CFLAGS=-I$(HDF5_INCDIR)
HDF5_LDFLAGS=-L$(HDF5_LIBDIR) -lhdf5_hl -lhdf5 -lz -ldl -lm

SQUIDS_CFLAGS=-I$(SQUIDS_INCDIR)
SQUIDS_LDFLAGS=-L$(SQUIDS_LIBDIR) -lSQuIDS

NUSQUIDS_CFLAGS=-I$(NUSQUIDS_INCDIR)
NUSQUIDS_LDFLAGS=-L$(NUSQUIDS_LIBDIR) -lnuSQuIDS

CUDANUSQUIDS_CFLAGS=-I./include -g -O3
CUDANUSQUIDS_LDFLAGS=-L./lib -l$(NAME)

CFLAGS=  $(HDF5_CFLAGS) $(NUSQUIDS_CFLAGS) $(SQUIDS_CFLAGS) $(GSL_CFLAGS) $(CUDANUSQUIDS_CFLAGS)
LDFLAGS= $(HDF5_LDFLAGS) $(NUSQUIDS_LDFLAGS) $(SQUIDS_LDFLAGS) $(GSL_LDFLAGS) $(CUDANUSQUIDS_LDFLAGS)

CPPFLAGS = -std=c++14

CUDAARCH = -gencode arch=compute_60,code=sm_60 \
	   -gencode arch=compute_61,code=sm_61 \
	   -gencode arch=compute_70,code=sm_70 \
	   -gencode arch=compute_70,code=compute_70


NVCCFLAGS = $(CUDAARCH) -lineinfo --expt-relaxed-constexpr -Xcudafe "--diag_suppress=subscript_out_of_range" -rdc=true -Xcompiler "-fPIC -Wall"

CU_SOURCES = $(wildcard src/*.cu)
CU_OBJS = $(patsubst src/%.cu,build/%.o,$(CU_SOURCES))

exampleList = examples/atmospheric/main examples/nsi/main

all:	$(DYNAMIC_LIBRARY) $(STATIC_LIBRARY)

$(STATIC_LIBRARY): $(CU_OBJS)
	@echo "creating static lib"
	@$(NVCC) $(CUDAARCH) -dlink $(CU_OBJS) -o build/devicecode.o
	@ar rcs $(STATIC_LIBRARY) $(CU_OBJS) build/devicecode.o

$(DYNAMIC_LIBRARY): $(CU_OBJS)
	@echo "creating dynamic lib"
	@$(NVCC) $(CUDAARCH) --shared $(CU_OBJS) -o $(DYNAMIC_LIBRARY)


build/%.o : src/%.cu
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

examples: $(exampleList)

examples/dev/main: examples/dev/main.cu $(DYNAMIC_LIBRARY)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) examples/dev/main.cu -o $@

examples/atmospheric/main: examples/atmospheric/main.cu $(DYNAMIC_LIBRARY)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) examples/atmospheric/main.cu -o $@

examples/nsi/main: examples/nsi/main.cu $(DYNAMIC_LIBRARY)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) examples/nsi/main.cu -o $@


clean:
	rm build/*.o
	rm $(DYNAMIC_LIBRARY) $(STATIC_LIBRARY)

cleanexamples:
	rm $(exampleList)

install: $(STATIC_LIBRARY) $(DYNAMIC_LIBRARY)
	mkdir -p $(PREFIX)/include
	mkdir -p $(PREFIX)/lib
	mkdir -p $(PREFIX)/lib/pkgconfig
	cp -r include/cudanuSQuIDS $(PREFIX)/include
	cp $(DYNAMIC_LIBRARY) $(STATIC_LIBRARY) $(PREFIX)/lib
	cp lib/*.pc $(PREFIX)/lib/pkgconfig
