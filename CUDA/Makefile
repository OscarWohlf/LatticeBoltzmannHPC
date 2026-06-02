# CUDA LBM Makefile
#
# Suggested SCITAS/cluster modules:
#   module load gcc cuda hdf5
#
# For V100 GPUs, sm_70 is appropriate.
# You can override with:
#   make CUDA_ARCH=sm_80
#
# If needed:
#   make HDF5_ROOT=/path/to/hdf5

CXX      ?= g++
NVCC     ?= nvcc
LD       := $(NVCC)

TARGET   := lbm_cuda

CUDA_ARCH ?= sm_70

HDF5_INC := $(if $(strip $(HDF5_ROOT)),-I$(HDF5_ROOT)/include)
HDF5_LIB := $(if $(strip $(HDF5_ROOT)),-L$(HDF5_ROOT)/lib)

CXXFLAGS += -O3 -Wall -Wextra -pedantic -std=c++11 $(HDF5_INC)
NVCCFLAGS += -O3 -std=c++11 -arch=$(CUDA_ARCH) $(HDF5_INC)

LDFLAGS += $(HDF5_LIB) -lhdf5 -lm

OBJS = main_cuda.o lbm_cuda.o output_cuda.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(LD) -o $@ $(OBJS) $(LDFLAGS)

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $<

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

clean:
	rm -f $(TARGET) *.o *~

distclean: clean
	rm -rf out/ probes/ probe.csv *.gif

.PHONY: all clean distclean