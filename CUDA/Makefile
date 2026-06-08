# CUDA LBM Makefile
#
# Suggested SCITAS/cluster modules:
#   module load gcc cuda hdf5
#
# For V100 GPUs, sm_70 is appropriate.
# Override if needed:
#   make -f Makefile.CUDA CUDA_ARCH=sm_80
#
# If HDF5 is not provided through the loaded module:
#   make -f Makefile.CUDA HDF5_ROOT=/path/to/hdf5
#
# Targets:
#   make -f Makefile.CUDA
#       Builds:
#         lbm_cuda       (default SoA version)
#         lbm_cuda_soa
#         lbm_cuda_aos
#
#   make -f Makefile.CUDA lbm_cuda_soa
#   make -f Makefile.CUDA lbm_cuda_aos

CXX  ?= g++
NVCC ?= nvcc
LD   := $(NVCC)

CUDA_ARCH ?= sm_70

TARGET_DEFAULT := lbm_cuda
TARGET_SOA     := lbm_cuda_soa
TARGET_AOS     := lbm_cuda_aos

HDF5_INC := $(if $(strip $(HDF5_ROOT)),-I$(HDF5_ROOT)/include)
HDF5_LIB := $(if $(strip $(HDF5_ROOT)),-L$(HDF5_ROOT)/lib)

CXXFLAGS  += -O3 -Wall -Wextra -pedantic -std=c++11 $(HDF5_INC)
NVCCFLAGS += -O3 -std=c++11 -arch=$(CUDA_ARCH) $(HDF5_INC)

LDFLAGS += $(HDF5_LIB) -lhdf5 -lm

COMMON_OBJS := main_cuda.o output_cuda.o
SOA_OBJS    := $(COMMON_OBJS) lbm_cuda_soa.o
AOS_OBJS    := $(COMMON_OBJS) lbm_cuda_aos.o


# Build all versions by default.
all: $(TARGET_DEFAULT) $(TARGET_SOA) $(TARGET_AOS)


# Existing default executable.
# This remains the SoA implementation so old scripts using ./lbm_cuda work.
$(TARGET_DEFAULT): $(SOA_OBJS)
	$(LD) -o $@ $(SOA_OBJS) $(LDFLAGS)


# Explicit SoA executable.
$(TARGET_SOA): $(SOA_OBJS)
	$(LD) -o $@ $(SOA_OBJS) $(LDFLAGS)


# Explicit AoS executable.
$(TARGET_AOS): $(AOS_OBJS)
	$(LD) -o $@ $(AOS_OBJS) $(LDFLAGS)


# Host-side output implementation.
output_cuda.o: output_cuda.cc output_cuda.hh lbm_cuda.hh
	$(CXX) $(CXXFLAGS) -c -o $@ $<


# Main CUDA program.
# This object can be shared because it contains no layout-dependent indexing.
main_cuda.o: main_cuda.cu lbm_cuda.hh output_cuda.hh
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<


# Structure-of-arrays solver.
# USE_AOS is not defined, so fidx() selects the SoA branch.
lbm_cuda_soa.o: lbm_cuda.cu lbm_cuda.hh
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<


# Array-of-structures solver.
# USE_AOS selects the AoS branch inside fidx().
lbm_cuda_aos.o: lbm_cuda.cu lbm_cuda.hh
	$(NVCC) $(NVCCFLAGS) -DUSE_AOS -c -o $@ $<


clean:
	rm -f \
		$(TARGET_DEFAULT) \
		$(TARGET_SOA) \
		$(TARGET_AOS) \
		*.o \
		*~


distclean: clean
	rm -rf out/ probes/ probe.csv *.gif


.PHONY: all clean distclean