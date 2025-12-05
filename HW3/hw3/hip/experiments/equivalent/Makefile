CC     = gcc
CXX    = g++
NVCC   = nvcc
HIPCC  = hipcc

CXXFLAGS      = -O2 -pthread -fopenmp
# NVIDIA Flags (GTX 1080)
NVFLAGS       = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 -cudart shared -DPROFILING
# AMD Flags (MI210 / MI100, gfx90a is usually for MI210, check if yours is gfx908 or gfx90a)
# Using gfx90a as default for newer MI210 clusters, change back to gfx908 if needed.
HIPCCFLAGS    = -std=c++11 -O3 --offload-arch=gfx908 -DPROFILING

LDFLAGS = -lm

# Define your experiment targets here
EXES_GROUP0 = hw3-2-group0-cuda hw3-2-group0-hip
EXES_GROUP1 = hw3-2-group1-cuda hw3-2-group1-hip

.PHONY: all clean

all: $(EXES_GROUP0) $(EXES_GROUP1)

clean:
	rm -f $(EXES_GROUP0) $(EXES_GROUP1)

# =========================
# Group 0 Targets (Best CUDA -> HIP)
# =========================
hw3-2-group0-cuda: hw3-2-group0-cuda.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $<

hw3-2-group0-hip: hw3-2-group0-hip.hip
	$(HIPCC) $(HIPCCFLAGS) $(LDFLAGS) -o $@ $<

# =========================
# Group 1 Targets (Best HIP -> CUDA)
# =========================
hw3-2-group1-cuda: hw3-2-group1-cuda.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $<

hw3-2-group1-hip: hw3-2-group1-hip.hip
	$(HIPCC) $(HIPCCFLAGS) $(LDFLAGS) -o $@ $<
