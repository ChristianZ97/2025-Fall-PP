# ==========================================
# Makefile for C, C++, CUDA
# ==========================================

# Compilers
CC     = gcc
CXX    = g++
NVCC   = nvcc

# Flags
CFLAGS    = -O2 -fopenmp
CXXFLAGS  = -O2 -pthread -fopenmp
NVFLAGS   = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 -cudart shared -Xcompiler="-fopenmp"
LDFLAGS   = -lm

# ------------------------------------------
# Source Detection
# ------------------------------------------
C_SRCS    := $(wildcard *.c)
CXX_SRCS  := $(wildcard *.cc)
CU_SRCS   := $(wildcard *.cu)

# ------------------------------------------
# Target Generation (Adding Suffixes!)
# ------------------------------------------
# nbody.c  -> nbody_c
# nbody.cu -> nbody_cu
C_EXES    := $(C_SRCS:.c=_c)
CXX_EXES  := $(CXX_SRCS:.cc=_cc)
CU_EXES   := $(CU_SRCS:.cu=_cu)

.PHONY: all clean c cpp cuda

all: c cpp cuda
	@echo "Build complete."

c: $(C_EXES)
cpp: $(CXX_EXES)
cuda: $(CU_EXES)

clean:
	rm -f $(C_EXES) $(CXX_EXES) $(CU_EXES)

# ------------------------------------------
# Build Rules
# ------------------------------------------

# 1. C Rule: %_c depends on %.c
# Example: nbody_c depends on nbody.c
%_c: %.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# 2. C++ Rule: %_cc depends on %.cc
%_cc: %.cc
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

# 3. CUDA Rule: %_cu depends on %.cu
%_cu: %.cu
	$(NVCC) $(NVFLAGS) -o $@ $< $(LDFLAGS)
