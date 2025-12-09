# ==========================================
# Simple Makefile for CPU (gcc) and GPU (nvcc)
# ==========================================

# Compilers
CC   = gcc
NVCC = nvcc

# Flags
CFLAGS  = -O2 -fopenmp -lm
NVFLAGS = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 -cudart shared -Xcompiler="-fopenmp"

# Automatically find source files
C_SOURCES  := $(wildcard *.c)
CU_SOURCES := $(wildcard *.cu)

# Generate executable names
C_TARGETS  := $(C_SOURCES:.c=)
CU_TARGETS := $(CU_SOURCES:.cu=)

.PHONY: all clean

# Default target: build both CPU and GPU executables
all: $(C_TARGETS) $(CU_TARGETS)
	@echo "Build complete."
	@echo "CPU Targets: $(C_TARGETS)"
	@echo "GPU Targets: $(CU_TARGETS)"

# Rule for C files (CPU)
%: %.c
	$(CC) $(CFLAGS) -o $@ $<

# Rule for CUDA files (GPU)
%: %.cu
	$(NVCC) $(NVFLAGS) -o $@ $<

# Clean up
clean:
	rm -f $(C_TARGETS) $(CU_TARGETS)
