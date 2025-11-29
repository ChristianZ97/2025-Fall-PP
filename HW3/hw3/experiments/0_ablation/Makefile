# Makefile for Unroll Experiments

# Compiler settings
NVCC = nvcc
# Optimized flags for sm_61 architecture with profiling enabled
NVFLAGS = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 -cudart shared -DPROFILING

# Auto-detect all .cu files in current directory
SOURCES := $(wildcard *.cu)
# Generate target executable names by removing .cu extension
TARGETS := $(SOURCES:.cu=)

.PHONY: all clean

# Default target: compile all .cu files to executables
all: $(TARGETS)
	@echo "Built targets: $(TARGETS)"

# Rule to compile any .cu file into an executable
%: %.cu
	$(NVCC) $(NVFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
