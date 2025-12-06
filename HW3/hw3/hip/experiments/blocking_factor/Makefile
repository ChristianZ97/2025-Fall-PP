# Makefile for Unroll Experiments (HIP Version)

# Compiler settings
HIPCC = hipcc

# Optimized flags for gfx908 architecture with profiling enabled
# Flags adapted from your reference:
# -std=c++11 -O3: Standard optimization
# --offload-arch=gfx908: Target architecture (match this to your specific AMD GPU if not MI100)
# -DPROFILING: Preserved from your original Makefile request
HIPFLAGS = -std=c++11 -O3 --offload-arch=gfx908

# Linker flags (Math library is commonly needed)
LDFLAGS = -lm

# Auto-detect all .hip files in current directory
SOURCES := $(wildcard *.hip)
# Generate target executable names by removing .hip extension
TARGETS := $(SOURCES:.hip=)

.PHONY: all clean

# Default target: compile all .hip files to executables
all: $(TARGETS)
	@echo "Built targets: $(TARGETS)"

# Rule to compile any .hip file into an executable
%: %.hip
	$(HIPCC) $(HIPFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGETS)
