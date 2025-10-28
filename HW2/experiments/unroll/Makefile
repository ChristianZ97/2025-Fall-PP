CC = gcc
CXX = g++
MPICC = mpicc
MPICXX = mpicxx
LDLIBS = -lpng
BASE_CXXFLAGS = -std=c++17 -O3 -march=native
PROF_CXXFLAGS = $(BASE_CXXFLAGS) -DPROFILING

# Auto-detect all hw2a_*.cc and hw2b_*.cc files
HW2A_SOURCES := $(wildcard hw2a_*.cc)
HW2B_SOURCES := $(wildcard hw2b_*.cc)

# Generate target names (remove .cc extension)
HW2A_TARGETS := $(HW2A_SOURCES:.cc=)
HW2B_TARGETS := $(HW2B_SOURCES:.cc=)

ALL_TARGETS := $(HW2A_TARGETS) $(HW2B_TARGETS)

.PHONY: all
all: $(ALL_TARGETS)
	@echo "=========================================="
	@echo "Compilation Complete!"
	@echo "hw2a targets: $(HW2A_TARGETS)"
	@echo "hw2b targets: $(HW2B_TARGETS)"
	@echo "=========================================="

# Pattern rule for hw2a_* (pthread version)
hw2a_%: hw2a_%.cc
	@echo "Compiling $< -> $@"
	$(CXX) $(PROF_CXXFLAGS) -pthread $< -o $@ $(LDLIBS)

# Pattern rule for hw2b_* (MPI+OpenMP version)
hw2b_%: hw2b_%.cc
	@echo "Compiling $< -> $@"
	$(MPICXX) $(PROF_CXXFLAGS) -fopenmp $< -o $@ $(LDLIBS)

# Clean all compiled binaries
.PHONY: clean
clean:
	rm -f $(ALL_TARGETS)
	@echo "Cleaned all targets"

# List all targets
.PHONY: list
list:
	@echo "hw2a targets: $(HW2A_TARGETS)"
	@echo "hw2b targets: $(HW2B_TARGETS)"
