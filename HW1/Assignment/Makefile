CC = mpicc
CXX = mpicxx
# Base flags for all builds
BASE_CXXFLAGS = -std=c++17 -O3 -march=native -flto -fomit-frame-pointer
# Append the profiling flag for the profiling build
PROF_CXXFLAGS = $(BASE_CXXFLAGS) -DPROFILING

# Define the standard targets
TARGETS = hw1
PROF_TARGET = hw1_prof

.PHONY: all
all: $(TARGETS)

# Rule for the standard submission build
hw1: hw1.cc
	$(CXX) $(BASE_CXXFLAGS) hw1.cc -o hw1

# New rule for the profiling build
.PHONY: prof
prof: $(PROF_TARGET)

$(PROF_TARGET): hw1.cc
	$(CXX) $(PROF_CXXFLAGS) hw1.cc -o $(PROF_TARGET)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(PROF_TARGET)
