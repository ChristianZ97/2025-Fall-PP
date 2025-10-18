CC = gcc
CXX = g++
LDLIBS = -lpng
BASE_CFLAGS = -lm -O3 -march=native
PROF_CFLAGS = $(BASE_CFLAGS) -DPROFILING
PROF_LDLIBS = $(LDLIBS) -lnvToolsExt

hw2a: CFLAGS = $(BASE_CFLAGS) -pthread
hw2a_prof: CFLAGS = $(PROF_CFLAGS) -pthread
hw2a_prof: LDLIBS = $(PROF_LDLIBS)

hw2b: CC = mpicc
hw2b: CXX = mpicxx
hw2b: CFLAGS = $(BASE_CFLAGS) -fopenmp
hw2b_prof: CC = mpicc
hw2b_prof: CXX = mpicxx
hw2b_prof: CFLAGS = $(PROF_CFLAGS) -fopenmp
hw2b_prof: LDLIBS = $(PROF_LDLIBS)

CXXFLAGS = $(CFLAGS)
TARGETS = hw2seq hw2a hw2b
PROF_TARGETS = hw2a_prof hw2b_prof

.PHONY: all
all: $(TARGETS)

.PHONY: prof
prof: $(PROF_TARGETS)

hw2a_prof: hw2a.cc
	$(CXX) $(CFLAGS) hw2a.cc $(LDLIBS) -o hw2a_prof

hw2b_prof: hw2b.cc
	$(CXX) $(CFLAGS) hw2b.cc $(LDLIBS) -o hw2b_prof

.PHONY: clean
clean:
	rm -f $(TARGETS) $(PROF_TARGETS) $(TARGETS:=.o) *.png

.PHONY: test
test: hw2a
	./hw2a output.png 10000 -0.5506164691618783 -0.5506164628264113 0.6273445437118131 0.6273445403522527 7680 4320

.PHONY: test-prof
test-prof: hw2a_prof
	./hw2a_prof output_prof.png 10000 -0.5506164691618783 -0.5506164628264113 0.6273445437118131 0.6273445403522527 7680 4320

.PHONY: test-from-file
test-from-file: hw2a
	./hw2a output.png $$(cat testcases/strict34.txt)
