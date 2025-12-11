# ==========================================
# Makefile for C, C++, CUDA
# Supports: Standard, Debug (-DDEBUG), Profiling (-DPROFILING)
# ==========================================

# Compilers
CC     = gcc
NVCC   = nvcc

# Base Flags (Optimization & Architecture)
CFLAGS_BASE   = -Wall -g
NVFLAGS_BASE  = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61
LDFLAGS       = -lm

# Variant Flags
# 1. Debug: Adds (symbols) and -DDEBUG (macro)
CFLAGS_DBG    = $(CFLAGS_BASE) -DDEBUG
NVFLAGS_DBG   = $(NVFLAGS_BASE) -DDEBUG

# 2. Profiling: Adds -DPROFILING (macro)
CFLAGS_PROF   = $(CFLAGS_BASE) -DPROFILING
NVFLAGS_PROF  = $(NVFLAGS_BASE) -DPROFILING

# ------------------------------------------
# Source & Target Generation
# ------------------------------------------
C_SRCS    := $(wildcard *.c)
CU_SRCS   := $(wildcard *.cu)

# Standard Targets (e.g., nbody_c, nbody_cu)
C_EXES    := $(C_SRCS:.c=_c)
CU_EXES   := $(CU_SRCS:.cu=_cu)

# Debug Targets (e.g., nbody_c_debug)
C_DBG     := $(C_SRCS:.c=_c_debug)
CU_DBG    := $(CU_SRCS:.cu=_cu_debug)

# Profiling Targets (e.g., nbody_c_prof)
C_PROF    := $(C_SRCS:.c=_c_prof)
CU_PROF   := $(CU_SRCS:.cu=_cu_prof)

# Grouped Lists
TARGETS_ALL  = $(C_EXES) $(CXX_EXES) $(CU_EXES)
TARGETS_DBG  = $(C_DBG) $(CXX_DBG) $(CU_DBG)
TARGETS_PROF = $(C_PROF) $(CXX_PROF) $(CU_PROF)

# ------------------------------------------
# Phony Rules
# ------------------------------------------
.PHONY: all debug prof clean c cpp cuda

# Default: Build only Standard versions
all: $(TARGETS_ALL)
	@echo "Build complete (Standard)."

# Build Debug versions
debug: $(TARGETS_DBG)
	@echo "Build complete (Debug)."

# Build Profiling versions
prof: $(TARGETS_PROF)
	@echo "Build complete (Profiling)."

clean:
	rm -f $(TARGETS_ALL) $(TARGETS_DBG) $(TARGETS_PROF)

# ------------------------------------------
# Compilation Rules
# ------------------------------------------

# --- Standard Rules ---
%_c: %.c
	$(CC) $(CFLAGS_BASE) -o $@ $? $(LDFLAGS)

%_cu: %.cu
	$(NVCC) $(NVFLAGS_BASE) -o $@ $? $(LDFLAGS)

# --- Debug Rules (_debug suffix) ---
%_c_debug: %.c
	$(CC) $(CFLAGS_DBG) -o $@ $? $(LDFLAGS)

%_cu_debug: %.cu
	$(NVCC) $(NVFLAGS_DBG) -o $@ $? $(LDFLAGS)

# --- Profiling Rules (_prof suffix) ---
%_c_prof: %.c
	$(CC) $(CFLAGS_PROF) -o $@ $? $(LDFLAGS)

%_cu_prof: %.cu
	$(NVCC) $(NVFLAGS_PROF) -o $@ $? $(LDFLAGS)
