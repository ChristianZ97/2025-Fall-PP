
# MPI Compilation Guide

## Overview

This document provides compilation instructions for all possible combinations of compilers and MPI implementations available on the server.

## Available Combinations

### 1. GCC + Intel MPI
- {40 182.33}

```bash
ml purge
module load mpi
make clean
make CC=mpicc CXX=mpicxx
```


### 2. GCC + OpenMPI (best)
- {40 142.02}

```bash
ml purge
module load openmpi
make clean
make CC=mpicc CXX=mpicxx
```


### 3. Intel Classic + Intel MPI
- {40 182.91}

```bash
ml purge
module load icc mpi
make clean
make CC=mpicc CXX=mpicxx
```


### 4. Intel Classic + OpenMPI
- {40 142.86}

```bash
ml purge
module load icc openmpi
make clean
make CC=mpicc CXX=mpicxx
```


### 5. Intel oneAPI + Intel MPI
- {40 181.06}

```bash
ml purge
module load compiler mpi
make clean
make CC=mpicc CXX=mpicxx
```


### 6. Intel oneAPI + OpenMPI
- {40 142.36}

```bash
ml purge
module load compiler openmpi
make clean
make CC=mpicc CXX=mpicxx
```

