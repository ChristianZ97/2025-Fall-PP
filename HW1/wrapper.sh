#! /bin/bash
mkdir -p nsys_reports
# Output to ./nsys_reports/rank_$N.nsys-rep
nsys profile \
-o "./nsys_reports/rank_$SLURM_PROCID.nsys-rep" \
--mpi-impl openmpi \
--trace mpi,nvtx \
$@