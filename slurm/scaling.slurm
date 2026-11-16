#!/bin/bash
# Scalabilite forte et faible du CG multi-GPU. Un job par nombre de noeuds :
#   sbatch -N 1 slurm/scaling.slurm   (1, 2 et 4 GPU)
#   sbatch -N 2 slurm/scaling.slurm   (8 GPU)
#   sbatch -N 4 slurm/scaling.slurm   (16 GPU)
#SBATCH -J cg-mpi
#SBATCH -p short
#SBATCH -A r260073
#SBATCH --constraint=armgpu
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH -c 16
#SBATCH --mem=200G
#SBATCH -t 01:30:00
#SBATCH -o results/scaling_%j.log

module load nvhpc/nvhpc-hpcx-cuda12/24.11 >/dev/null 2>&1
OUT=results/scaling_N${SLURM_NNODES}_${SLURM_JOB_ID}.csv
REPS=5
if [ "$SLURM_NNODES" = 1 ]; then PLIST="1 2 4"; else PLIST=$((4 * SLURM_NNODES)); fi
# chaque rang sur le CPU Grace (noeud NUMA) de son GPU
RUN="mpirun --map-by ppr:4:node --bind-to numa"
echo "kind,mode,ranks,m,M,n_global,rep,iters,time_s,time_per_iter_us,t_halo_s,t_allreduce_s,relres,true_relres,err_inf" > $OUT
for P in $PLIST; do
  for md in aware-sync aware-overlap staged-sync; do
    # forte : 512^3 = 134 M d'inconnues, resolu jusqu'a convergence
    $RUN -np $P ./bin/cg_mpi $md 512 512 $REPS 2>/dev/null | grep -v "^mode" | sed "s/^/strong,/" >> $OUT
    # faible : 256 x 256 x (256 P), 16,8 M d'inconnues par GPU, 1000 iterations fixes
    $RUN -np $P ./bin/cg_mpi $md 256 $((256 * P)) $REPS 1000 2>/dev/null | grep -v "^mode" | sed "s/^/weak,/" >> $OUT
  done
  echo "fini P=$P $(date +%T)"
done
