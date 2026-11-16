# Sur un noeud GH200 de ROMEO, avec le MPI HPC-X de NVHPC (UCX compile avec CUDA) :
#   module load nvhpc/nvhpc-hpcx-cuda12/24.11 && make
# nvcc utilise mpicxx comme compilateur hote : il ajoute lui-meme les options MPI.
bin/cg_mpi: src/cg_mpi.cu
	@mkdir -p bin
	nvcc -O3 -std=c++17 -arch=sm_90 -ccbin mpicxx -o $@ $<

clean:
	rm -rf bin
