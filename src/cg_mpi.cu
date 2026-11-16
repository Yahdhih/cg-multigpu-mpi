// cg_mpi.cu : gradient conjugue preconditionne (Jacobi) multi-GPU, MPI + CUDA.
//
// Probleme : Poisson 3D, schema 7 points, grille m x m x M (Dirichlet), b = A*1.
// Decoupage : chaque rang possede une tranche de plans z consecutifs. Avec la
// numerotation k-majeure, un plan est contigu en memoire : le halo a echanger
// avec chaque voisin est donc UN plan (m*m doubles), sans empaquetage.
//
// Le vecteur p (le seul que la SpMV lit avec des voisins) a deux plans
// fantomes apres la partie locale :
//   p[0 .. nloc)               inconnues locales
//   p[nloc .. nloc+pl)         plan fantome du rang du dessous (z-1)
//   p[nloc+pl .. nloc+2pl)     plan fantome du rang du dessus  (z+1)
//
// Modes d'echange (argument 1) :
//   aware-sync     MPI CUDA-aware sur pointeurs device, echange PUIS SpMV
//   aware-overlap  idem, mais lignes interieures calculees PENDANT l'echange,
//                  lignes de bord (1er et dernier plan) apres
//   staged-sync    copie device -> hote, MPI sur buffers hote, copie retour
//                  (ce qu'on fait sans MPI CUDA-aware)
#include <cuda_runtime.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

static const int BS = 256;
static int grid_for(long n) { return (int)std::min<long>((n + BS - 1) / BS, 65535L * 16); }

// ------------------------------------------------------------- noyaux

__global__ void k_spmv_rows(int r0, int r1, const int* __restrict__ rp, const int* __restrict__ ci,
                            const double* __restrict__ v, const double* __restrict__ x, double* __restrict__ y) {
    for (int r = r0 + blockIdx.x * blockDim.x + threadIdx.x; r < r1; r += gridDim.x * blockDim.x) {
        double s = 0.0;
        for (int p = rp[r]; p < rp[r + 1]; p++) s += v[p] * x[ci[p]];
        y[r] = s;
    }
}

__device__ void block_sum_atomic(double s, double* out) {
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
    __shared__ double w[BS / 32];
    if ((threadIdx.x & 31) == 0) w[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x < 32) {
        s = threadIdx.x < BS / 32 ? w[threadIdx.x] : 0.0;
        for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
        if (threadIdx.x == 0) atomicAdd(out, s);
    }
}
__global__ void k_dot(int n, const double* __restrict__ x, const double* __restrict__ y, double* out) {
    double s = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) s += x[i] * y[i];
    block_sum_atomic(s, out);
}
// x += a p ; r -= a q ; out += r.r
__global__ void k_upd_dot(int n, double a, const double* __restrict__ p, const double* __restrict__ q,
                          double* __restrict__ x, double* __restrict__ r, double* out) {
    double s = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        x[i] += a * p[i];
        double ri = r[i] - a * q[i];
        r[i] = ri; s += ri * ri;
    }
    block_sum_atomic(s, out);
}
// z = r / diag ; out += r.z   (diagonale constante egale a 6)
__global__ void k_prec_dot(int n, double dinv, const double* __restrict__ r, double* __restrict__ z, double* out) {
    double s = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        double ri = r[i], zi = dinv * ri;
        z[i] = zi; s += ri * zi;
    }
    block_sum_atomic(s, out);
}
__global__ void k_xpay(int n, const double* __restrict__ x, double a, double* __restrict__ y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) y[i] = x[i] + a * y[i];
}
__global__ void k_axpby(int n, double a, const double* __restrict__ x, double b, double* __restrict__ y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) y[i] = a * x[i] + b * y[i];
}

// ---------------------------------------------------------- contexte

struct Dist {
    int rank, size, up, down;          // voisins MPI (MPI_PROC_NULL aux bords)
    int m, M, nz, z0;                  // grille globale m*m*M, tranche [z0, z0+nz)
    int nloc, pl;                      // inconnues locales, taille d'un plan
    int *rp, *ci; double* val;         // CSR local (colonnes : locales puis fantomes)
    double* scal;                      // resultat des produits scalaires (device)
    double *hs_dn, *hs_up, *hr_dn, *hr_up;   // buffers hote (mode staged)
    cudaStream_t s;
    std::string mode;
    int dot_blocks;
    double t_halo = 0, t_allred = 0;   // temps cumules (attente halo, Allreduce)
};

// Assemble la tranche locale de la matrice directement sur chaque rang
// (aucun rang ne construit la matrice globale) et le second membre b = A*1.
static void assemble(Dist& D, std::vector<double>& b) {
    int m = D.m, pl = D.pl, nloc = D.nloc;
    std::vector<int> rp(nloc + 1), ci; std::vector<double> v;
    ci.reserve(7L * nloc); v.reserve(7L * nloc); b.assign(nloc, 0.0);
    rp[0] = 0;
    for (int kl = 0; kl < D.nz; kl++) {
        int kg = D.z0 + kl;
        for (int j = 0; j < m; j++)
            for (int i = 0; i < m; i++) {
                int r = kl * pl + j * m + i, ij = j * m + i;
                double rowsum = 6.0;
                auto add = [&](int c) { ci.push_back(c); v.push_back(-1.0); rowsum -= 1.0; };
                if (kg > 0) add(kl > 0 ? r - pl : nloc + ij);                  // z-1 (fantome bas)
                if (j > 0) add(r - m);
                if (i > 0) add(r - 1);
                ci.push_back(r); v.push_back(6.0);
                if (i < m - 1) add(r + 1);
                if (j < m - 1) add(r + m);
                if (kg < D.M - 1) add(kl < D.nz - 1 ? r + pl : nloc + pl + ij); // z+1 (fantome haut)
                rp[r + 1] = (int)ci.size();
                b[r] = rowsum;                                                   // (A*1)_r
            }
    }
    long nnz = (long)ci.size();
    CK(cudaMalloc(&D.rp, (nloc + 1) * sizeof(int)));
    CK(cudaMalloc(&D.ci, nnz * sizeof(int)));
    CK(cudaMalloc(&D.val, nnz * sizeof(double)));
    CK(cudaMemcpy(D.rp, rp.data(), (nloc + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(D.ci, ci.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(D.val, v.data(), nnz * sizeof(double), cudaMemcpyHostToDevice));
}

// ------------------------------------------------------ operations

static double gdot_finish(Dist& D) {         // termine un produit scalaire local + Allreduce
    double loc, glob;
    CK(cudaMemcpyAsync(&loc, D.scal, sizeof(double), cudaMemcpyDeviceToHost, D.s));
    CK(cudaStreamSynchronize(D.s));
    double t0 = MPI_Wtime();
    MPI_Allreduce(&loc, &glob, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    D.t_allred += MPI_Wtime() - t0;
    return glob;
}
static double gdot(Dist& D, const double* x, const double* y) {
    CK(cudaMemsetAsync(D.scal, 0, sizeof(double), D.s));
    k_dot<<<std::min(grid_for(D.nloc), D.dot_blocks), BS, 0, D.s>>>(D.nloc, x, y, D.scal);
    return gdot_finish(D);
}

// y = A p, avec echange du halo de p selon le mode choisi.
static void spmv(Dist& D, double* p, double* y) {
    int nloc = D.nloc, pl = D.pl;
    double* gdn = p + nloc;            // fantome bas  <- plan du haut du rang "down"
    double* gup = p + nloc + pl;       // fantome haut <- plan du bas du rang "up"
    MPI_Request rq[4];
    double t0;
    if (D.mode == "staged-sync") {
        CK(cudaMemcpyAsync(D.hs_dn, p, pl * sizeof(double), cudaMemcpyDeviceToHost, D.s));
        CK(cudaMemcpyAsync(D.hs_up, p + nloc - pl, pl * sizeof(double), cudaMemcpyDeviceToHost, D.s));
        CK(cudaStreamSynchronize(D.s));
        t0 = MPI_Wtime();
        MPI_Irecv(D.hr_dn, pl, MPI_DOUBLE, D.down, 0, MPI_COMM_WORLD, &rq[0]);
        MPI_Irecv(D.hr_up, pl, MPI_DOUBLE, D.up, 1, MPI_COMM_WORLD, &rq[1]);
        MPI_Isend(D.hs_dn, pl, MPI_DOUBLE, D.down, 1, MPI_COMM_WORLD, &rq[2]);
        MPI_Isend(D.hs_up, pl, MPI_DOUBLE, D.up, 0, MPI_COMM_WORLD, &rq[3]);
        MPI_Waitall(4, rq, MPI_STATUSES_IGNORE);
        D.t_halo += MPI_Wtime() - t0;
        CK(cudaMemcpyAsync(gdn, D.hr_dn, pl * sizeof(double), cudaMemcpyHostToDevice, D.s));
        CK(cudaMemcpyAsync(gup, D.hr_up, pl * sizeof(double), cudaMemcpyHostToDevice, D.s));
        k_spmv_rows<<<grid_for(nloc), BS, 0, D.s>>>(0, nloc, D.rp, D.ci, D.val, p, y);
        return;
    }
    CK(cudaStreamSynchronize(D.s));    // p doit etre a jour avant que MPI le lise
    t0 = MPI_Wtime();
    MPI_Irecv(gdn, pl, MPI_DOUBLE, D.down, 0, MPI_COMM_WORLD, &rq[0]);
    MPI_Irecv(gup, pl, MPI_DOUBLE, D.up, 1, MPI_COMM_WORLD, &rq[1]);
    MPI_Isend(p, pl, MPI_DOUBLE, D.down, 1, MPI_COMM_WORLD, &rq[2]);
    MPI_Isend(p + nloc - pl, pl, MPI_DOUBLE, D.up, 0, MPI_COMM_WORLD, &rq[3]);
    if (D.mode == "aware-overlap" && D.nz > 2) {
        // lignes interieures : n'utilisent aucun fantome, calculees pendant l'echange
        k_spmv_rows<<<grid_for(nloc - 2 * pl), BS, 0, D.s>>>(pl, nloc - pl, D.rp, D.ci, D.val, p, y);
        MPI_Waitall(4, rq, MPI_STATUSES_IGNORE);
        D.t_halo += MPI_Wtime() - t0;
        k_spmv_rows<<<grid_for(pl), BS, 0, D.s>>>(0, pl, D.rp, D.ci, D.val, p, y);
        k_spmv_rows<<<grid_for(pl), BS, 0, D.s>>>(nloc - pl, nloc, D.rp, D.ci, D.val, p, y);
    } else {
        MPI_Waitall(4, rq, MPI_STATUSES_IGNORE);
        D.t_halo += MPI_Wtime() - t0;
        k_spmv_rows<<<grid_for(nloc), BS, 0, D.s>>>(0, nloc, D.rp, D.ci, D.val, p, y);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    Dist D;
    MPI_Comm_rank(MPI_COMM_WORLD, &D.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &D.size);
    if (argc < 5) {
        if (D.rank == 0) fprintf(stderr, "usage: %s <aware-sync|aware-overlap|staged-sync> <m> <M> <reps> [maxit]\n", argv[0]);
        MPI_Finalize(); return 1;
    }
    D.mode = argv[1]; D.m = atoi(argv[2]); D.M = atoi(argv[3]);
    int reps = atoi(argv[4]), maxit = argc > 5 ? atoi(argv[5]) : 100000;

    // un GPU par rang : rang local dans le noeud -> GPU de meme numero
    MPI_Comm node; int lrank;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
    MPI_Comm_rank(node, &lrank);
    int ngpu; CK(cudaGetDeviceCount(&ngpu)); CK(cudaSetDevice(lrank % ngpu));
    int sms; CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, lrank % ngpu));
    D.dot_blocks = sms * 8;
    CK(cudaStreamCreate(&D.s));

    // tranches de plans les plus egales possible
    D.nz = D.M / D.size + (D.rank < D.M % D.size);
    D.z0 = D.rank * (D.M / D.size) + std::min(D.rank, D.M % D.size);
    D.pl = D.m * D.m; D.nloc = D.nz * D.pl;
    D.down = D.rank > 0 ? D.rank - 1 : MPI_PROC_NULL;
    D.up = D.rank < D.size - 1 ? D.rank + 1 : MPI_PROC_NULL;
    std::vector<double> bh;
    assemble(D, bh);
    CK(cudaMalloc(&D.scal, sizeof(double)));
    CK(cudaMallocHost(&D.hs_dn, D.pl * sizeof(double))); CK(cudaMallocHost(&D.hs_up, D.pl * sizeof(double)));
    CK(cudaMallocHost(&D.hr_dn, D.pl * sizeof(double))); CK(cudaMallocHost(&D.hr_up, D.pl * sizeof(double)));

    int n = D.nloc; size_t vb = n * sizeof(double);
    double *b, *x, *r, *z, *p, *q;
    CK(cudaMalloc(&b, vb)); CK(cudaMalloc(&x, vb)); CK(cudaMalloc(&r, vb));
    CK(cudaMalloc(&z, vb)); CK(cudaMalloc(&q, vb));
    CK(cudaMalloc(&p, vb + 2 * D.pl * sizeof(double)));
    CK(cudaMemcpy(b, bh.data(), vb, cudaMemcpyHostToDevice));
    CK(cudaMemset(p, 0, vb + 2 * D.pl * sizeof(double)));
    const double dinv = 1.0 / 6.0, tol = 1e-8;
    long Ntot = (long)D.m * D.m * D.M;

    if (D.rank == 0)
        printf("mode,ranks,m,M,n_global,rep,iters,time_s,time_per_iter_us,t_halo_s,t_allreduce_s,relres,true_relres,err_inf\n");

    for (int rep = -1; rep < reps; rep++) {   // rep = -1 : echauffement
        CK(cudaMemset(x, 0, vb));
        CK(cudaMemcpy(r, b, vb, cudaMemcpyDeviceToDevice));
        D.t_halo = D.t_allred = 0;
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        CK(cudaMemsetAsync(D.scal, 0, sizeof(double), D.s));
        k_prec_dot<<<std::min(grid_for(n), D.dot_blocks), BS, 0, D.s>>>(n, dinv, r, z, D.scal);
        double rz = gdot_finish(D);
        CK(cudaMemcpyAsync(p, z, vb, cudaMemcpyDeviceToDevice, D.s));
        double bnorm = std::sqrt(gdot(D, b, b)), rnorm = bnorm;
        int k = 0;
        for (; k < maxit; k++) {
            spmv(D, p, q);
            double alpha = rz / gdot(D, p, q);
            CK(cudaMemsetAsync(D.scal, 0, sizeof(double), D.s));
            k_upd_dot<<<std::min(grid_for(n), D.dot_blocks), BS, 0, D.s>>>(n, alpha, p, q, x, r, D.scal);
            rnorm = std::sqrt(gdot_finish(D));
            if (rnorm / bnorm < tol) { k++; break; }
            CK(cudaMemsetAsync(D.scal, 0, sizeof(double), D.s));
            k_prec_dot<<<std::min(grid_for(n), D.dot_blocks), BS, 0, D.s>>>(n, dinv, r, z, D.scal);
            double rz2 = gdot_finish(D), beta = rz2 / rz; rz = rz2;
            k_xpay<<<grid_for(n), BS, 0, D.s>>>(n, z, beta, p);
        }
        CK(cudaStreamSynchronize(D.s));
        double t = MPI_Wtime() - t0;

        // verification : vrai residu b - A x (SpMV distribuee sur une copie de x)
        CK(cudaMemcpy(p, x, vb, cudaMemcpyDeviceToDevice));
        std::string keep = D.mode; D.mode = "aware-sync";
        double th = D.t_halo, ta = D.t_allred;
        spmv(D, p, q);
        D.mode = keep;
        k_axpby<<<grid_for(n), BS, 0, D.s>>>(n, 1.0, b, -1.0, q);     // q = b - A x
        double tr = std::sqrt(gdot(D, q, q)) / bnorm;
        std::vector<double> xh(n); CK(cudaMemcpy(xh.data(), x, vb, cudaMemcpyDeviceToHost));
        double e = 0, eg; for (double xi : xh) e = std::max(e, std::fabs(xi - 1.0));
        MPI_Reduce(&e, &eg, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        double tmax, thm, tam;           // le rang le plus lent fixe le temps
        MPI_Reduce(&t, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&th, &thm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&ta, &tam, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (D.rank == 0 && rep >= 0)
            printf("%s,%d,%d,%d,%ld,%d,%d,%.6e,%.2f,%.4e,%.4e,%.3e,%.3e,%.3e\n", D.mode.c_str(), D.size, D.m,
                   D.M, Ntot, rep, k, tmax, tmax / k * 1e6, thm, tam, rnorm / bnorm, tr, eg);
        fflush(stdout);
    }
    MPI_Finalize();
    return 0;
}
