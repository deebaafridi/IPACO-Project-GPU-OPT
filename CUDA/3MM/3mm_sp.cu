
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>

#define POLYBENCH_TIME 1

#include "3mm.cuh"
#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"

#define GPU_DEVICE 0

#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

// Optimization defines
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define WARPS_PER_BLOCK ((BLOCK_SIZE_X * BLOCK_SIZE_Y) / 32)
#define FP64_WARPS (WARPS_PER_BLOCK / 2) // Half the warps for FP64

#define RUN_ON_CPU


void init_array(int ni, int nj, int nk, int nl, int nm, DATA_TYPE POLYBENCH_2D(A, NI, NK, ni, nk), DATA_TYPE POLYBENCH_2D(B, NK, NJ, nk, nj),
        DATA_TYPE POLYBENCH_2D(C, NJ, NM, nj, nm), DATA_TYPE POLYBENCH_2D(D, NM, NL, nm, nl))
{
    int i, j;

    for (i = 0; i < ni; i++)
    {
        for (j = 0; j < nk; j++)
        {
            A[i][j] = ((DATA_TYPE) i*j) / ni;
        }
    }

    for (i = 0; i < nk; i++)
    {
        for (j = 0; j < nj; j++)
        {
            B[i][j] = ((DATA_TYPE) i*(j+1)) / nj;
        }
    }

    for (i = 0; i < nj; i++)
    {
        for (j = 0; j < nm; j++)
        {
            C[i][j] = ((DATA_TYPE) i*(j+3)) / nl;
        }
    }

    for (i = 0; i < nm; i++)
    {
        for (j = 0; j < nl; j++)
        {
            D[i][j] = ((DATA_TYPE) i*(j+2)) / nk;
        }
    }
}


void compareResults(int ni, int nl, DATA_TYPE POLYBENCH_2D(G, NI, NL, ni, nl), DATA_TYPE POLYBENCH_2D(G_outputFromGpu, NI, NL, ni, nl))
{
    int i,j,fail;
    fail = 0;

    for (i=0; i < ni; i++)
    {
        for (j=0; j < nl; j++)
        {
            if (percentDiff(G[i][j], G_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
            {
                fail++;
            }
        }
    }

    // print results
    printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void GPU_argv_init()
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
    printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
    cudaSetDevice( GPU_DEVICE );
}


__global__ void mm3_kernel1_optimized(int ni, int nj, int nk, int nl_d, int nm_d, DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *E)
{
    // E(ni,nj) = A(ni,nk) * B(nk,nj)
    int tx = threadIdx.x;    int ty = threadIdx.y;
    int bx = blockIdx.x;    int by = blockIdx.y;

    int row = by * BLOCK_SIZE_Y + ty; // Global row index for E (maps to ni)
    int col = bx * BLOCK_SIZE_X + tx; // Global col index for E (maps to nj)

    int warpId = (ty * BLOCK_SIZE_X + tx) / 32;
    bool useFP64 = (warpId < FP64_WARPS);

    if (row < ni && col < nj) {
        if (useFP64) {
            double e_val = 0.0;
            for (int k_loop = 0; k_loop < nk; ++k_loop) {
                e_val += (double)A[row * nk + k_loop] * (double)B[k_loop * nj + col];
            }
            E[row * nj + col] = (DATA_TYPE)e_val;
        } else {
            float e_val = 0.0f;
            for (int k_loop = 0; k_loop < nk; ++k_loop) {
                e_val += (float)A[row * nk + k_loop] * (float)B[k_loop * nj + col];
            }
            E[row * nj + col] = (DATA_TYPE)e_val;
        }
    }
}


__global__ void mm3_kernel2_optimized(int ni_d, int nj, int nk_d, int nl, int nm, DATA_TYPE *C, DATA_TYPE *D, DATA_TYPE *F)
{
    // F(nj,nl) = C(nj,nm) * D(nm,nl)
    int tx = threadIdx.x;    int ty = threadIdx.y;
    int bx = blockIdx.x;    int by = blockIdx.y;

    int row = by * BLOCK_SIZE_Y + ty; // Global row index for F (maps to nj)
    int col = bx * BLOCK_SIZE_X + tx; // Global col index for F (maps to nl)

    int warpId = (ty * BLOCK_SIZE_X + tx) / 32;
    bool useFP64 = (warpId < FP64_WARPS);

    if (row < nj && col < nl) {
        if (useFP64) {
            double f_val = 0.0;
            for (int k_loop = 0; k_loop < nm; ++k_loop) { // nm is common dimension
                f_val += (double)C[row * nm + k_loop] * (double)D[k_loop * nl + col];
            }
            F[row * nl + col] = (DATA_TYPE)f_val;
        } else {
            float f_val = 0.0f;
            for (int k_loop = 0; k_loop < nm; ++k_loop) {
                f_val += (float)C[row * nm + k_loop] * (float)D[k_loop * nl + col];
            }
            F[row * nl + col] = (DATA_TYPE)f_val;
        }
    }
}


__global__ void mm3_kernel3_optimized(int ni, int nj, int nk_d, int nl, int nm_d, DATA_TYPE *E, DATA_TYPE *F, DATA_TYPE *G)
{
    // G(ni,nl) = E(ni,nj) * F(nj,nl)
    int tx = threadIdx.x;    int ty = threadIdx.y;
    int bx = blockIdx.x;    int by = blockIdx.y;

    int row = by * BLOCK_SIZE_Y + ty; // Global row index for G (maps to ni)
    int col = bx * BLOCK_SIZE_X + tx; // Global col index for G (maps to nl)

    int warpId = (ty * BLOCK_SIZE_X + tx) / 32;
    bool useFP64 = (warpId < FP64_WARPS);

    if (row < ni && col < nl) {
        if (useFP64) {
            double g_val = 0.0;
            for (int k_loop = 0; k_loop < nj; ++k_loop) { // nj is common dimension
                g_val += (double)E[row * nj + k_loop] * (double)F[k_loop * nl + col];
            }
            G[row * nl + col] = (DATA_TYPE)g_val;
        } else {
            float g_val = 0.0f;
            for (int k_loop = 0; k_loop < nj; ++k_loop) {
                g_val += (float)E[row * nj + k_loop] * (float)F[k_loop * nl + col];
            }
            G[row * nl + col] = (DATA_TYPE)g_val;
        }
    }
}


/* Main computational kernel on CPU */
void mm3_cpu(int ni, int nj, int nk, int nl, int nm,
        DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk), // A,B,C,D are inputs
        DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
        DATA_TYPE POLYBENCH_2D(C,NJ,NM,nj,nm),
        DATA_TYPE POLYBENCH_2D(D,NM,NL,nm,nl),
        DATA_TYPE POLYBENCH_2D(E,NI,NJ,ni,nj), // E,F are intermediate
        DATA_TYPE POLYBENCH_2D(F,NJ,NL,nj,nl),
        DATA_TYPE POLYBENCH_2D(G,NI,NL,ni,nl)) // G is final output
{
    int i, j, k;

    /* E := A*B */
    for (i = 0; i < ni; i++)
        for (j = 0; j < nj; j++)
        {
            E[i][j] = 0;
            for (k = 0; k < nk; ++k)
                E[i][j] += A[i][k] * B[k][j];
        }

    /* F := C*D */
    for (i = 0; i < nj; i++)
        for (j = 0; j < nl; j++)
        {
            F[i][j] = 0;
            for (k = 0; k < nm; ++k)
                F[i][j] += C[i][k] * D[k][j];
        }

    /* G := E*F */
    for (i = 0; i < ni; i++)
        for (j = 0; j < nl; j++)
        {
            G[i][j] = 0;
            for (k = 0; k < nj; ++k)
                G[i][j] += E[i][k] * F[k][j];
        }
}


void mm3Cuda(int ni, int nj, int nk, int nl, int nm,
        DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk), // Input A
        DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj), // Input B
        DATA_TYPE POLYBENCH_2D(C,NJ,NM,nj,nm), // Input C
        DATA_TYPE POLYBENCH_2D(D,NM,NL,nm,nl), // Input D
        DATA_TYPE POLYBENCH_2D(E_h,NI,NJ,ni,nj), // Host copy for intermediate E (not strictly needed by GPU if E is purely GPU-calculated)
        DATA_TYPE POLYBENCH_2D(F_h,NJ,NL,nj,nl), // Host copy for intermediate F (not strictly needed by GPU if F is purely GPU-calculated)
        DATA_TYPE POLYBENCH_2D(G_h,NI,NL,ni,nl), // Host copy for G (used for CPU verification)
        DATA_TYPE POLYBENCH_2D(G_outputFromGpu,NI,NL,ni,nl)) // Output G (from GPU)
{
    DATA_TYPE *A_gpu;
    DATA_TYPE *B_gpu;
    DATA_TYPE *C_gpu;
    DATA_TYPE *D_gpu;
    DATA_TYPE *E_gpu;
    DATA_TYPE *F_gpu;
    DATA_TYPE *G_gpu;

    cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NI * NK);
    cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * NK * NJ);
    cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * NJ * NM);
    cudaMalloc((void **)&D_gpu, sizeof(DATA_TYPE) * NM * NL);
    cudaMalloc((void **)&E_gpu, sizeof(DATA_TYPE) * NI * NJ);
    cudaMalloc((void **)&F_gpu, sizeof(DATA_TYPE) * NJ * NL);
    cudaMalloc((void **)&G_gpu, sizeof(DATA_TYPE) * NI * NL);

    cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NI * NK, cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * NK * NJ, cudaMemcpyHostToDevice);
    cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * NJ * NM, cudaMemcpyHostToDevice);
    cudaMemcpy(D_gpu, D, sizeof(DATA_TYPE) * NM * NL, cudaMemcpyHostToDevice);
  
    cudaMemcpy(E_gpu, E_h, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyHostToDevice);
    cudaMemcpy(F_gpu, F_h, sizeof(DATA_TYPE) * NJ * NL, cudaMemcpyHostToDevice);
    cudaMemcpy(G_gpu, G_h, sizeof(DATA_TYPE) * NI * NL, cudaMemcpyHostToDevice);


    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    
    dim3 grid1((nj + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (ni + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    
    dim3 grid2((nl + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (nj + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
   
    dim3 grid3((nl + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (ni + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);

    
      polybench_start_instruments;

    
    mm3_kernel1_optimized<<<grid1,block>>>(ni, nj, nk, nl, nm, A_gpu, B_gpu, E_gpu);
    cudaDeviceSynchronize();
    mm3_kernel2_optimized<<<grid2,block>>>(ni, nj, nk, nl, nm, C_gpu, D_gpu, F_gpu);
    cudaDeviceSynchronize(); 
    mm3_kernel3_optimized<<<grid3,block>>>(ni, nj, nk, nl, nm, E_gpu, F_gpu, G_gpu);
    cudaDeviceSynchronize(); 

    printf("GPU Time in seconds:\n");
      polybench_stop_instruments;
     polybench_print_instruments;
    cudaMemcpy(G_outputFromGpu, G_gpu, sizeof(DATA_TYPE) * NI * NL, cudaMemcpyDeviceToHost);

    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(C_gpu);
    cudaFree(D_gpu);
    cudaFree(E_gpu);
    cudaFree(F_gpu);
    cudaFree(G_gpu);
}


static
void print_array(int ni, int nl,
         DATA_TYPE POLYBENCH_2D(G,NI,NL,ni,nl))
{
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
    fprintf (stderr, DATA_PRINTF_MODIFIER, G[i][j]);
    if ((i * ni + j) % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}


int main(int argc, char** argv)
{
    int ni = NI;
    int nj = NJ;
    int nk = NK;
    int nl = NL;
    int nm = NM;

    /* Variable declaration/allocation. */
    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NI, NK, ni, nk);
    POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, NK, NJ, nk, nj);
    POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE, NJ, NM, nj, nm);
    POLYBENCH_2D_ARRAY_DECL(D, DATA_TYPE, NM, NL, nm, nl);
    POLYBENCH_2D_ARRAY_DECL(E, DATA_TYPE, NI, NJ, ni, nj); 
    POLYBENCH_2D_ARRAY_DECL(F, DATA_TYPE, NJ, NL, nj, nl); 
    POLYBENCH_2D_ARRAY_DECL(G, DATA_TYPE, NI, NL, ni, nl); 
    POLYBENCH_2D_ARRAY_DECL(G_outputFromGpu, DATA_TYPE, NI, NL, ni, nl); 

    init_array(ni, nj, nk, nl, nm, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(D));


    GPU_argv_init();

    mm3Cuda(ni, nj, nk, nl, nm,
            POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(D),
            POLYBENCH_ARRAY(E), POLYBENCH_ARRAY(F), POLYBENCH_ARRAY(G), 
            POLYBENCH_ARRAY(G_outputFromGpu));

    #ifdef RUN_ON_CPU
        
          polybench_start_instruments;
        
        mm3_cpu(ni, nj, nk, nl, nm,
                POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(D),
                POLYBENCH_ARRAY(E), POLYBENCH_ARRAY(F), POLYBENCH_ARRAY(G));

        printf("CPU Time in seconds:\n");
          polybench_stop_instruments;
         polybench_print_instruments;

        compareResults(ni, nl, POLYBENCH_ARRAY(G), POLYBENCH_ARRAY(G_outputFromGpu));
    #else
        print_array(ni, nl, POLYBENCH_ARRAY(G_outputFromGpu));
    #endif


    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(B);
    POLYBENCH_FREE_ARRAY(C);
    POLYBENCH_FREE_ARRAY(D);
    POLYBENCH_FREE_ARRAY(E);
    POLYBENCH_FREE_ARRAY(F);
    POLYBENCH_FREE_ARRAY(G);
    POLYBENCH_FREE_ARRAY(G_outputFromGpu);

    return 0;
}

#include "../../common/polybench.c"