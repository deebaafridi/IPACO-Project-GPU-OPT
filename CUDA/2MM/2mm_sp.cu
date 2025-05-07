

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>

#define POLYBENCH_TIME 1

#include "2mm.cuh"
#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"


#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

// Optimization defines
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define WARPS_PER_BLOCK ((BLOCK_SIZE_X * BLOCK_SIZE_Y) / 32)
#define FP64_WARPS (WARPS_PER_BLOCK / 2) // Half the warps for FP64

#define RUN_ON_CPU


void init_array(int ni, int nj, int nk, int nl, DATA_TYPE *alpha, DATA_TYPE *beta,
                DATA_TYPE POLYBENCH_2D(A, NI, NK, ni, nk),
                DATA_TYPE POLYBENCH_2D(B, NK, NJ, nk, nj),
                DATA_TYPE POLYBENCH_2D(C, NJ, NL, nj, nl),
                DATA_TYPE POLYBENCH_2D(D, NI, NL, ni, nl))
{
    int i, j;
    *alpha = (DATA_TYPE)32412.0;
    *beta = (DATA_TYPE)2123.0;

    for (i = 0; i < ni; i++)
        for (j = 0; j < nk; j++)
            A[i][j] = ((DATA_TYPE) i*j) / NI;

    for (i = 0; i < nk; i++)
        for (j = 0; j < nj; j++)
            B[i][j] = ((DATA_TYPE) i*(j+1)) / NJ;

    for (i = 0; i < nj; i++)
        for (j = 0; j < nl; j++)
            C[i][j] = ((DATA_TYPE) i*(j+3)) / NJ;

    for (i = 0; i < ni; i++)
        for (j = 0; j < nl; j++)
            D[i][j] = ((DATA_TYPE) i*(j+2)) / NK;
}


void compareResults(int ni, int nl, DATA_TYPE POLYBENCH_2D(D_cpu, NI, NL, ni, nl), DATA_TYPE POLYBENCH_2D(D_gpu, NI, NL, ni, nl))
{
    int i,j,fail;
    fail = 0;
    for (i=0; i < ni; i++)
        for (j=0; j < nl; j++)
            if (percentDiff(D_cpu[i][j], D_gpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
                fail++;
    printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void GPU_argv_init()
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
    printf("Setting device %d: %s\n",GPU_DEVICE,deviceProp.name);
    cudaSetDevice( GPU_DEVICE );
}

__global__ void mm2_kernel1_optimized(int ni_val, int nj_val, int nk_val,
                                    DATA_TYPE alpha_val,
                                    const DATA_TYPE *A, const DATA_TYPE *B, DATA_TYPE *tmp)
{
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    int row = by * BLOCK_SIZE_Y + ty;
    int col = bx * BLOCK_SIZE_X + tx;
    int warpId = (ty * BLOCK_SIZE_X + tx) / 32;
    bool useFP64 = (warpId < FP64_WARPS);

    if (row < ni_val && col < nj_val) {
        if (useFP64) {
            double acc = 0.0;
            for (int k_loop = 0; k_loop < nk_val; ++k_loop)
                acc += (double)A[row * nk_val + k_loop] * (double)B[k_loop * nj_val + col];
            tmp[row * nj_val + col] = (DATA_TYPE)((double)alpha_val * acc);
        } else {
            float acc = 0.0f;
            for (int k_loop = 0; k_loop < nk_val; ++k_loop)
                acc += (float)A[row * nk_val + k_loop] * (float)B[k_loop * nj_val + col];
            tmp[row * nj_val + col] = (DATA_TYPE)((float)alpha_val * acc);
        }
    }
}

__global__ void mm2_kernel2_optimized(int ni_val, int nj_val, int nl_val,
                                    DATA_TYPE beta_val,
                                    const DATA_TYPE *tmp, const DATA_TYPE *C, DATA_TYPE *D)
{
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    int row = by * BLOCK_SIZE_Y + ty;
    int col = bx * BLOCK_SIZE_X + tx;
    int warpId = (ty * BLOCK_SIZE_X + tx) / 32;
    bool useFP64 = (warpId < FP64_WARPS);

    if (row < ni_val && col < nl_val) {
        if (useFP64) {
            double acc = 0.0;
            double d_initial = (double)D[row * nl_val + col]; // Read initial D value
            for (int k_loop = 0; k_loop < nj_val; ++k_loop)
                acc += (double)tmp[row * nj_val + k_loop] * (double)C[k_loop * nl_val + col];
            D[row * nl_val + col] = (DATA_TYPE)((double)beta_val * d_initial + acc);
        } else {
            float acc = 0.0f;
            float d_initial = (float)D[row * nl_val + col]; // Read initial D value
            for (int k_loop = 0; k_loop < nj_val; ++k_loop)
                acc += (float)tmp[row * nj_val + k_loop] * (float)C[k_loop * nl_val + col];
            D[row * nl_val + col] = (DATA_TYPE)((float)beta_val * d_initial + acc);
        }
    }
}

void mm2_cpu(int ni, int nj, int nk, int nl,
        DATA_TYPE alpha, DATA_TYPE beta,
        DATA_TYPE POLYBENCH_2D(tmp,NI,NJ,ni,nj),
        DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
        DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
        DATA_TYPE POLYBENCH_2D(C,NJ,NL,nj,nl),
        DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
    int i, j, k_inner;
    // tmp = alpha * A * B
    for (i = 0; i < ni; i++)
        for (j = 0; j < nj; j++) {
            tmp[i][j] = (DATA_TYPE)0.0; // Initialize tmp element
            for (k_inner = 0; k_inner < nk; ++k_inner)
                tmp[i][j] += alpha * A[i][k_inner] * B[k_inner][j];
        }
    // D = beta * D + tmp * C
    for (i = 0; i < ni; i++)
        for (j = 0; j < nl; j++) {
            D[i][j] *= beta; // Apply beta to initial D
            for (k_inner = 0; k_inner < nj; ++k_inner)
                D[i][j] += tmp[i][k_inner] * C[k_inner][j];
        }
}

static void print_array(int ni, int nl,
         DATA_TYPE POLYBENCH_2D(D_to_print,NI,NL,ni,nl))
{
  int i, j;
  #ifdef POLYBENCH_DUMP_ARRAYS
  fprintf (stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf (stderr, "begin dump: %s", "D");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
        if ((i * ni + j) % 20 == 0) fprintf (stderr, "\n");
        fprintf (stderr, DATA_PRINTF_MODIFIER, D_to_print[i][j]);
    }
  fprintf (stderr, "\nend   dump: %s\n", "D");
  fprintf (stderr, "==END   DUMP_ARRAYS==\n");
  #else
 
  if (ni > 0 && nl > 0 && (long long)ni * nl <= 100) {
       for (i = 0; i < ni; i++) {
           for (j = 0; j < nl; j++) {
               fprintf (stderr, DATA_PRINTF_MODIFIER, D_to_print[i][j]);
           }
           fprintf (stderr, "\n");
       }
       fprintf (stderr, "\n");
  } else if (ni > 0 && nl > 0) { // Added check for positive ni, nl
       fprintf (stderr, "INFO: Array D is large (%d x %d), not printing all elements to stderr.\n", ni, nl);
  }
  #endif
}

void mm2Cuda(int ni_runtime, int nj_runtime, int nk_runtime, int nl_runtime,
            DATA_TYPE alpha, DATA_TYPE beta,
            DATA_TYPE POLYBENCH_2D(tmp_host, NI,NJ, ni_runtime, nj_runtime),
            DATA_TYPE POLYBENCH_2D(A_host, NI,NK, ni_runtime, nk_runtime),
            DATA_TYPE POLYBENCH_2D(B_host, NK,NJ, nk_runtime, nj_runtime),
            DATA_TYPE POLYBENCH_2D(C_host, NJ,NL, nj_runtime, nl_runtime),
            DATA_TYPE POLYBENCH_2D(D_host, NI,NL, ni_runtime, nl_runtime),
            DATA_TYPE POLYBENCH_2D(D_outputFromGpu, NI,NL, ni_runtime, nl_runtime))
{
    DATA_TYPE *tmp_gpu, *A_gpu, *B_gpu, *C_gpu, *D_gpu;

    cudaMalloc((void **)&tmp_gpu, sizeof(DATA_TYPE) * (size_t)NI * NJ);
    cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * (size_t)NI * NK);
    cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * (size_t)NK * NJ);
    cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * (size_t)NJ * NL);
    cudaMalloc((void **)&D_gpu, sizeof(DATA_TYPE) * (size_t)NI * NL);

    cudaMemcpy(A_gpu, A_host, sizeof(DATA_TYPE) * (size_t)NI * NK, cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B_host, sizeof(DATA_TYPE) * (size_t)NK * NJ, cudaMemcpyHostToDevice);
    cudaMemcpy(C_gpu, C_host, sizeof(DATA_TYPE) * (size_t)NJ * NL, cudaMemcpyHostToDevice);
    cudaMemcpy(D_gpu, D_host, sizeof(DATA_TYPE) * (size_t)NI * NL, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridDim1((nj_runtime + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (ni_runtime + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    dim3 gridDim2((nl_runtime + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (ni_runtime + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);

      polybench_start_instruments;
    mm2_kernel1_optimized<<<gridDim1, blockDim>>>(ni_runtime, nj_runtime, nk_runtime, alpha, A_gpu, B_gpu, tmp_gpu);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { fprintf(stderr, "CUDA Error K1L: %s\n", cudaGetErrorString(err)); exit(1); }
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) { fprintf(stderr, "CUDA Error K1S: %s\n", cudaGetErrorString(err)); exit(1); }

    mm2_kernel2_optimized<<<gridDim2, blockDim>>>(ni_runtime, nj_runtime, nl_runtime, beta, tmp_gpu, C_gpu, D_gpu);
    err = cudaGetLastError();
    if (err != cudaSuccess) { fprintf(stderr, "CUDA Error K2L: %s\n", cudaGetErrorString(err)); exit(1); }
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) { fprintf(stderr, "CUDA Error K2S: %s\n", cudaGetErrorString(err)); exit(1); }

    printf("GPU Time in seconds:\n");
      polybench_stop_instruments;
     polybench_print_instruments;
    cudaMemcpy(D_outputFromGpu, D_gpu, sizeof(DATA_TYPE) * (size_t)NI * NL, cudaMemcpyDeviceToHost);

    cudaFree(tmp_gpu); cudaFree(A_gpu); cudaFree(B_gpu); cudaFree(C_gpu); cudaFree(D_gpu);
}

int main(int argc, char** argv)
{
    int ni, nj, nk, nl;

    // Initialize with default values from defines
    ni = NI;
    nj = NJ;
    nk = NK;
    nl = NL;

    // Parse command-line arguments for problem sizes
    if (argc > 1) {
        int arg_idx = 1;
        if (arg_idx < argc) {
            int parsed_val = atoi(argv[arg_idx++]);
            if (parsed_val > 0) ni = parsed_val;
        }
        if (arg_idx < argc) {
            int parsed_val = atoi(argv[arg_idx++]);
            if (parsed_val > 0) nj = parsed_val;
        }
        if (arg_idx < argc) {
            int parsed_val = atoi(argv[arg_idx++]);
            if (parsed_val > 0) nk = parsed_val;
        }
        if (arg_idx < argc) {
            int parsed_val = atoi(argv[arg_idx++]);
            if (parsed_val > 0) nl = parsed_val;
        }
    }
   
    if (ni > NI) ni = NI;
    if (nj > NJ) nj = NJ;
    if (nk > NK) nk = NK;
    if (nl > NL) nl = NL;


    DATA_TYPE alpha, beta;
    POLYBENCH_2D_ARRAY_DECL(h_tmp, DATA_TYPE, NI, NJ, ni, nj);
    POLYBENCH_2D_ARRAY_DECL(h_A,   DATA_TYPE, NI, NK, ni, nk);
    POLYBENCH_2D_ARRAY_DECL(h_B,   DATA_TYPE, NK, NJ, nk, nj);
    POLYBENCH_2D_ARRAY_DECL(h_C,   DATA_TYPE, NJ, NL, nj, nl);
    POLYBENCH_2D_ARRAY_DECL(h_D_cpu, DATA_TYPE, NI, NL, ni, nl);
    POLYBENCH_2D_ARRAY_DECL(h_D_gpu, DATA_TYPE, NI, NL, ni, nl);

      init_array(ni, nj, nk, nl, &alpha, &beta, POLYBENCH_ARRAY(h_A),
               POLYBENCH_ARRAY(h_B), POLYBENCH_ARRAY(h_C), POLYBENCH_ARRAY(h_D_cpu));
    GPU_argv_init();
    mm2Cuda(ni, nj, nk, nl, alpha, beta, POLYBENCH_ARRAY(h_tmp), POLYBENCH_ARRAY(h_A),
            POLYBENCH_ARRAY(h_B), POLYBENCH_ARRAY(h_C), POLYBENCH_ARRAY(h_D_cpu), POLYBENCH_ARRAY(h_D_gpu));

    #ifdef RUN_ON_CPU
          polybench_start_instruments;
        mm2_cpu(ni, nj, nk, nl, alpha, beta, POLYBENCH_ARRAY(h_tmp), POLYBENCH_ARRAY(h_A),
                POLYBENCH_ARRAY(h_B), POLYBENCH_ARRAY(h_C), POLYBENCH_ARRAY(h_D_cpu));
        printf("CPU Time in seconds:\n");
          polybench_stop_instruments;
         polybench_print_instruments;
        compareResults(ni, nl, POLYBENCH_ARRAY(h_D_cpu), POLYBENCH_ARRAY(h_D_gpu));
    #else
        print_array(ni, nl, POLYBENCH_ARRAY(h_D_gpu));
    #endif

    POLYBENCH_FREE_ARRAY(h_tmp); POLYBENCH_FREE_ARRAY(h_A); POLYBENCH_FREE_ARRAY(h_B);
    POLYBENCH_FREE_ARRAY(h_C); POLYBENCH_FREE_ARRAY(h_D_cpu); POLYBENCH_FREE_ARRAY(h_D_gpu);
      return 0;
}

#ifndef POLYBENCH_FPGA_DRIVER
#include "../../common/polybench.c"
#endif