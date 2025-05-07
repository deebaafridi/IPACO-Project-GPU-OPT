#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda.h>

#define POLYBENCH_TIME 1

#include "gemm.cuh"
#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"

#define GPU_DEVICE 0

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0) */
#define ALPHA 32412.0f
#define BETA 2123.0f

#define RUN_ON_CPU

static const int RESOURCE_PER_SM = 4;

void gemm(int ni, int nj, int nk, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A, NI, NK, ni, nk),
          DATA_TYPE POLYBENCH_2D(B, NK, NJ, nk, nj), DATA_TYPE POLYBENCH_2D(C, NI, NJ, ni, nj))
{
    int i, j, k;

    for (i = 0; i < _PB_NI; i++)
    {
        for (j = 0; j < _PB_NJ; j++)
        {
            C[i][j] *= beta;
            for (k = 0; k < _PB_NK; ++k)
            {
                C[i][j] += alpha * A[i][k] * B[k][j];
            }
        }
    }
}

void init(int ni, int nj, int nk, DATA_TYPE *alpha, DATA_TYPE *beta, DATA_TYPE POLYBENCH_2D(A, NI, NK, ni, nk),
          DATA_TYPE POLYBENCH_2D(B, NK, NJ, nk, nj), DATA_TYPE POLYBENCH_2D(C, NI, NJ, ni, nj))
{
    int i, j;

    *alpha = ALPHA;
    *beta = BETA;

    for (i = 0; i < ni; i++)
    {
        for (j = 0; j < nk; j++)
        {
            A[i][j] = ((DATA_TYPE)i * j) / NI;
        }
    }

    for (i = 0; i < nk; i++)
    {
        for (j = 0; j < nj; j++)
        {
            B[i][j] = ((DATA_TYPE)i * j) / NI;
        }
    }

    for (i = 0; i < ni; i++)
    {
        for (j = 0; j < nj; j++)
        {
            C[i][j] = ((DATA_TYPE)i * j) / NI;
        }
    }
}

void compareResults(int ni, int nj, DATA_TYPE POLYBENCH_2D(C, NI, NJ, ni, nj), DATA_TYPE POLYBENCH_2D(C_outputFromGpu, NI, NJ, ni, nj))
{
    int i, j, fail = 0;

    for (i = 0; i < ni; i++)
    {
        for (j = 0; j < nj; j++)
        {
            if (percentDiff(C[i][j], C_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
            {
                fail++;
            }
        }
    }

    printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void GPU_argv_init()
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
    printf("setting device %d with name %s\n", GPU_DEVICE, deviceProp.name);
    cudaSetDevice(GPU_DEVICE);
}

__global__ void gemm_kernel(int ni, int nj, int nk,
                            DATA_TYPE alpha, DATA_TYPE beta,
                            DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C)
{
    
    //Declaring a varibales in shared memory which are common for a Threadblock
    __shared__ int fp32CountInSM;     
    __shared__ int fp64CountInSM;

    if (threadIdx.x == 0 && threadIdx.y == 0)  //initialisation of shared memory
    {
        fp32CountInSM = 0;
        fp64CountInSM = 0;
    }
    __syncthreads();  // Ensure all threads see initialized values

    //compute row and column that is to be computed by the thread

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    //if row and column are greater than max_rows and max_cols then return
    if (row >= ni || col >= nj)
        return;

    unsigned threadIdInWarp = threadIdx.x % 32;  //computing thread ID within a warp
    bool usingFP32Unit = false; // Assume initially that this warp will not use FP32 units

    if (threadIdInWarp == 0)  //only first thread in warp will decide which compute unit to use.
    {
        bool occupyResourceUnit = false;
        while (!occupyResourceUnit)
        {
            int currentStatusOfFP32 = atomicAdd(&fp32CountInSM, 0);  //retrive the current value of fp32Count 
            if (currentStatusOfFP32 < RESOURCE_PER_SM)   // if the FP32 resources utilization is less than RESOURCE_PER_SM(4) then the warp will take a decision to use FP32 units
            {
                if (atomicCAS(&fp32CountInSM, currentStatusOfFP32, currentStatusOfFP32 + 1) == currentStatusOfFP32) //increment the FP32 usage count
                {
                    usingFP32Unit = true;   // Successfully reserved an FP32 unit
                    occupyResourceUnit = true;
                    break;
                }
            }
            int currentStatusOfFP64 = atomicAdd(&fp64CountInSM, 0); //if FP32 unit is not available then use FP64 units
            if (currentStatusOfFP64 < RESOURCE_PER_SM)
            {
                if (atomicCAS(&fp64CountInSM, currentStatusOfFP64, currentStatusOfFP64 + 1) == currentStatusOfFP64) //increment the FP64 usage
                {
                    usingFP32Unit = false; // Successfully reserved an FP64 unit
                    occupyResourceUnit = true;
                    break;
                }
            }
        }
    }
    __syncwarp(); // warp sync

    bool warpFP32Choice = __shfl_sync(0xFFFFFFFF, usingFP32Unit, 0); //allow all threads in warp to know the decision of first thread in warp
    
    double product_d = 0.0;
    float product_f = 0.0f;
    int idx = row * nj + col;
    
    if (warpFP32Choice)   // Perform matrix multiplication using either FP32 or FP64 units based on warp decision
    {
        for (int k = 0; k < nk; ++k)
        {
            float a = A[row * nk + k];
            float b = B[k * nj + col];
            product_f += alpha * a * b;
        }
        C[idx] = beta * C[idx] + product_f;
    }
    else
    {
        for (int k = 0; k < nk; ++k)
        {
            float a = A[row * nk + k];
            float b = B[k * nj + col];
            product_d += static_cast<double>(alpha) * a * b;  //cast data to double so FP64 units are utilized
        }
        C[idx] = beta * C[idx] + static_cast<float>(product_d); // Cast result back to float before storing
    }
    __syncwarp();

    if (threadIdInWarp == 0)  //decrement the usage of FP32 and FP64 units so other warps can utilize.
    {
        if (warpFP32Choice)
        {
            atomicSub(&fp32CountInSM, 1);
        }
        else
        {
            atomicSub(&fp64CountInSM, 1);
        }
    }
}

void gemmCuda(int ni, int nj, int nk,
              DATA_TYPE alpha, DATA_TYPE beta,
              DATA_TYPE POLYBENCH_2D(A, NI, NK, ni, nk),
              DATA_TYPE POLYBENCH_2D(B, NK, NJ, nk, nj),
              DATA_TYPE POLYBENCH_2D(C, NI, NJ, ni, nj),
              DATA_TYPE POLYBENCH_2D(C_outputFromGpu, NI, NJ, ni, nj))
{
    DATA_TYPE *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cudaMalloc(&dA, sizeof(DATA_TYPE) * NI * NK);
    cudaMalloc(&dB, sizeof(DATA_TYPE) * NK * NJ);
    cudaMalloc(&dC, sizeof(DATA_TYPE) * NI * NJ);

    cudaMemcpy(dA, A, sizeof(DATA_TYPE) * NI * NK, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(DATA_TYPE) * NK * NJ, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyHostToDevice);

    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    dim3 grid((ni + block.x - 1) / block.x,
              (nj + block.y - 1) / block.y);

    polybench_start_instruments;
    gemm_kernel<<<grid, block>>>(ni, nj, nk, alpha, beta, dA, dB, dC);
    cudaDeviceSynchronize();

    polybench_stop_instruments;
    printf("GPU Time in seconds:\n");
    polybench_print_instruments;

    cudaMemcpy(C_outputFromGpu, dC, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

/* DCE code. Must scan the entire live-out data. */
static void print_array(int ni, int nj,
                        DATA_TYPE POLYBENCH_2D(C, NI, NJ, ni, nj))
{
    int i, j;
    for (i = 0; i < ni; i++)
        for (j = 0; j < nj; j++)
        {
            fprintf(stderr, DATA_PRINTF_MODIFIER, C[i][j]);
            if ((i * ni + j) % 20 == 0)
                fprintf(stderr, "\n");
        }
    fprintf(stderr, "\n");
}

int main(int argc, char *argv[])
{
    /* Retrieve problem size. */
    int ni = NI;
    int nj = NJ;
    int nk = NK;

    /* Variable declaration/allocation. */
    DATA_TYPE alpha, beta;
    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NI, NK, ni, nk);
    POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, NK, NJ, nk, nj);
    POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE, NI, NJ, ni, nj);
    POLYBENCH_2D_ARRAY_DECL(C_outputFromGpu, DATA_TYPE, NI, NJ, ni, nj);

    init(ni, nj, nk, &alpha, &beta,
         POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C));

    GPU_argv_init();

    gemmCuda(ni, nj, nk, alpha, beta,
             POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(C_outputFromGpu));

#ifdef RUN_ON_CPU

    /* Start timer. */
    polybench_start_instruments;

    gemm(ni, nj, nk, alpha, beta,
         POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C));

    /* Stop and print timer. */
    printf("CPU Time in seconds:\n");
    polybench_stop_instruments;
    polybench_print_instruments;

    compareResults(ni, nj,
                   POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(C_outputFromGpu));
#else
    print_array(ni, nj, POLYBENCH_ARRAY(C_outputFromGpu));
#endif

    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(B);
    POLYBENCH_FREE_ARRAY(C);
    POLYBENCH_FREE_ARRAY(C_outputFromGpu);

    return 0;
}

#include "../../common/polybench.c"