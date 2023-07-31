#include <stdio.h>
#include <stdlib.h>
#include "kernel.cu"
#include "support.h"

const unsigned int n = 4;

int main (int argc, char *argv[])
{

    Timer timer;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    float *A_u, *B_u, *C_u;

    size_t A_sz, B_sz, C_sz;
    unsigned matArow, matAcol;
    unsigned matBrow, matBcol;
    dim3 dim_grid, dim_block;




    cudaStream_t stream[n];
    for (int i = 0; i < n; i++)
        cudaStreamCreate(&stream[i]);


    if (argc == 1) {
        matArow = 1000;
        matAcol = matBrow = 1000;
        matBcol = 1000;
    } else if (argc == 2) {
        matArow = atoi(argv[1]);
        matAcol = matBrow = atoi(argv[1]);
        matBcol = atoi(argv[1]);
    } else if (argc == 4) {
        matArow = atoi(argv[1]);
        matAcol = matBrow = atoi(argv[2]);
        matBcol = atoi(argv[3]);
    } else {
        printf("\n    Invalid input parameters!"
      "\n    Usage: ./sgemm-tiled                # All matrices are 1000 x 1000"
      "\n    Usage: ./sgemm-tiled <m>            # All matrices are m x m"
      "\n    Usage: ./sgemm-tiled <m> <k> <n>    # A: m x k, B: k x n, C: m x n"
      "\n");
        exit(0);
    }
   
    A_sz = matArow*matAcol;
    B_sz = matBrow*matBcol;
    C_sz = matArow*matBcol;

    const int segmentLen = (matArow*matAcol)/ n;


    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    A: %u x %u\n    B: %u x %u\n    C: %u x %u\n", matArow, matAcol,
        matBrow, matBcol, matArow, matBcol);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    /*************************************************************************/
    //INSERT CODE HERE

    cudaMallocManaged(&A_u, sizeof(float) * A_sz);
    for (unsigned int i=0; i < A_sz; i++) { A_u[i] = (rand()%100)/100.00; }
    cudaMallocManaged(&B_u, sizeof(float) * B_sz);
    for (unsigned int i=0; i < B_sz; i++) { B_u[i] = (rand()%100)/100.00; }

    cudaMallocManaged(&C_u, sizeof(float) * C_sz);



    /*************************************************************************/
	
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------
    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);
	
    /*************************************************************************/

    //INSERT CODE HERE
    for (int i = 0; i < n; i++)
    {   
        int device = -1;
        cudaGetDevice(&device);
        int Offset = i * segmentLen;
        cudaMemPrefetchAsync(A_u, sizeof(float) * segmentLen, device, stream[i]);
        cudaMemPrefetchAsync(B_u, sizeof(float) * B_sz, device, stream[i]);
        cudaMemPrefetchAsync(C_u, sizeof(float) * segmentLen, device, stream[i]);
            
        basicSgemmStream(matArow,matArow,matArow, &A_u[Offset], B_u, &C_u[Offset], stream[i]);

    }

    /*************************************************************************/
    

    // Launch kernel using standard sgemm interface ---------------------------
    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);



    // Copy device variables from host ----------------------------------------
    printf("Copying data from device to host..."); fflush(stdout);

    /*************************************************************************/
    //INSERT CODE HERE	

    /*************************************************************************/

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Verify correctness -----------------------------------------------------

    printf("Verifying results..."); fflush(stdout);


    verify(A_u, B_u, C_u, matArow, matAcol, matBcol);


    // Free memory ------------------------------------------------------------

    /*************************************************************************/
    //INSERT CODE HERE
    cudaFree(A_u);
    cudaFree(B_u);
    cudaFree(C_u);
    for (int i = 0; i < n; i++)
    {
        cudaStreamDestroy(stream[i]);
    }
    /*************************************************************************/
;
    return 0;
}

