#include <stdio.h>

#define TILE_SIZE 16

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use shared memory for tiling
     *
     ********************************************************************/

    /*************************************************************************/
    // INSERT KERNEL CODE HERE
    __shared__ float d_A[TILE_SIZE][TILE_SIZE];
    __shared__ float d_B[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = ty + ( by * TILE_SIZE );
    int col = tx + ( bx * TILE_SIZE );
    float P = 0;

    for(int i=0; i<((k-1)/TILE_SIZE) + 1; i++)
    {
        if (row<m && i * TILE_SIZE + tx<k)
            d_A[ty][tx] = A[tx + (row*k) + (i*TILE_SIZE)];
        else
            d_A[ty][tx] = 0.0;
        if((ty + (TILE_SIZE * i)) < k && col<n)
            d_B[ty][tx] = B[((i * TILE_SIZE+ty)*n) + col];
        else
            d_B[ty][tx] = 0.0;

        __syncthreads();
        if(col<n && row<m)
        {
         for(int k=0; k<TILE_SIZE; ++k)
            P = P + d_A[ty][k] * d_B[k][tx];
        }
        __syncthreads();
    }
    if(row<m && col<n) C[(row*n)+col] = P;      
    /*************************************************************************/
}

void basicSgemm(int m, int n, int k, const float *A, const float *B, float *C)
{
    // Initialize thread block and kernel grid dimensions ---------------------

    const unsigned int BLOCK_SIZE = TILE_SIZE;
    const int Width = 1024;
    /*************************************************************************/
    //INSERT CODE HERE
    dim3 dim_block(BLOCK_SIZE,BLOCK_SIZE);
    dim3 dim_grid((Width / BLOCK_SIZE), (Width / BLOCK_SIZE));
    /*************************************************************************/

    // Invoke CUDA kernel -----------------------------------------------------

    /*************************************************************************/
    //INSERT CODE HERE
    mysgemm<<<dim_grid, dim_block>>>(m,n,k,A,B,C);	
    /*************************************************************************/
}



