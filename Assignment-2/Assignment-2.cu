#include <chrono>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cuda.h>

using namespace std;

using std::cin;
using std::cout;

typedef long long ll;

// CUDA kernel for convolution
__global__ void dkernel(long int *gpu_input, long int *gpu_output, long int *gpu_filter, int k, int m, int n)
{
    extern __shared__ long int filter[]; 

    
    unsigned long int id = threadIdx.x;  // calculating id within a block
    unsigned long int global_id = blockIdx.x * blockDim.x + threadIdx.x; // calculating global id for a thread

    // Load the entire filter into shared memory
    while (id < k * k)
    {
        filter[id] = gpu_filter[id];
        id += blockDim.x; 
    }
    __syncthreads();  //synchronization of threads within block

    int i = global_id/n; // calcuating row of matrix on which convolution to be performed
    int j = global_id%n; // // calcuating column of matrix on which convolution to be performed

    //biasRow and biasCol is used to map filter to input matrix
    int biasRow = -(i - k / 2); // 
    int biasCol = -(j - k / 2);
    

        
    // Compute convolution for each element
    long int res = 0;
    for (int a = i - k / 2; a <= i + k / 2; a++)
    {
        for (int b = j - k / 2; b <= j + k / 2; b++)
        {
            int row = a + biasRow;  // calculating row of filter matrix
            int col = b + biasCol; // calculating col of filter matrix
            if (a >= 0 && b >= 0 && a < m && b < n)
            {
                res = res + gpu_input[a * n + b] * filter[row * k + col];  // computing res
            }
        }
    }
    gpu_output[global_id] = res; // adding result in gpu_output
}


int main(int argc, char** argv) {

    int m,n,k;
    cin>>m>>n>>k;


    long int* h_mat = new long int[m * n];
    long int* h_filter = new long int[k * k];

    long int* h_ans = new long int[m * n];


    for (long int i = 0; i < m * n; i++) {
        cin>>h_mat[i];
    }

    for (long int i = 0; i < k * k; i++) {
        cin>>h_filter[i];
    }

    /**
     *
     * DO NOT CHANGE ANYTHING ABOVE THIS LINE
     *
    **/

    /****************************************************Start Here***********************************************************/
    long int *gpu_mat;  // declaring array on gpu for input matrix
    long int *gpu_filter; // declaring array on gpu for filter matrix 
    long int *gpu_output; // declaring array on gpu which is used to store output

    // Memory creation and copying on gpu for input matrix
    cudaMalloc(&gpu_mat, m*n*sizeof(long int));
    cudaMemcpy(gpu_mat, h_mat, m*n*sizeof(long int), cudaMemcpyHostToDevice);

    // Memory creation on gpu for output matrix
    cudaMalloc(&gpu_output, m*n*sizeof(long int));

    // Memory creation and copying on gpu for filter matrix
    cudaMalloc(&gpu_filter, k*k*sizeof(long int));
    cudaMemcpy(gpu_filter, h_filter, k * k * sizeof(long int), cudaMemcpyHostToDevice);

    int threads = 1024; // No. of threads in one block

    int blocks = ceil((float)(m * n)/threads); // calculating number of blocks

    int sharedMemSize = k * k * sizeof(long int);  // Shared memory size for filter


    auto start = std::chrono::high_resolution_clock::now();//keep it just before the kernel launch

    // Launch kernel
    dkernel<<<blocks, 1024, sharedMemSize>>>(gpu_mat, gpu_output, gpu_filter, k, m, n);
    cudaDeviceSynchronize(); 
    
    auto end = std::chrono::high_resolution_clock::now();//keep it just after the kernel launch

    // Copying output matrix from gpu to cpu 
    cudaMemcpy(h_ans, gpu_output, m*n*sizeof(long int), cudaMemcpyDeviceToHost); 

    // Free Memory from GPU
    cudaFree(gpu_mat);
    cudaFree(gpu_filter);
    cudaFree(gpu_output);
    
    /*$$$$$$$$$$$$$$$$$$$$$$$$Make sure your final output from the device is stored in h_ans.$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
    std::chrono::duration<double> elapsed1 = end - start;
    /**
     *
     * DO NOT CHANGE ANYTHING BELOW THIS LINE
     *
  */



    std::ofstream file("cuda.out");
    if (file.is_open()) {
        for (long int i = 0; i < m; i++) {
            for (long int j = 0; j < n; j++) {
                file << h_ans[i * n + j] << " ";
            }
            file << "\n";
        }
        file.close();
    } else {
        std::cout << "Unable to open file";
    }

    std::ofstream file2("cuda_timing.out");
    if(file2.is_open()) {
        file2 << elapsed1.count() << "\n";
        file2.close();
    } else {
        std::cout << "Unable to open file";
    }

    return 0;
}
/*
3 4 3
1 2 3 4
5 6 7 8
9 10 11 12
5 16 1
0 4 2
19 15 8

131 255 303 269
265 487 557 500
142 190 218 211

*/