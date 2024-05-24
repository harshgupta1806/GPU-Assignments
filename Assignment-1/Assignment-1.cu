#include <chrono>
#include <fstream>
#include <iostream>
#include <cuda.h>

using std::cin;
using std::cout;



// Task 1
__global__
void CalculateHadamardProduct(long int* A, long int* B, int N) {

    // TODO: Write your kernel here
    long long unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N*N)
    {
      unsigned int col = id% N;  
      unsigned int row = id / N;
      unsigned int IDA = N * row + col;
      unsigned int IDB = N * col + row;
      A[IDA] = A[IDA] * B[IDB];
    }

}

// Task 2
__global__
void FindWeightMatrix(long int* A, long int* B, int N) {
    long long unsigned id = blockIdx.x * blockDim.x * blockDim.y + threadIdx.x * blockDim.x + threadIdx.y;
     if(id < N*N)
    {
      if(A[id] < B[id])
        A[id] = B[id];
    }

}

//Task 3
__global__
void CalculateFinalMatrix(long int* A, long int* B, int N) {

    // TODO: Write your kernel here
      unsigned int blockNo = gridDim.x * blockIdx.x + blockIdx.y;
      unsigned int noOfThreadsCrossed = blockNo * (blockDim.x * blockDim.y);
      unsigned int currThreadInBlock = noOfThreadsCrossed +  (blockDim.x * threadIdx.x + threadIdx.y);
      unsigned int id = currThreadInBlock; // Unique ID

    if(id < 4*N*N)
    {
      unsigned int col = id% (2*N);
      unsigned int row = id / (2*N);
      // printf("%d %d %d\t\t",id, row, col);
      unsigned int row_ = row%N;
      unsigned int col_ = col%N;
      unsigned int id1 = row_ * N + col_;
      B[id] = B[id] * A[id1];
    }
}


int main(int argc, char** argv) {

    // cout << "Enter N :: ";
    int N;
    cin >> N;
    long int* A = new long int[N * N];
    long int* B = new long int[N * N];
    long int* C = new long int[N * N];
    long int* D = new long int[2 * N * 2 * N];


    for (long int i = 0; i < N * N; i++) {
        cin >> A[i];
    }

    for (long int i = 0; i < N * N; i++) {
        cin >> B[i];
    }

    for (long int i = 0; i < N * N; i++) {
        cin >> C[i];
    }

    for (long int i = 0; i < 2 * N * 2 * N; i++) {
        cin >> D[i];
    }



    long int* d_A;
    long int* d_B;
    long int* d_C;
    long int* d_D;

    dim3 threadsPerBlock(1024, 1, 1);
    dim3 blocksPerGrid(ceil(N * N / 1024.0), 1, 1);

    cudaMalloc(&d_A, sizeof(long int) * N*N); // Allocating memory on GPU for Array A
    cudaMalloc(&d_B, sizeof(long int) * N*N); // Allocating memory on GPU for Array B
 
    cudaMemcpy(d_A, A, sizeof(long int) * N*N, cudaMemcpyHostToDevice); // Copying array A on GPU
    cudaMemcpy(d_B, B, sizeof(long int) * N*N, cudaMemcpyHostToDevice); // Copying array B on GPU

    auto start = std::chrono::high_resolution_clock::now();
    CalculateHadamardProduct<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, N);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end - start;

    cudaFree(d_B); // Memory Free from GPU for Array B




    threadsPerBlock = dim3(32, 32, 1);
    blocksPerGrid = dim3(ceil(N * N / 1024.0), 1, 1);


    cudaMalloc(&d_C, sizeof(long int) * N*N); // Allocating memory on GPU for Array C
    cudaMemcpy(d_C, C, sizeof(long int) * N*N, cudaMemcpyHostToDevice); // Copying Array C on GPU

    start = std::chrono::high_resolution_clock::now();
    FindWeightMatrix<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, N); 
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed2 = end - start;

    cudaFree(d_C); // Memory Free for Array C on GPU
// Task 3
    cudaMalloc(&d_D, sizeof(long int) *  4*N*N); // Allocating memory on GPU for Array C
    cudaMemcpy(d_D, D, sizeof(long int) * 4*N*N, cudaMemcpyHostToDevice); // Copying Array C on GPU

    threadsPerBlock = dim3(32, 32, 1);
    blocksPerGrid = dim3(ceil(2 * N / 32.0), ceil(2 * N / 32.0), 1);

    start = std::chrono::high_resolution_clock::now();
    CalculateFinalMatrix<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_D, N);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed3 = end - start;
    cudaMemcpy(A, d_A, sizeof(long int) * N*N, cudaMemcpyDeviceToHost);

    cudaMemcpy(D, d_D, 2 * N * 2 * N * sizeof(long int), cudaMemcpyDeviceToHost);

    cudaFree(d_A); // Memory Free
    cudaFree(d_D); // Memory Free 

    std::ofstream file("cuda.out");
     if (file.is_open()) {
        for (long int i = 0; i < 2 * N; i++) {
            for (long int j = 0; j < 2 * N; j++) {
                file << D[i * 2 * N + j] << " ";
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
        file2 << elapsed2.count() << "\n";
        file2 << elapsed3.count() << "\n";
        file2.close();
    } else {
        std::cout << "Unable to open file";
    }

    return 0;
}