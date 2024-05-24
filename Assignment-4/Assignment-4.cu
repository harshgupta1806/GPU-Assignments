#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <chrono>
#include <thrust/device_vector.h>
#include <thrust/fill.h>

using namespace std;

//*******************************************

// Write down the kernels here


__device__ unsigned rounds = 1;

// Kernel that handle fire for single round
__global__ void fire(int *x, int *y, unsigned *score, int *power, unsigned *is_Dead, int T)
{
    __shared__ int lockVar;          // Variable that handle lock
    __shared__ volatile int min_dis; // variable that handle minimum distance of a tank
    __shared__ int target;           // target tank number
    __shared__ volatile int temp;    // true target
    __shared__ long long int xtarget;          // x-cord of target tank
    __shared__ long long int ytarget;          // y-cord of target tank

    // Initialize the shared variables
    if (threadIdx.x == 0)
    {
        lockVar = 0;
        min_dis = INT_MAX;
        temp = -1;
        target = (blockIdx.x + rounds) % T;
        xtarget = x[target];
        ytarget = y[target];
    }

    __syncthreads(); // Syncthreads with in blocks to ensure shared variables are initialized properly.

    // Cordinates of tank that is firing
    long long int xsrc = x[blockIdx.x];
    long long int ysrc = y[blockIdx.x];

    if (threadIdx.x != blockIdx.x && !is_Dead[blockIdx.x] && !is_Dead[threadIdx.x]) // checking condition (firing tank != target tank && firing tank must be alive)
    {
        long long int xtank = x[threadIdx.x];
        long long int ytank = y[threadIdx.x];

        // Check tank is in firing line or not   
        if(((xsrc - xtarget) * (ytank - ysrc)) == ((ysrc - ytarget) * (xtank - xsrc)))
        {
        
            // check tank can be in line of fire or not (is_same_dir : 0 --> Not in line of fire, is_same_dir : 1 --> tank is in line of fire)
            int is_same_dir = (xsrc != xtarget) ? ((xsrc > xtarget && xsrc < xtank) || (xsrc < xtarget && xsrc > xtank) ? 0 : 1) : ((ysrc > ytarget && ysrc < ytank) || (ysrc < ytarget && ysrc > ytank)? 0 : 1);
            
            if(is_same_dir)
            {
                // Shifting corditates of tank (Making firing tank at (0, 0) & and shifting tank wrt firing tank)
                xtank = x[threadIdx.x] - xsrc;
                ytank = y[threadIdx.x] - ysrc;

                // calcuation of nearest collinear tank and having same direction as firing line
                int dis = (xtank != 0) ? ((xtank < 0) ? -xtank : xtank) : ((ytank < 0) ? -ytank : ytank);

                // LOCK IMPLEMENTATION 
                for (int i = 0; i < 32; i++) // Loop to interate Each Thread within WARP
                {
                    if (threadIdx.x % 32 == i) // Checking only one thread of warp get into loop
                    {
                        while (atomicCAS(&lockVar, 0, 1) != 0); // Lock
                        // CRITICAL SECTION, calculation of true target
                        if (min_dis > dis)
                        {
                            min_dis = dis;
                            temp = threadIdx.x;
                        }
                        atomicExch(&lockVar, 0); // Unlock
                    }
                }
            }
        }
    }

    // __syncthreads to ensure min_dis and true target is calculated properly
    __syncthreads();

    // updating Score and Power (HP) for the round
    if (threadIdx.x == 0 && temp != -1)
    {
        atomicAdd(&score[blockIdx.x], 1); // Increment Score of Firing Tank
        atomicSub(&power[temp], 1); // Decrement HP of tank which gets hit
    }
}

// kernel that handel which tank distroyed after the round completion
__global__ void checkDistroyed(int *power, unsigned *is_Dead, unsigned *distroyed, int T)
{
    // Check any tank is died in current round
    if (power[threadIdx.x] <= 0 && is_Dead[threadIdx.x] == 0)
    {
        is_Dead[threadIdx.x] = 1; // Update tank is dead
        atomicAdd(distroyed, 1); // Increment in No of Distroyed Tank
    }
    
    
    // Update No. of Rounds
    if (threadIdx.x == 0)
    {
        if (rounds % T == T - 1) // Condition to skip round if tank gone to fire iteself
            rounds += 2;
        else
            rounds++; // Increment in No. of Rounds
    }
}

__global__ void initialize_arrays(int *power, unsigned *scr, unsigned *is_Dead, int H)
{
    power[threadIdx.x] = H;
    scr[threadIdx.x] = 0;
    is_Dead[threadIdx.x] = 0;
}

//***********************************************

int main(int argc, char **argv)
{
    // Variable declarations
    int M, N, T, H, *xcoord, *ycoord, *score;

    FILE *inputfilepointer;

    // File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer = fopen(inputfilename, "r");

    if (inputfilepointer == NULL)
    {
        printf("input.txt file failed to open.");
        return 0;
    }

    fscanf(inputfilepointer, "%d", &M);
    fscanf(inputfilepointer, "%d", &N);
    fscanf(inputfilepointer, "%d", &T); // T is number of Tanks
    fscanf(inputfilepointer, "%d", &H); // H is the starting Health point of each Tank

    // Allocate memory on CPU
    xcoord = (int *)malloc(T * sizeof(int)); // X coordinate of each tank
    ycoord = (int *)malloc(T * sizeof(int)); // Y coordinate of each tank
    score = (int *)malloc(T * sizeof(int));  // Score of each tank (ensure that at the end you have copied back the score calculations on the GPU back to this allocation)

    // Get the Input of Tank coordinates
    for (int i = 0; i < T; i++)
    {
        fscanf(inputfilepointer, "%d", &xcoord[i]);
        fscanf(inputfilepointer, "%d", &ycoord[i]);
    }

    auto start = chrono::high_resolution_clock::now();

    //*********************************
    // Your Code begins here (Do not change anything in main() above this comment)
    //********************************

    int *power;
    unsigned *scr, *is_Dead;

    // Allocating memory in GPU and copying required data from CPU or Initialize memory with proper number

    // Allocating score array of GPU and initialize with 0
    cudaMalloc(&scr, sizeof(int) * T);

    // Allocation of array on GPU which keeps track of tanks i.e tanks are alive or not
    cudaMalloc(&is_Dead, sizeof(int) * T);

    // Allocate space on GPU for HP Array
    cudaMalloc(&power, sizeof(int) * T);
    
    //Initialize power array using kernel
    initialize_arrays<<<1, T>>>(power, scr, is_Dead, H);
    



    
    // cudaMemset(&is_Dead, 0, sizeof(int) * T);

    // Copying x and y cordinates of tanks to GPU
    int *x_cord, *y_cord;
    cudaMalloc(&x_cord, sizeof(int) * T);
    cudaMemcpy(x_cord, xcoord, sizeof(int) * T, cudaMemcpyHostToDevice);

    cudaMalloc(&y_cord, sizeof(int) * T);
    cudaMemcpy(y_cord, ycoord, sizeof(int) * T, cudaMemcpyHostToDevice);

    unsigned *distroyed; // Tracks how many tanks distroyed

    // Allocate Distroyed Variable that can be access from both CPU and GPU, Initialize with 0
    cudaHostAlloc(&distroyed, sizeof(unsigned), 0);
    *distroyed = 0;

    cudaDeviceSynchronize();  // To assure proper memory allocation

    // Loop Continues till Total number of alive tank is not equal to 1 or 0
    while (*distroyed < (T - 1))
    {
        fire<<<T, T>>>(x_cord, y_cord, scr, power, is_Dead, T); // Fire for a perticular round
        checkDistroyed<<<1, T>>>(power, is_Dead, distroyed, T); // Calulate which tanks distroyed after above round of firing
        cudaDeviceSynchronize();
    }

    // Deallocating/Free memory FROM GPU
    cudaFreeHost(distroyed);
    
    cudaFree(power);
    cudaFree(is_Dead);
    cudaFree(x_cord);
    cudaFree(y_cord);

    // Cuda Memcpy to copy score from GPU to CPU
    cudaMemcpy(score, scr, sizeof(unsigned) * T, cudaMemcpyDeviceToHost);

    cudaFree(scr);

    //*********************************
    // Your Code ends here (Do not change anything in main() below this comment)
    //********************************

    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double, std::micro> timeTaken = end - start;

    printf("Execution time : %f\n", timeTaken.count());

    // Output
    char *outputfilename = argv[2];
    char *exectimefilename = argv[3];
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename, "w");

    for (int i = 0; i < T; i++)
    {
        fprintf(outputfilepointer, "%d\n", score[i]);
    }

    fclose(inputfilepointer);
    fclose(outputfilepointer);

    outputfilepointer = fopen(exectimefilename, "w");
    fprintf(outputfilepointer, "%f", timeTaken.count());
    fclose(outputfilepointer);

    free(xcoord);
    free(ycoord);
    free(score);
    cudaDeviceSynchronize();
    return 0;
}

//************************************************ CODE END ****************************************************
