/*
	CS 6023 Assignment 3.
	Do not make any changes to the boiler plate code or the other files in the folder.
	Use cudaFree to deallocate any memory not in usage.
	Optimize as much as possible.
//  */

#include "SceneNode.h"
#include <queue>
#include "Renderer.h"
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <chrono>
// #include "Renderer.cc"
// #include "SceneNode.cc"



void readFile (const char *fileName, std::vector<SceneNode*> &scenes, std::vector<std::vector<int> > &edges, std::vector<std::vector<int> > &translations, int &frameSizeX, int &frameSizeY) {
	/* Function for parsing input file*/

	FILE *inputFile = NULL;
	// Read the file for input.
	if ((inputFile = fopen (fileName, "r")) == NULL) {
		printf ("Failed at opening the file %s\n", fileName) ;
		return ;
	}

	// Input the header information.
	int numMeshes ;
	fscanf (inputFile, "%d", &numMeshes) ;
	fscanf (inputFile, "%d %d", &frameSizeX, &frameSizeY) ;


	// Input all meshes and store them inside a vector.
	int meshX, meshY ;
	int globalPositionX, globalPositionY; // top left corner of the matrix.
	int opacity ;
	int* currMesh ;
	for (int i=0; i<numMeshes; i++) {
		fscanf (inputFile, "%d %d", &meshX, &meshY) ;
		fscanf (inputFile, "%d %d", &globalPositionX, &globalPositionY) ;
		fscanf (inputFile, "%d", &opacity) ;
		currMesh = (int*) malloc (sizeof (int) * meshX * meshY) ;
		for (int j=0; j<meshX; j++) {
			for (int k=0; k<meshY; k++) {
				fscanf (inputFile, "%d", &currMesh[j*meshY+k]) ;
			}
		}
		//Create a Scene out of the mesh.
		SceneNode* scene = new SceneNode (i, currMesh, meshX, meshY, globalPositionX, globalPositionY, opacity) ;
		scenes.push_back (scene) ;
	}

	// Input all relations and store them in edges.
	int relations;
	fscanf (inputFile, "%d", &relations) ;
	int u, v ;
	for (int i=0; i<relations; i++) {
		fscanf (inputFile, "%d %d", &u, &v) ;
		edges.push_back ({u,v}) ;
	}

	// Input all translations.
	int numTranslations ;
	fscanf (inputFile, "%d", &numTranslations) ;
	std::vector<int> command (3, 0) ;
	for (int i=0; i<numTranslations; i++) {
		fscanf (inputFile, "%d %d %d", &command[0], &command[1], &command[2]) ;
		translations.push_back (command) ;
	}
}


void writeFile (const char* outputFileName, int *hFinalPng, int frameSizeX, int frameSizeY) {
	/* Function for writing the final png into a file.*/
	FILE *outputFile = NULL;
	if ((outputFile = fopen (outputFileName, "w")) == NULL) {
		printf ("Failed while opening output file\n") ;
	}

	for (int i=0; i<frameSizeX; i++) {
		for (int j=0; j<frameSizeY; j++) {
			fprintf (outputFile, "%d ", hFinalPng[i*frameSizeY+j]) ;
		}
		fprintf (outputFile, "\n") ;
	}
}




// Kernel that final print the matrix
__global__ void printMesh(int id, int V, int currOpacity, int R, int C, int *x, int *y, int sizex, int sizey, int *mesh, int *gans, int *gOpacity)
{
	int global_id = threadIdx.x + blockIdx.x * blockDim.x; // global id that represent indivisual element in the mesh
	int r = global_id/sizey; // row of mesh element
	int c = global_id%sizey; // col of mesh element
	if(r >= 0 && r < sizex && c >= 0 && c < sizey) {   // check condition r, c are in valid range
		// find the transalated cordinates of mesh element
		int corX = r + x[id];
		int corY = c + y[id];

		if (corX >= 0 && corX < R && corY >= 0 && corY < C)  // checking condition that translated cordinates are in valid scene index or not
		{
			if(gOpacity[corX * C + corY] <= currOpacity)
			{  // checking opacitity of mesh element with opacity of existing element
				gans[corX * C + corY] = mesh[r * sizey + c];   // update element in scene matrix
				gOpacity[corX * C + corY] = currOpacity; 		// update opacity matrix 
			}
		}
	}
}


// kernel to translations on all child nodes
__global__ void translateChild(int len, int start, int parent, int *gCsr, int *x, int *y)
{
  	// printf("translateChild Called\n");
	int global_id = blockIdx.x * blockDim.x + threadIdx.x;
	if(global_id < len)
	{
		int node = gCsr[start + global_id];
		x[node] += x[parent];
		y[node] += y[parent];
	}
}


// kernel to get translations correnspoding to indivisual nodes
__global__ void getNetTranslations(int *translations, int *x_arr, int *y_arr, int numTranslations)
{
	int global_id = threadIdx.x + blockDim.x * blockIdx.x;
	if(global_id < numTranslations)
	{
		int meshNo = translations[3 * global_id + 0];
		int command = translations[3 * global_id + 1];
		int amount = translations[3 * global_id + 2];
		int flag = ((command == 0 || command == 2) ? -1 : 1);
		int *arr = ((command == 0 || command == 1) ? x_arr : y_arr);
		// arr[meshNo] += (amount * flag);
		atomicAdd(arr+meshNo, amount * flag);
  	}
}

// kernel that actullaty translates the cordinates 
__global__ void finalTranslate(int v, int *arr_x, int *arr_y, int *x, int *y)
{
	int global_id = blockIdx.x * blockDim.x + threadIdx.x;
	if(global_id < v)
	{
		x[global_id] += arr_x[global_id];
		y[global_id] += arr_y[global_id];
	}
}



int main (int argc, char **argv) {

	// Read the scenes into memory from File.
	const char *inputFileName = argv[1] ;
	int* hFinalPng ;

	int frameSizeX, frameSizeY ;
	std::vector<SceneNode*> scenes ;
	std::vector<std::vector<int> > edges ;
	std::vector<std::vector<int> > translations ;
	readFile (inputFileName, scenes, edges, translations, frameSizeX, frameSizeY) ;
	hFinalPng = (int*) malloc (sizeof (int) * frameSizeX * frameSizeY) ;

	// Make the scene graph from the matrices.
    Renderer* scene = new Renderer(scenes, edges) ;

	// Basic information.
	int V = scenes.size () ;
	int E = edges.size () ;
	int numTranslations = translations.size () ;

	// Convert the scene graph into a csr.
	scene->make_csr () ; // Returns the Compressed Sparse Row representation for the graph.
	int *hOffset = scene->get_h_offset () ;
	int *hCsr = scene->get_h_csr () ;
	int *hOpacity = scene->get_opacity () ; // hOpacity[vertexNumber] contains opacity of vertex vertexNumber.
	int **hMesh = scene->get_mesh_csr () ; // hMesh[vertexNumber] contains the mesh attached to vertex vertexNumber.
	int *hGlobalCoordinatesX = scene->getGlobalCoordinatesX () ; // hGlobalCoordinatesX[vertexNumber] contains the X coordinate of the vertex vertexNumber.
	int *hGlobalCoordinatesY = scene->getGlobalCoordinatesY () ; // hGlobalCoordinatesY[vertexNumber] contains the Y coordinate of the vertex vertexNumber.
	int *hFrameSizeX = scene->getFrameSizeX () ; // hFrameSizeX[vertexNumber] contains the vertical size of the mesh attached to vertex vertexNumber.
	int *hFrameSizeY = scene->getFrameSizeY () ; // hFrameSizeY[vertexNumber] contains the horizontal size of the mesh attached to vertex vertexNumber.

	auto start = std::chrono::high_resolution_clock::now () ;


	// Code begins here.
	// Do not change anything above this comment.
	// ************************************************************
	int *gpuCordinatesX;
	int *gpuCordinatesY;
	int *gputrans;
	int *cpuTrans;
	int *gCsr;
	int *gans, *gOpacity;
	int *y_change_gpu;
	int *x_change_gpu;

	// Allocate the dummy array on GPU to handle the translations on each vertex in x-direction
	cudaMalloc(&x_change_gpu, sizeof(int) * V);
	cudaMemset(&x_change_gpu, sizeof(int) * V, 0); // Initially set array to 0

	// Allocate the dummy array on GPU to handle the translations on each vertex in y-direction
	cudaMalloc(&y_change_gpu, sizeof(int) * V);
	cudaMemset(&y_change_gpu, sizeof(int) * V, 0); // Initially set array to 0

	//creating 2D translations to 1D array
	cpuTrans = (int *)malloc(sizeof(int) * numTranslations * 3);
	// printf("numTranslations :: %d\n", numTranslations);
	for(int i = 0; i<numTranslations; i++)
	{
		cpuTrans[3*i + 0] = translations[i][0];
		cpuTrans[3*i + 1] = translations[i][1];
		cpuTrans[3*i + 2] = translations[i][2];
		// printf("%d %d %d \n", translations[i][0], translations[i][1], translations[i][2]);
	}


	// Allocate memory in GPU for translations and copy translations to GPU
	cudaMalloc(&gputrans, sizeof(int) * numTranslations * 3);
	cudaMemcpy(gputrans, cpuTrans, sizeof(int) * numTranslations * 3, cudaMemcpyHostToDevice);


	// Kernel call to get translations on indivisual nodes
	getNetTranslations<<<ceil(numTranslations/1024.0), 1024>>>(gputrans, x_change_gpu, y_change_gpu, numTranslations);

	// Free Translations stored on GPU
	cudaFree(gputrans);

	// Memory allocation and copy to GPU
	cudaMalloc(&gCsr, E * sizeof(int));
	cudaMemcpy(gCsr, hCsr, sizeof(int) * E, cudaMemcpyHostToDevice);

	cudaMalloc(&gpuCordinatesX, V * sizeof(int));
	cudaMemcpy(gpuCordinatesX, hGlobalCoordinatesX, V*sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc(&gpuCordinatesY, V * sizeof(int));
	cudaMemcpy(gpuCordinatesY, hGlobalCoordinatesY, V*sizeof(int), cudaMemcpyHostToDevice);
	

	// Perform BFS to handle trasitive relationship
  	// printf("Hello from above queue");
	std::queue<int> que;
	que.push(0);
	

	while(que.empty() == false)
	{
		int node = que.front();
		que.pop();
		int start = hOffset[node];
		int end = hOffset[node+1];
		if(start != end)
		{
			for(int i = start; i<end; i++)
			{
				que.push(hCsr[i]);
			}
			int threads = 1024;
			int blocks = ceil((end - start)/1024.0);
			translateChild<<<blocks, threads>>>(end - start, start, node, gCsr, x_change_gpu, y_change_gpu);
			cudaDeviceSynchronize();
		}
	}

	// Free CSR from GPU
	cudaFree(gCsr);

	// final update in cordinates;
    finalTranslate<<<ceil(V/1024.0), 1024>>>(V, x_change_gpu, y_change_gpu, gpuCordinatesX, gpuCordinatesY);
    cudaDeviceSynchronize();
    
	// Deallocate (free) GPU MEMEORY
	cudaFree(x_change_gpu);
	cudaFree(y_change_gpu);

    // Memory Allocation that stores the final scene matrix on GPu
	cudaMalloc(&gans, sizeof (int) * frameSizeX * frameSizeY);

  	cudaMalloc(&gOpacity, sizeof (int) * frameSizeX * frameSizeY);
	cudaMemset(&gOpacity, sizeof(int) * frameSizeX * frameSizeY, INT_MIN); // Initially set array to INT_MIN

 	// cudaMemcpy(gOpacity, hOpp, sizeof (int) * frameSizeX * frameSizeY, cudaMemcpyHostToDevice);

	// Allocate Memory for Mesh
	int *mesh;
	cudaMalloc(&mesh, sizeof(int) * 100 * 100); // max_row = 100, max_col = 100
	// Filling Scene based on opacity
  	for(int i = 0; i<V; i++) 
    {
		int currOpacity = hOpacity[i]; 
		cudaMemcpy(mesh, hMesh[i], sizeof(int) * hFrameSizeX[i] * hFrameSizeY[i], cudaMemcpyHostToDevice);
		printMesh<<<ceil((hFrameSizeX[i] * hFrameSizeY[i])/1024.0), 1024>>>(i, V, currOpacity, frameSizeX, frameSizeY, gpuCordinatesX, gpuCordinatesY, hFrameSizeX[i], hFrameSizeY[i], mesh, gans, gOpacity);
		
	}
	// Cuda Free 
	cudaFree(mesh);
	cudaFree(gpuCordinatesX);
	cudaFree(gpuCordinatesY);
	cudaFree(gOpacity);
	
	// Copying final Scene Matrix from GPU to CPU
  	cudaMemcpy(hFinalPng, gans, sizeof (int) * frameSizeX * frameSizeY, cudaMemcpyDeviceToHost);

	//Deallocate memory from gpu
	
	cudaFree(gans);
	// Do not change anything below this comment.
	// Code ends here.

	auto end  = std::chrono::high_resolution_clock::now () ;

	std::chrono::duration<double, std::micro> timeTaken = end-start;

	printf ("execution time : %f\n", timeTaken) ;
	// Write output matrix to file.
	const char *outputFileName = argv[2] ;
	writeFile (outputFileName, hFinalPng, frameSizeX, frameSizeY) ;
}