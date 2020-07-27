#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>

#include "VectorAverageKernel.cu"
#include "VectorAverage.h"
#include "VectorAverage_gold.cpp"
#define MAXLINE 100000
extern "C"
void computeGold(float* C, const float* A, unsigned int N);

Vector AllocateDeviceVector(const Vector V);
Vector AllocateVector(int size, int init);
void CopyToDeviceVector(Vector Vdevice, const Vector Vhost);
void CopyFromDeviceVector(Vector Vhost, const Vector Vdevice);
int ReadFile(Vector* V, char* file_name);
void WriteFile(Vector V, char* file_name);

void VectorAverage(const Vector A, Vector C);
void printVector(Vector M);


int main(int argc, char** argv){
	Vector A;
	Vector C;
	// Number of elements in the vectors
	unsigned int size_elements = VSIZE;
	int errorA = 0;
	
	srand(2012);
	
	// Check command line for input vector files
	printf("argc %d\n",argc);
	if(argc == 1 || argc == 2) 
	{
		// No inputs provided
		// Allocate and initialize the vectors
		A  = AllocateVector(VSIZE, 1);
		C  = AllocateVector(VSIZE, 0);
	}
	else
	{
		// Inputs provided
		// Allocate and read source vectors from disk
		A  = AllocateVector(VSIZE, 0);		
		C  = AllocateVector(VSIZE, 0);
		errorA = ReadFile(&A, argv[2]);
		printVector(A);
		// check for read errors
		if(errorA != size_elements)
		{
			printf("Error reading input files %d, %d\n", errorA);
			return 1;
		}
	}
	
	// A + B on the device
    VectorAverage(A, C);
    // compute the vector addition on the CPU for comparison
    Vector reference = AllocateVector(size_elements, 0);
    computeGold(reference.elements,A.elements,size_elements);    
    // check if the device result is equivalent to the expected solution
    //CUTBoolean res = cutComparefe(reference.elements, C.elements, 
	//								size_elements, 0.0001f);
    unsigned res = 1;
    for (unsigned i = 0; i < size_elements; i++)
        if (abs(reference.elements[i] - C.elements[i]) > 0.0001f)
            res = 0;

    printf("Test %s\n", (1 == res) ? "PASSED" : "FAILED");
    
    // output result if output file is requested
    if(argc == 3)
    {
		WriteFile(C, argv[1]);
	}
	else if(argc == 2)
	{
	    WriteFile(C, argv[1]);
	}    

	// Free host matrices
    free(A.elements);
    A.elements = NULL;
    free(C.elements);
    C.elements = NULL;
	return 0;
}

//Vector A: input
//Vector C: output
void VectorAverage(const Vector A, Vector C){
	
	float average = 0;
	Vector d_A;
	Vector d_C;
	
	d_A = AllocateDeviceVector(A);
	d_C = AllocateDeviceVector(C);
	//Copy data from host to device
	CopyToDeviceVector(d_A,A);


	dim3 dimGrid, dimBlock;
	dimBlock.x = BLOCK_SIZE;
	dimBlock.y = 1;
	dimBlock.z = 1;

	dimGrid.x = (A.length ) /(BLOCK_SIZE);
	if(A.length % BLOCK_SIZE != 0) {dimGrid.x++;}
	dimGrid.y = 1;
	dimGrid.z = 1;

	CalculateAverageKernel <<< dimGrid, dimBlock >>> (d_A, d_C);


	CopyFromDeviceVector(C,d_C);
	average = C.elements[0];
	printf("Average: %d", average);

	cudaFree(d_A.elements);
	cudaFree(d_C.elements);
}

Vector AllocateVector(int length, int init)
{
    Vector V;
    V.length = length;
    V.elements = NULL;
		
	V.elements = (float*) malloc(length*sizeof(float));

	for(unsigned int i = 0; i < V.length; i++)
	{
		V.elements[i] = (init == 0) ? (0.0f) : (rand() / (float)RAND_MAX);
	}
    return V;
}	


Vector AllocateDeviceVector(const Vector V)
{
    Vector Vdevice = V;
    int size = V.length * sizeof(float);
    cudaError_t cuda_ret = cudaMalloc((void**)&Vdevice.elements, size);
    if(cuda_ret != cudaSuccess) {
        printf("Unable to allocate device memory");
        exit(0);
    }
    return Vdevice;
}

void CopyToDeviceVector(Vector Vdevice, const Vector Vhost)
{
    int size = Vhost.length * sizeof(float);
    Vdevice.length = Vhost.length;
    cudaMemcpy(Vdevice.elements, Vhost.elements, size, 
					cudaMemcpyHostToDevice);
}

// Copy a device vector to a host vector.
void CopyFromDeviceVector(Vector Vhost, const Vector Vdevice)
{
    int size = Vdevice.length * sizeof(float);
    cudaMemcpy(Vhost.elements, Vdevice.elements, size, 
					cudaMemcpyDeviceToHost);
}

int ReadFile(Vector* V, char* file_name)
{
	unsigned int data_read = VSIZE;
	FILE* input = fopen(file_name, "r");
    char vector_string[MAXLINE];
    fgets(vector_string, MAXLINE, input);
    char* part = strtok(vector_string, " ");
    for (unsigned i = 0; i < VSIZE; i++) {
        V->elements[i] = atof(part);
        part = strtok(NULL, " ");
    }
	return data_read;
}

// Write a floating point vector to file
void WriteFile(Vector V, char* file_name)
{
    FILE* output = fopen(file_name, "w");
    for (unsigned i = 0; i < VSIZE; i++) {
        fprintf(output, "%f ", V.elements[i]);
    }
}


void printVector(Vector M){   
    for (int i = 0; i < M.length; i++){
		printf("%f|", M.elements[i]);
	}
	printf("\n");
}
