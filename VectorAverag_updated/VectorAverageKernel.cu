
#include <stdio.h>
#include "VectorAverage.h"

//Use shared memory to compute average
//Reduction

//C = Sum(A[x]/ n)
//Input: Vector A
//Ouput: Vector C
__global__ void CalculateAverageKernel(Vector A, Vector C){

unsigned int totalThreads = blockDim.x; //Total number of active threads
unsigned int t = threadIdx.x;
unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;

__shared__ float average[BLOCK_SIZE];


while(totalThreads > 1){

unsigned int half = (totalThreads >>1) ; //Only the first half of threads will be active

	if(t < half){
		if(( t + half ) < blockDim.x){
		//Add value to the sum 
		average[t] += average[threadIdx.x + half];
		//Divide the value by 2
		average[t] /= 2;
		}
	}
__syncthreads();
totalThreads = half;
}

//Store the result in the first elements of average[0]
C.elements[x] = average[0]/ A.length;

for(int i = 1 ; i < C.length; i++){
	C.elements[i] = 0;
}
}



