/* Vector percent change: C = log(A[x]/A[x-1])
 * Device code.
 */

#ifndef _VECTOR_KERNEL_H_
#define _VECTOR_KERNEL_H_

#include <stdio.h>
#include <math.h>
#include "Vector.h"


// Vector addition kernel thread specification
__global__ void VectorPercentChangeKernel(Vector A, Vector C)
{
	//Add the two vectors
	unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
	if(x == 0){
		C.elements[x] = 0;
		//printf("A.elements[x]: %f\n", A.elements[x]);
	}
	else if(x < A.length && x!=0) {
	   C.elements[x] = __logf(A.elements[x]/A.elements[x-1]);
	}
    __syncthreads();

}

__global__ void VectorVarianceKernel(Vector A, Vector D,float mean){
	unsigned int idx = threadIdx.x + blockDim.x *blockIdx.x ;
	if(idx < D.length){
	D.elements[idx] = __powf(fabs(A.elements[idx] - mean), 2.0f);
	}
	
}


//Genearte random values
//random value = standard deviation * NORMSINV(RAND())
//next day's price = today's price * e ^ (drift + random value)
__global__ void VectorRandomValueAndNextDayPriceKernel(Vector today, float* randVector, float std_deviation, float drift, float *next_day){
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < today.length) {
		float normal_random_value;
     	normal_random_value = normcdfinvf(randVector[idx]) * std_deviation;
     	//printf("thread %d, value %f\n",idx,today.elements[idx]);
     	next_day[idx] = today.elements[idx] * exp(drift + normal_random_value);
    }
    __syncthreads();


}
#endif
