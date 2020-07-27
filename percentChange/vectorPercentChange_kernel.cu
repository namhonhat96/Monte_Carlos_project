/* Vector percent change: C = log(A[x]/A[x-1])
 * Device code.
 */

#ifndef _PERCENTCHANGE_KERNEL_H_
#define _PERCENTCHANGE_KERNEL_H_

#include <stdio.h>
#include "vectorPercentChange.h"

// Vector addition kernel thread specification
__global__ void VectorPercentChangeKernel(Vector A, Vector C)
{
	//Add the two vectors
	unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    if(x == 0){
		C.elements[x] = NAN;
	}
    else if(x < A.length && x!=0) {
       C.elements[x] = __logf(A.elements[x]/A.elements[x-1]);
       printf("A.elements[x]: %f\n", A.elements[x]);
       printf("A.elements[x-1]: %f\n", A.elements[x-1]);
       printf("C.elements[x]: %f\n", C.elements[x]);
    }
    __syncthreads();

}

#endif
