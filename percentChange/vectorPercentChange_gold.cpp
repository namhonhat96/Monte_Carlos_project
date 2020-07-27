#include <stdlib.h>
#include <math.h>
////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C"
void computeGold(float* C, const float* A, unsigned int N);

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! C = A/A;
//! @param C          reference data, computed but preallocated
//! @param A          vector A as provided to device
//! @param N         length of vectors
////////////////////////////////////////////////////////////////////////////////
void computeGold(float* C, const float* A, unsigned int N)
{
	unsigned int i = 0;
	C[i] = NAN;
    for (i=1 ;i < N; i++){
		C[i] = logf((A[i])/(A[i-1]));
	}
}
