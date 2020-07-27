#ifndef __AVERAGE_H__
#define __AVERAGE_H__

#define BLOCK_SIZE 256
#define NUM_ELEMENTS 1500
#define VSIZE 256

typedef struct {
	//length of the vector
    unsigned int length;
	//Pointer to the first element of the vector
    float* elements;
} Vector;


#endif