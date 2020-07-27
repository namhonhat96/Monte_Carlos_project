#ifndef _PERCENTCHANGE_H_
#define _PERCENTCHANGE_H_

// Thread block size
#define BLOCK_SIZE 4

// Vector dimensions
#define VSIZE 4 // vector size

#define FATAL(msg, ...) \
    do {\
        fprintf(stderr, "[%s:%d] "msg"\n", __FILE__, __LINE__, ##__VA_ARGS__);\
        exit(-1);\
    } while(0)


// Vector Structure declaration
typedef struct {
	//length of the vector
    unsigned int length;
	//Pointer to the first element of the vector
    float* elements;
} Vector;


#endif

