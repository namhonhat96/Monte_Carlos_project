#ifndef __VECTOR_H__
#define __VECTOR_H__

#define BLOCK_SIZE 1000
#define VSIZE 1000
#define SIM_SIZE 10000
#define DAYS 8



#define FATAL(msg, ...) \
    do {\
        fprintf(stderr, "[%s:%d] "msg"\n", __FILE__, __LINE__, ##__VA_ARGS__);\
        exit(-1);\
    } while(0)


typedef struct {
	//length of the vector
    unsigned int length;
	//Pointer to the first element of the vector
    float* elements;
} Vector;

#endif
