#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <math.h>
#include <cuda.h>
#include <curand.h>
#include <time.h>
#include "VectorKernel.cu"
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <functional>
#define MAXLINE 100000
extern "C"
void computeGold(float* C, const float* A, unsigned int N);

Vector AllocateDeviceVector(const Vector V);
Vector AllocateVector(int length, int init, float initvalue);
void CopyToDeviceVector(Vector Vdevice, const Vector Vhost);
void CopyFromDeviceVector(Vector Vhost, const Vector Vdevice);
const char* getfield(char* line, int num);
void ReadFile(float* stock_data, char* file_name, int nums);
void WriteFile(Vector V, char* file_name);
//-----------------------------Parallel---------------------------------
float VectorVariance(const Vector A, float mean); // Variance
// void VectorStDeviation(Vector A, Vector E, float mean); // Standard deviation
void PrintVector(Vector M);
float* UniformNumberGenerator(int size,char * pickRNG);
void VectorPercentChange(const Vector A, Vector C);
//----------------------------------------------------------------------


float computeGold_percentChange(float* C, const float* A, unsigned int N);
float computeGold_stdDev(float* C, float average_daily, unsigned int N);
float* computeGold_randVal(float std_dev, int size, float* input);
float* computeGold_NextVal(float* today, float drift, float* rand_val);
//----------------------------------------------------------------------

void PrintData(float* data, int size);
void VectorDrift(Vector E, float variance);
float* RandomValueAndNextDayPrice(Vector today, float* randVector, float std_deviation,float drift);
void VectorNextDay(float today, float drift, float* randomValue );

int main(int argc, char** argv){
	srand(time(0));
	Vector StockData; //Input data
	Vector PercentChange; //Store percent change
	Vector PercentChangeS; // Sequential Version
	// float std_deviation = 0;
	
	// Check command line for input vector files
	printf("argc %d\n",argc);
	printf("argv %s\n",argv[1]);
	char RNG[10] = "mt1";
	if(argc == 1) 
	{
		// No inputs provided
		// Allocate and initialize the vectors
		StockData  = AllocateVector(VSIZE, 1, 0.0f);
		PercentChange  = AllocateVector(VSIZE, 0, 0.0f);
		PercentChangeS = AllocateVector(VSIZE, 0, 0.0f);
	}
	else if( argc == 2)
	{
		// Inputs provided
		// Allocate and read source vectors from disk
		StockData  = AllocateVector(VSIZE, 0, 0.0f);		
		PercentChange  = AllocateVector(VSIZE, 0, 0.0f);
		PercentChangeS = AllocateVector(VSIZE, 0, 0.0f);
		ReadFile(StockData.elements, argv[1], VSIZE);
		//PrintVector(StockData);
	}
	else if( argc == 3)
	{
		// Inputs provided
		// Allocate and read source vectors from disk
		StockData  = AllocateVector(VSIZE, 0, 0.0f);		
		PercentChange  = AllocateVector(VSIZE, 0, 0.0f);
		PercentChangeS = AllocateVector(VSIZE, 0, 0.0f);
		ReadFile(StockData.elements, argv[1], VSIZE);
		strcpy(RNG,argv[2]);
		//PrintVector(StockData);
	}
	
	// A + B on the device
	
	mkdir("output", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	//-----------------------------Parallel---------------------------------
	cudaEvent_t start_GPU, stop_GPU;
	float GPU_time = 0;
	cudaEventCreate(&start_GPU);
	cudaEventCreate(&stop_GPU);
	cudaEventRecord(start_GPU);
	
	VectorPercentChange(StockData, PercentChange);
	//PrintVector(PercentChange);
	float result = thrust::reduce(thrust::host, PercentChange.elements, PercentChange.elements + PercentChange.length, 0.0f, thrust::plus<float>());
	float AVG = result/(PercentChange.length);
	//printf("AVG: %f\n", AVG);
	float VAR = VectorVariance(PercentChange,AVG);
	//printf("VAR: %f\n", VAR);
	float DRIFT = AVG - (VAR/2.0f);
	//printf("DRIFT: %f\n", DRIFT);
	float STD = sqrt(VAR);
	//printf("STD: %f\n", STD);
	Vector Today = AllocateVector(SIM_SIZE, 0, StockData.elements[StockData.length-1]);
	Vector next_day;
	Vector RANDSTORE;
	RANDSTORE.elements = (float*)malloc(SIM_SIZE*DAYS*sizeof(float));
	RANDSTORE.length = SIM_SIZE*DAYS;
	char Buffer[30];
	for(int i = 0; i<DAYS;i++){
		//PrintData(Today.elements, Today.length);
		sprintf(Buffer,"output/output%d.txt",i);
		//WriteFile(Today,Buffer);
		float * randVector = UniformNumberGenerator(SIM_SIZE,(char*)RNG);
		memcpy(RANDSTORE.elements + (i*SIM_SIZE),randVector,SIM_SIZE*sizeof(float));
		//printf("Random numbers: ");
		//PrintData(randVector, SIM_SIZE);
		next_day.length = Today.length;
		next_day.elements = RandomValueAndNextDayPrice(Today,randVector,STD,DRIFT);
		memcpy(Today.elements, next_day.elements, SIM_SIZE*sizeof(float));
		free(randVector);
	}
	cudaEventRecord(stop_GPU);
	cudaEventSynchronize(stop_GPU);
	cudaEventElapsedTime(&GPU_time, start_GPU, stop_GPU);
	printf("GPU time: %f ms\n", GPU_time);
	cudaEventDestroy(start_GPU);
	cudaEventDestroy(stop_GPU);
	
	//------------------------------------------------------------------
	WriteFile(RANDSTORE,(char*)"random.txt");
	free(PercentChange.elements);
    PercentChange.elements = NULL;
    
    
    //----------------------------Sequential----------------------------
    cudaEvent_t start, stop;
	float CPU_time = 0.0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
    float average_daily;
    average_daily = computeGold_percentChange(PercentChangeS.elements, StockData.elements, VSIZE);
    float std_dev;
    std_dev = computeGold_stdDev(PercentChangeS.elements, average_daily, VSIZE);
    float drift ;
    drift = average_daily - (pow(std_dev,2.0) / 2 );
	float* rand_num;
    float* input;
	input = (float*)malloc(SIM_SIZE * DAYS * sizeof(float));
	rand_num = (float*)malloc(DAYS * SIM_SIZE * sizeof(float));
	for(int i = 0 ; i < DAYS *SIM_SIZE ; i ++){
		input[i] = (double)rand()/ (RAND_MAX);
    }
    float *Next_dayS, *today;
    Next_dayS = (float*)malloc(DAYS * SIM_SIZE * sizeof(float));
    today = (float*)malloc(SIM_SIZE *DAYS* sizeof(float));
	for(int i = 0 ; i < SIM_SIZE* DAYS ; i ++){
		today[i]=StockData.elements[VSIZE-1];
	}	
	rand_num = computeGold_randVal(std_dev, SIM_SIZE * DAYS ,input);	             	
	Next_dayS=computeGold_NextVal(today, drift, rand_num);
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
    cudaEventElapsedTime(&CPU_time, start, stop);
    printf("CPU time: %f ms\n", CPU_time);
    cudaEventDestroy(start_GPU);
	cudaEventDestroy(stop_GPU);
	
	mkdir("outputS", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    Vector CHUNK;
    CHUNK.elements = (float*)malloc(SIM_SIZE*sizeof(float));
    CHUNK.length=SIM_SIZE;
    memcpy(CHUNK.elements,Next_dayS,SIM_SIZE*sizeof(float));
	for(int i = 0; i < DAYS; i++){
        sprintf(Buffer, "outputS/output%d.txt",i);
        WriteFile(CHUNK,Buffer);
        memcpy(CHUNK.elements,Next_dayS+(i*SIM_SIZE),SIM_SIZE*sizeof(float));
    }
    
    
    
    free(PercentChangeS.elements);
    PercentChangeS.elements = NULL;
    free(rand_num);
    free(input);
    free(Next_dayS);	
	rand_num = NULL;
	input = NULL;
	Next_dayS = NULL;
	free(StockData.elements);
    StockData.elements = NULL;
    //------------------------------------------------------------------
	return 0;
	
}

float* RandomValueAndNextDayPrice(Vector today, float* randVector, float std_deviation,float drift){
	
	//---------------------------input----------------------------------
	Vector today_dev = AllocateDeviceVector(today);
	CopyToDeviceVector(today_dev, today);
	float* randVector_dev;
	cudaMalloc((void**) &randVector_dev, today.length*sizeof(float));
	cudaMemcpy(randVector_dev, randVector, today.length * sizeof(float), cudaMemcpyHostToDevice);
	//---------------------------output---------------------------------
	float* next_day_dev;
	float* next_day_host;
	next_day_host = (float *)calloc(today.length, sizeof(float));
	cudaMalloc((void**) &next_day_dev, today.length*sizeof(float));
	//--------------------------kernel----------------------------------
	dim3 dim_grid, dim_block;
	dim_grid.x = (today.length + BLOCK_SIZE ) / (BLOCK_SIZE);
	dim_grid.y = 1;
	dim_grid.z = 1;
	dim_block.x = BLOCK_SIZE;
	dim_block.y = 1;
	dim_block.z = 1;
	VectorRandomValueAndNextDayPriceKernel<<<dim_grid,dim_block>>>(today_dev,randVector_dev,std_deviation,drift,next_day_dev);
	//------------------------------------------------------------------
	cudaMemcpy(next_day_host, next_day_dev, today.length * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(today_dev.elements);
	cudaFree(next_day_dev);
	cudaFree(randVector_dev);
	return next_day_host;

}

void VectorPercentChange(const Vector A, Vector C){
	//Interface host call to the device kernel code and invoke the kernel
   	//cudaError_t cuda_ret;
	Vector d_A, d_C;
    // steps:
    
    // 1. allocate device vectors d_A and d_C with length same as input vector
    d_A = AllocateDeviceVector(A);
    d_C = AllocateDeviceVector(C);
    // 2. copy A to d_A,
    CopyToDeviceVector(d_A, A);
    //CopyToDeviceVector(d_C, C);
    // 3. launch kernel to compute d_C = d_An[1]/d_An[x-1]
    dim3 dim_grid, dim_block;
    dim_block.x = BLOCK_SIZE; dim_block.y = dim_block.z = 1;
    dim_grid.x = A.length / dim_block.x;
    if(A.length % dim_block.x != 0) dim_grid.x++;
    dim_grid.y = 1;
    dim_grid.z = 1;
    VectorPercentChangeKernel<<<dim_grid, dim_block>>>(d_A, d_C);
    // 4. copy d_C back to host vector C
    CopyFromDeviceVector(C,d_C);
    // 5. free device vectors d_A, d_B, d_C
    cudaFree(d_A.elements);
    cudaFree(d_C.elements);
    
}



float* UniformNumberGenerator(int size,char * pickRNG){
	curandGenerator_t gen;
	float *devData, *host;
	host = (float *)calloc(size, sizeof(float));
	cudaMalloc((void**) &devData, size*sizeof(float));
	if(strcmp(pickRNG,"mt1") == 0){
		curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32); //Mersenne Twister family number 1
	}else if(strcmp(pickRNG,"mt2")== 0){
		curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937); //Mersenne Twister family number 2
	}else if(strcmp(pickRNG,"lfsr")== 0){
		curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW); //linear feedback shift register
	}else if(strcmp(pickRNG,"mrg")== 0){
		curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MRG32K3A); //multiple recursive generator
	}else if(strcmp(pickRNG,"phi")== 0){
		curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10); //Philox
	}else if(strcmp(pickRNG,"sobol")== 0){
		curandCreateGenerator(&gen, CURAND_RNG_QUASI_SOBOL32); //SOBOL
	}
	long randvalue = (long)rand();
	//printf("%ld\n", randvalue);
	curandSetPseudoRandomGeneratorSeed(gen, randvalue);
	curandGenerateUniform(gen, devData, size);
	cudaMemcpy(host, devData, size * sizeof(float), cudaMemcpyDeviceToHost);
	curandDestroyGenerator(gen);
	cudaFree(devData);
	return host;

}

//Vector A: Input
//Vector D: output
//mean: C.elements[0]
float VectorVariance(const Vector A,float mean){
	Vector d_A;
	Vector d_D;
	Vector D  = AllocateVector(A.length, 0, 0.0f);
	d_A = AllocateDeviceVector(A);
	d_D = AllocateDeviceVector(D);
	CopyToDeviceVector(d_A, A);
	dim3 dimGrid, dimBlock;
	dimBlock.x = BLOCK_SIZE;
	dimBlock.y = 1;
	dimBlock.z = 1;
	dimGrid.x = (VSIZE ) /(BLOCK_SIZE);
	if(VSIZE % BLOCK_SIZE != 0) {dimGrid.x++;}
	dimGrid.y = 1;
	dimGrid.z = 1;
	VectorVarianceKernel <<< dimGrid, dimBlock >>>(d_A, d_D,mean);
	CopyFromDeviceVector(D,d_D);
	cudaFree(d_D.elements);
	float result = thrust::reduce(thrust::host, D.elements, D.elements + D.length,0.0f,thrust::plus<float>());
	float VAR = result/D.length;
	free(D.elements);
	return VAR;
}

Vector AllocateVector(int length, int init, float initvalue)
{
    Vector V;
    V.length = length;
    V.elements = NULL;
	V.elements = (float*) malloc(length*sizeof(float));
	for(unsigned int i = 0; i < V.length; i++)
	{	
		if(init == 0){
			V.elements[i] = initvalue;
		}else if(init == 1){
			V.elements[i] = (rand()/(float)RAND_MAX);
		}
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



const char* getfield(char* line, int num)
{
    const char* tok;
    for (tok = strtok(line, ",");tok && *tok;tok = strtok(NULL, ",\n"))
    {
        if (!--num)
            return tok;
    }
    return NULL;
}

void ReadFile(float* stock_data, char* file_name, int nums)
{
    FILE* stream = fopen(file_name, "r");
    char line[1024];

    fgets(line, 1024, stream);
    for(int i = 0; i < nums; i++) {
      fgets(line, 1024, stream);
      char* tmp = strdup(line);
      stock_data[i] = atof(getfield(tmp, 2));
      free(tmp);
    }

}
// Write a floating point vector to file
void WriteFile(Vector V, char* file_name)
{
    FILE* output = fopen(file_name, "w");
    for (unsigned i = 0; i < V.length; i++) {
        fprintf(output, "%f\n", V.elements[i]);
    }
    fclose(output);
}
// print float array
void PrintData(float* data, int size){
	//Print out the result
	for(int i = 0; i < size ; i++){
		printf(" %9.6f ", data[i]);
	}

	printf("\n");
}

void PrintVector(Vector M){
	printf("---------------------------------\n");
    for (int i = 0; i < M.length; i++){
		printf("%f|", M.elements[i]);
	}
	printf("\n");
}

//------------------------------sequential code-------------------------
float computeGold_percentChange(float* C, const float* A, unsigned int N){
	float average_daily;
	unsigned int i = 0;
	C[i] = 0;
	for (i=1; i < N; i++){
		C[i] = logf((A[i])/(A[i-1]));
    }
	float total= 0.0f;
	for (int i=1; i < N; i++){
		total += C[i];
	}
	average_daily = total/(N-1);
	return average_daily;
}

float computeGold_stdDev(float* C, float average_daily, unsigned int N){
	float variance, std_deviation;
	float sum1=0.0f; 
	for (int i = 1; i < N; i++)
	{
		sum1 = sum1 + pow((C[i] - average_daily), 2);
	}
	variance = sum1 / (float)(N-1);
	std_deviation = sqrt(variance);
	return std_deviation;
}

float* computeGold_randVal(float std_dev, int size, float* input){
	float* rand_num;
	rand_num = (float*)malloc(size * sizeof(float));
	for(int i = 0; i < DAYS; i++){
		for(int j = 0; j <  SIM_SIZE; j++){
			//rand_num[j* DAYS + i] = std_dev * normcdfinvf(input[j* DAYS + i]);
			rand_num[i* SIM_SIZE + j] = std_dev * normcdfinvf(input[i* SIM_SIZE + j]);
		}
	}
	return rand_num;
}

float* computeGold_NextVal(float* today, float drift, float* rand_val){
	for(int i = 0; i < DAYS; i++){
        for(int j = 0; j <  SIM_SIZE; j++){
			if(i == 0){
				today[i* SIM_SIZE + j] = today[i* SIM_SIZE + j]* exp( drift +rand_val[i* SIM_SIZE + j]);       
			}
			else{
				today[i* SIM_SIZE + j] = today[(i-1)* SIM_SIZE + j]* exp( drift +rand_val[i* SIM_SIZE + j]);
			}
		}	
	}	
	return today;
}
