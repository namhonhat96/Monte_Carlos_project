
NVCC        = nvcc

NVCC_FLAGS  = -I/usr/local/cuda/include -gencode=arch=compute_50,code=\"sm_50,compute_50\"
ifdef dbg
	NVCC_FLAGS  += -g -G
else
	NVCC_FLAGS  += -O2
endif

# LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64 
LD_FLAGS 	= -lcurand -lcudart -L/usr/local/cuda/lib64
EXE	        = vector
OBJ	        = Vector_cu.o 
#VectorAverage_gold.o
# OBJ	        = distribution.o 

default: $(EXE)

Vector_cu.o: main.cu VectorKernel.cu Vector.h
	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS)
# VectorAverage_gold.o: VectorAverage_gold.cpp
# 	$(NVCC) -c -o $@ VectorAverage_gold.cpp $(NVCC_FLAGS) 

# distribution.o: distribution.cu 
# 	$(NVCC) -c -o $@ distribution.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS) $(NVCC_FLAGS)

clean:
	rm -rf *.o $(EXE)
