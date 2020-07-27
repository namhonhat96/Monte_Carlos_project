
NVCC        = nvcc

NVCC_FLAGS  = -I/usr/local/cuda/include -gencode=arch=compute_50,code=\"sm_50,compute_50\"
ifdef dbg
	NVCC_FLAGS  += -g -G
else
	NVCC_FLAGS  += -O2
endif

LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	        = vectorPercentChange
OBJ	        = vectorPercentChange_cu.o vectorPercentChange_gold.o

default: $(EXE)

vectorPercentChange_cu.o: vectorPercentChange.cu vectorPercentChange_kernel.cu vectorPercentChange.h
	$(NVCC) -c -o $@ vectorPercentChange.cu $(NVCC_FLAGS)

vectorPercentChange_gold.o: vectorPercentChange_gold.cpp
	$(NVCC) -c -o $@ vectorPercentChange_gold.cpp $(NVCC_FLAGS) 

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS) $(NVCC_FLAGS)

clean:
	rm -rf *.o $(EXE)
