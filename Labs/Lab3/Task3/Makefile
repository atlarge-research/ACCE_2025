# Makefile to compile all .cu files in the folder using nvcc

NVCC = module load cuda12.3/toolkit/12.3 && nvcc
SRCS = $(wildcard *.cu)
EXECS = $(SRCS:.cu=)

all: $(EXECS)

%: %.cu
	$(NVCC) -o $@ $<

clean:
	rm -f $(EXECS)