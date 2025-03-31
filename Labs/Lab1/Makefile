# Makefile to compile, run, and clean up a CUDA program
CC ?= clang
CFLAGS = -Ofast -Wno-unused-result -Wno-ignored-pragmas -Wno-unknown-attributes
LDFLAGS =
INCLUDES =
LDLIBS = -fopenmp -lm
OUTPUT_FILE = -o $@

all: vectorAdd_sequential vectorAdd_openmp

vectorAdd_sequential: vectorAdd_sequential.c
	gcc -o vectorAdd_sequential vectorAdd_sequential.c

vectorAdd_openmp: vectorAdd_openmp.c
	gcc -fopenmp -o vectorAdd_openmp vectorAdd_openmp.c

# Rule to compile the matMul openMP example
matMul_omp: matMul_omp.c
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $^ $(LDLIBS) $(OUTPUT_FILE)

# Clean up the executable
clean:
	rm -f vectorAdd_sequential vectorAdd_openmp