## Goals in the Lab3:

### Task1: Profiling

Lab3/Task1/vectorAdd.cu is the program implemented last week.

1. Use make vectorAdd to compile the file.

2. First, use sbatch job1.sh to show all the metrics of nvprof.

3. Use sbatch job2.sh to get the profiling result.

### Task2: Occupancy

1. Students can use cudaGetDeviceProperties.cu to get the Properties of the device they use, such as maxThreadsPerBlock and maxBlocksPerMultiProcessor.

```sh
make TARGET=cudaGetDeviceProperties
./cudaGetDeviceProperties
```

2. First calculate the theoretical occupancy.

3. Then use vectorAdd_adjust.cu as an example to measure its occupancy. (usage: ./vectorAdd_adjust_blkdim.cu block_dim)

### Task3: Stride Loop

1. Comparison of Execution Time

```sh
nvcc vector_add_benchmark.cu -o benchmark
./benchmark
```

2. Comparison in Achieved Occupancy

Add execute permission and then run

```sh
chmod +x run_occupancy.sh
make
make run
```

### Task4: Tiled MMM

1. The NaiveMatrixMul.cu will do the MMM directly (without using shared memory).

2. The TiledMatrixMul.cu will do the MMM with shared memory.