## Goals in the Lab3:

### Task1: Profiling

`Lab3/Task1/vectorAdd.cu` is the program implemented last week.

1. Use `make vectorAdd` to compile the file.

2. First, use `sbatch job1.sh` to show all the metrics of nvprof.

3. Use `sbatch job2.sh` to get the profiling result.

### Task2: Occupancy

1. Students can use cudaGetDeviceProperties.cu to get the Properties of the device they use, such as maxThreadsPerBlock and maxBlocksPerMultiProcessor.

```sh
sbatch job1.sh
```

2. First calculate the theoretical occupancy.

3. Then use `sbatch job2.sh` to measure both its warp_execution_efficiency and achieved_occupancy . (usage: ./vectorAdd_adjust_blkdim.cu block_dim)

### Task3: Stride Loop

1. Measure the occupancy of vectorADD without using Grid Stride Loop Methodology

```sh
make vectorAdd_no_stride
sbatch job1.sh 
```

2. Comparison in Achieved Occupancy


```sh
sbatch job2.sh 
```

### Task4: Tiled MMM

1. The NaiveMatrixMul.cu will do the MMM directly (without using shared memory).

2. The TiledMatrixMul.cu will do the MMM with shared memory.
