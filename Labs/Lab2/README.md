# Goals in Lab2

## CUDA Programming

By using command `make filename` in each Task folder, it can automatically compile the file. 

### Task1: cudaMemcpy
The vectorAdd_not_Mem.cu file doesn't move the array from host to device, so it gets error. Student need to modify it to get the "Test PASSED".

--> Modify the vectorAdd_no_Mencpy.cu

### Task2: Multi-Block Execution

Then the vectorAdd.cu still have oen block so the index is simple. Student need to modify the code to use more then one block.

--> Modify the vectorAdd_more_blk_exercise.cu

--> The answer is the vectorAdd_more_blk_solution.cu

### Task3: Unified Memory

1. Measure the execution time in vectorAdd_more_blk_unified_memory.cu

2. Change the position of timer to include data transfer time in vectorAdd_more_blk_solution.cu

### Task4: Grid stride-loop methodology

This method is particularly useful for large datasets where the number of elements exceeds the number of threads available. It allows for efficient utilization of the GPU resources by distributing the workload evenly across all threads.

--> Modify the vectorAdd_more_blk.cu

--> The answer is the grid_strike_vectorAdd.cu
