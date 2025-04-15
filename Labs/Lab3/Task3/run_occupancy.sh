#!/bin/bash
# run_occupancy.sh (final revision)
KERNELS=("vectorAdd_grid_stride" "vectorAdd_no_stride")
BLOCK_DIMS=(16 32 64 128 256)
OUTPUT_FILE="occupancy_results.csv"

# Correctly get the number of SMs (using dedicated parameter)
NUM_SMS=$(./occupancy_test get_num_sms 2>/dev/null)
if ! [[ "$NUM_SMS" =~ ^[0-9]+$ ]] || (( NUM_SMS < 1 )); then
  echo "Error: Unable to get the number of SMs (received: '$NUM_SMS')" >&2
  exit 1
fi

# Initialize output file
echo "GPU: $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | head -n1 | tr -d '\n'), SMs: $NUM_SMS" > "$OUTPUT_FILE"
echo "Kernel,BlockDim,GridDim,Occupancy" >> "$OUTPUT_FILE"

for kernel in "${KERNELS[@]}"; do
  for blk_dim in "${BLOCK_DIMS[@]}"; do
  # Calculate grid dimension (with validation)
  if [ "$kernel" == "vectorAdd_grid_stride" ]; then
    grid_dim=$(( (NUM_SMS * 8 * 256) / blk_dim ))
    (( grid_dim == 0 )) && grid_dim=1  # Ensure minimum grid dimension
  else
    grid_dim=$(( ( (1 << 20) + blk_dim - 1 ) / blk_dim ))
  fi

  echo "Measuring $kernel blk_dim=$blk_dim grid_dim=$grid_dim..."
  
  # Complete performance measurement command
  nvprof --metrics achieved_occupancy --csv --log-file tmp.csv ./occupancy_test $blk_dim $kernel 2>/dev/null
  
  # Accurately parse CSV output
  awk -F ',' -v kernel="$kernel" -v bd="$blk_dim" -v gd="$grid_dim" '
  /achieved_occupancy/ && $10 ~ /[0-9.]+/ {
    occ = $10 * 100  # Directly use the value in the fifth column
    printf "%s,%d,%d,%.2f%%\n", kernel, bd, gd, occ
  }' tmp.csv >> "$OUTPUT_FILE"
  
  rm tmp.csv
  done
done

echo "Results saved to $OUTPUT_FILE"