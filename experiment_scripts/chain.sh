#!/bin/bash

script=$1
chain_len=$2

id=$(sbatch --parsable $script)

for ((i=1; i < $chain_len; i++)); do
    id=$(sbatch --parsable --dependency=afterany:$id $script)
done
