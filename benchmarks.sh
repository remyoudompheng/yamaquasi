#!/bin/bash

MODE=${1:-mpqs}

for bits in 40 60 80 100 120 140; do
	# Prepare random inputs
	for i in {0..50}; do
	    python semiprimes.py $bits
        done 1> testinputs.txt 2>/dev/null
	args=${args%,}
	echo "=== INPUT $bits bits ==="
	hyperfine -m 10 "bin/ymqs --mode $MODE "'$(shuf -n 1 testinputs.txt)'
done
