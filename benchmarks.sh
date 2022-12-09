#!/bin/bash

MODE=${1:-mpqs}

case $MODE in
    msieve)
        COMMAND="msieve -l /dev/null -v"
        ;;
    flintqs)
        COMMAND="QuadraticSieve <<<"
        ;;
    qs|mpqs|siqs)
        COMMAND="bin/ymqs --mode $MODE"
        ;;
    *)
        echo Unknown mode $MODE
        exit 1
        ;;
esac

for bits in 40 60 80 100 120 140 160 180 200 220; do
    # Prepare random inputs
    for i in {0..50}; do
        python semiprimes.py $bits
    done 1> testinputs.txt 2>/dev/null
    args=${args%,}
    echo "=== INPUT $bits bits ==="
    hyperfine -i -m 10 -p "rm -f msieve.dat" "$COMMAND "'$(shuf -n 1 testinputs.txt)'
done
