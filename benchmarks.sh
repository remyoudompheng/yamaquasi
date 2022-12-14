#!/bin/bash

MODE=${1:-mpqs}

case $MODE in
    msieve)
        COMMAND="msieve -l /dev/null -v"
        ;;
    flintqs)
        COMMAND="QuadraticSieve <<<"
        ;;
    qs64|qs|mpqs|siqs)
        COMMAND="bin/ymqs --mode $MODE"
        ;;
    mpqs6)
        COMMAND="bin/ymqs --mode mpqs --threads 6"
        ;;
    siqs6)
        COMMAND="bin/ymqs --mode siqs --threads 6"
        ;;
    *)
        echo Unknown mode $MODE
        exit 1
        ;;
esac

for bits in 40 60 80 100 120 140 160 180 200 220 240 260; do
    # Prepare random inputs
    for i in {0..50}; do
        python semiprimes.py $bits
    done 1> testinputs$bits.txt 2>/dev/null
    args=${args%,}
    echo "=== INPUT $bits bits ==="
    hyperfine -i -m 10 -S "bash --norc" -p "rm -f msieve.dat" "$COMMAND \$(shuf -n 1 testinputs$bits.txt)"
done
