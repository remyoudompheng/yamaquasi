"""
Computation of group structure from relations

This script requires SageMath and scipy, networkx
libraries (already dependencies of SageMath).

The following steps are followed:
* prune: remove primes appearing 1x or 2x
  (requires networkx)
* filter: run structured Gaussian elimination
* classnumber: compute the class number
  (requires SageMath and scipy)
* group structure: compute the group presentation
  (requires SageMath and optionally Cado-NFS)
* group structure extra: compute extra relations

Each step is run as a separate Python process: the memory usage
should never exceed 4GB.
"""

import argparse
import json
import os
from pathlib import Path
from multiprocessing import Process


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("-j", type=int, default=2, help="Number of threads")
    argp.add_argument(
        "--sage",
        action="store_true",
        help="Allow using Sage sparse linear algebra (slow)",
    )
    argp.add_argument("DATADIR", help="Directory containing data files")
    args = argp.parse_args()

    if not args.sage and not os.getenv("CADONFS_BWCDIR"):
        print("If the class number has a large prime factor, Cado-NFS is necessary to")
        print("achieve good performance. Use --sage option to ignore this warning.")
        exit(1)

    if os.getenv("CADONFS_BWCDIR"):
        bwc = Path(os.getenv("CADONFS_BWCDIR"))
        files = ["bwc.pl", "krylov", "mksol", "mf_scan2"]
        for f in files:
            if not (bwc / f).is_file():
                print(f"Could not find mandatory Cado-NFS program {bwc / f}")
                exit(1)

    datadir = Path(args.DATADIR)
    assert (datadir / "args.json").is_file()
    assert (datadir / "relations.sieve").is_file()

    with open(datadir / "args.json") as f:
        meta = json.load(f)

    # To save memory and clean extra imports, each step runs as a separate process
    if not (datadir / "relations.pruned").is_file():
        proc = Process(target=prune, args=[datadir])
        proc.start()
        proc.join()

    if not (datadir / "relations.filtered").is_file():
        proc = Process(target=filter, args=[datadir, meta])
        proc.start()
        proc.join()

    if not (datadir / "classnumber").is_file():
        proc = Process(target=classno, args=[datadir, meta, args.j])
        proc.start()
        proc.join()

    if not (datadir / "group.structure").is_file():
        proc = Process(target=group, args=[datadir, meta, args.sage])
        proc.start()
        proc.join()

    if not (datadir / "group.structure.extra").is_file():
        proc = Process(target=groupextra, args=[datadir, meta])
        proc.start()
        proc.join()


def prune(datadir):
    from step1_filter import step_prune

    step_prune(datadir)


def filter(datadir, meta):
    from step1_filter import step_filter

    step_filter(datadir, meta)


def classno(datadir, meta, threads):
    from step2_classno import classno

    classno(datadir, meta, threads)


def group(datadir, meta, sage):
    from step3_group import structure

    structure(datadir, meta, sage=sage)


def groupextra(datadir, meta):
    from step3_group import extra

    extra(datadir, meta)


if __name__ == "__main__":
    main()
