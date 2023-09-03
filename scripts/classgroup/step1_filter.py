"""
This step implementing pruning/filtering/merging of relations
produced by sieving.

The method uses simple scoring (similar to [Biasse]) without
advanced optimizations.
"""

from pathlib import Path
import time

from networkx import Graph, connected_components


def step_prune(datadir):
    print("=> PRUNING STEP")
    assert (datadir / "relations.sieve").is_file()
    t0 = time.time()
    rels = read_relations_flat(datadir / "relations.sieve")
    print("Imported", len(rels), f"relations in {time.time() - t0:.3f}s")

    # FIXME: save pruned relations

    # Prune relations in place: a removed relation is replaced by None.
    # We are only interested in coefficients ±1
    stats = {}
    for ridx, r in enumerate(rels):
        for p, v in r.items():
            if abs(v) > 1:
                stats[p] = None
                continue
            stats.setdefault(p, [])
            if stats[p] is None:
                continue
            stats[p].append(ridx)
            if len(stats[p]) > 20:
                stats[p] = None

    excess = len(rels) - len(stats)
    print(f"{len(stats)} primes appear in relations")

    removed = 0
    max_removed = (excess - 200) // 2

    def prune(ridx):
        r = rels[ridx]
        for p, v in r.items():
            if abs(v) == 1:
                sp = stats[p]
                if sp is not None:
                    sp.remove(ridx)
        rels[ridx] = None

    def score(clique):
        s = 0
        for ridx in clique:
            # Score is weight of relation
            # + bonus point is some primes have low weight.
            r = rels[ridx]
            s += len(r)
            for p in r:
                if (sp := stats[p]) is not None and len(sp) < 5:
                    s += 1
        return s

    while removed < max_removed:
        m1 = [p for p, rs in stats.items() if rs is not None and len(rs) == 1]
        singles = 0
        for p in m1:
            if stats[p]:
                prune(stats[p][0])
                singles += 1
        if singles:
            print(f"pruned {singles} singletons")

        m2 = [p for p, rs in stats.items() if rs is not None and len(rs) == 2]
        g = Graph()
        for p in m2:
            g.add_edge(*stats[p])
        # They are not cliques at all but the term is used in literature.
        cliques = list(connected_components(g))
        cliques.sort(key=score)
        to_remove = max(100, max_removed // 4)
        to_remove = min(max_removed - removed, to_remove)
        assert to_remove > 0
        cliques_removed = cliques[-to_remove:]
        size = sum(len(c) for c in cliques_removed)
        if size:
            print(f"pruning {len(cliques_removed)} cliques of {size} relations")
        for c in cliques_removed:
            for ridx in c:
                prune(ridx)
        removed += len(cliques_removed)
        if not singles and not size:
            break

    cols = set()
    rels = [r for r in rels if r is not None]
    for r in rels:
        cols.update(r)
    print(f"After pruning: {len(rels)} relations with {len(cols)} primes")

    with open(datadir / "relations.pruned", "w") as w:
        for r in rels:
            items = []
            for p in sorted(r):
                if (e := r[p]) > 0:
                    [items.append(p) for _ in range(e)]
                else:
                    [items.append(-p) for _ in range(-e)]
            w.write(" ".join(str(x) for x in items))
            w.write("\n")
        print("Relations written to", w.name)


def step_filter(datadir, meta):
    print("=> FILTERING STEP")

    # Density limit
    D = int(meta["d"])
    dense_limit = D.bit_length() // 2

    assert (datadir / "relations.pruned").is_file()
    t0 = time.time()
    rels = read_relations_flat(datadir / "relations.pruned")
    print("Imported", len(rels), f"relations in {time.time() - t0:.3f}s")

    stats = {}
    for ridx, r in enumerate(rels):
        for p in r:
            stats.setdefault(p, set()).add(ridx)

    def addstat(ridx, r):
        for p in r:
            stats.setdefault(p, set()).add(ridx)

    def delstat(ridx, r):
        for p in r:
            stats[p].remove(ridx)
            if not stats[p]:
                stats.pop(p)

    def pivot(piv, r, p):
        assert abs(piv[p]) == 1
        k = r[p] * piv[p]
        out = {}
        for l in r:
            if l in piv:
                e = r[l] - k * piv[l]
                if e:
                    out[l] = e
            else:
                out[l] = r[l]
        for l in piv:
            if l not in r:
                out[l] = -k * piv[l]
        assert p not in out
        return out

    excess = len(rels) - len(stats)
    print(f"{len(stats)} primes appear in relations")
    print(f"{excess} relations can be removed")

    # prime p = product(l^e)
    saved_pivots = []

    Ds = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]
    Ds += [25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
    t = time.time()
    removed = 0
    for d in Ds:
        remaining = [_r for _r in rels if _r is not None]
        avgw = sum(len(r) for r in remaining) / len(remaining)
        maxe = max(abs(e) for r in remaining for _, e in r.items())
        nc, nr = len(stats), len(remaining)
        assert nr > nc
        print(
            f"Starting {d}-merge: {nc} columns {nr} rows excess={nr-nc} weight={avgw:.3f} maxcoef={maxe} elapsed={time.time() - t:.1f}s"
        )

        if d > nc // 3:
            # Matrix is too small
            break

        # Modulo p^k we have probabikity 1/p of missing a generator
        # for each excess relation
        MIN_EXCESS = 64 + D.bit_length()
        while True:
            # d-merges
            md = [k for k in stats if len(stats[k]) <= d]
            if not md:
                break
            print(f"{len(md)} {d}-merges candidates {min(md)}..{max(md)}")
            merged = 0
            for p in md:
                rs = stats.get(p)
                if not rs or len(rs) > d:
                    # prime already eliminated or weight has grown
                    continue
                # Pivot has fewest coefficients and pivot value is ±1
                assert all(p in rels[ridx] for ridx in stats[p])
                rs = sorted(rs, key=lambda ridx: (abs(rels[ridx][p]), len(rels[ridx])))
                pividx = rs[0]
                piv = rels[pividx]
                if abs(piv[p]) > 1:
                    print(f"skip pivoting on {p}")
                    continue
                for ridx in rs[1:]:
                    rp = pivot(piv, rels[ridx], p)
                    delstat(ridx, rels[ridx])
                    addstat(ridx, rp)
                    rels[ridx] = rp
                # Remove and save pivot
                delstat(pividx, piv)
                rels[pividx] = None
                saved_pivots.append(
                    (p, {l: e * -piv[p] for l, e in piv.items() if l != p})
                )
                removed += 1
                assert p not in stats
                # FIXME: print pivot
                merged += 1

            if not merged:
                break
            print(f"{merged} pivots done")

        remaining = [_r for _r in rels if _r is not None]
        nr, nc = len(remaining), len(stats)
        avgw = sum(len(r) for r in remaining) / nr

        def score_sparse(rel, stats):
            return len(rel) + sum(1 for l in rel if len(stats[l]) < 2 * d)

        stop = avgw > dense_limit
        # Remove most annoying relations
        excess = nr - nc
        if stop:
            break
        if excess > MIN_EXCESS:
            to_remove = (excess - MIN_EXCESS) // (len(Ds) // 2)
            if d < 10:
                # Still actively merging
                to_remove = 0
            if to_remove:
                scores = []
                for ridx, r in enumerate(rels):
                    if r is None:
                        continue
                    scores.append((score_sparse(r, stats), ridx))
                scores.sort()
                worst = scores[-to_remove:]
                print(
                    f"Worst rows ({len(worst)}) have score {worst[0][0]:.3f}..{worst[-1][0]:.3f}"
                )
                for _, ridx in worst:
                    # Not a pivot, no need to save.
                    delstat(ridx, rels[ridx])
                    rels[ridx] = None

    # For the last step, we just want to minimize length.
    if excess > MIN_EXCESS:
        scores = [(len(r), ridx) for ridx, r in enumerate(rels) if r is not None]
        scores.sort()
        to_remove = excess - MIN_EXCESS
        worst = scores[-to_remove:]
        print(
            f"Worst rows ({len(worst)}) have score {worst[0][0]:.3f}..{worst[-1][0]:.3f}"
        )
        for _, ridx in worst:
            # Not a pivot, no need to save.
            delstat(ridx, rels[ridx])
            rels[ridx] = None

    rels = [_r for _r in rels if _r is not None]
    nr, nc = len(rels), len(stats)
    avgw = sum(len(r) for r in rels) / len(rels)
    maxe = max(abs(e) for r in rels for e in r.values())
    dt = time.time() - t0
    print(
        f"Final: {nc} columns {nr} rows excess={nr-nc} weight={avgw:.3f} maxcoef={maxe} elapsed={dt:.1f}s"
    )
    # Dump result
    with open(datadir / "relations.removed", "w") as w:
        for p, rel in reversed(saved_pivots):
            line = f"{p} = " + " ".join(f"{l}^{e}" for l, e in sorted(rel.items()))
            w.write(line)
            w.write("\n")
        print(f"{len(saved_pivots)} removed relations written to", w.name)

    with open(datadir / "relations.filtered", "w") as w:
        for r in rels:
            line = " ".join(f"{l}^{e}" for l, e in sorted(r.items()))
            w.write(line)
            w.write("\n")
        print(f"{len(rels)} relations written to", w.name)


def read_relations_flat(pth: Path):
    """
    Read a flat relation file (primes with sign)
    They are read as dictionaries.
    """
    # Intern integers:
    primes = {}
    rels = []
    with open(pth) as f:
        for l in f:
            if l.isspace():
                continue
            v = {}
            for w in l.split():
                p = int(w)
                k = abs(p)
                k = primes.setdefault(k, k)
                v.setdefault(k, 0)
                if p < 0:
                    v[k] -= 1
                else:
                    v[k] += 1
            rels.append(v)
    return rels
