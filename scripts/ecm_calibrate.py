from concurrent.futures import ProcessPoolExecutor
import random
import sys
import time

import cypari2
import gmpy2
import pymqs

pari = cypari2.Pari()
pari.allocatemem(90_000_000)

params = [
    # (minsize, maxsize, cost class, B1, B2)
    (0, 24, 300, 10, 480.0),
    (0, 24, 300, 12, 252.0),
    (0, 24, 500, 10, 660.0),
    (0, 24, 700, 30, 660.0),
    (0, 24, 700, 10, 1080.0),
    (0, 32, 1000, 10, 5040.0),
    (0, 32, 1000, 40, 1920.0),
    (0, 32, 1000, 50, 660.0),
    (0, 40, 2000, 40, 7700.0),
    (0, 40, 2000, 80, 5040.0),
    (0, 40, 2000, 100, 1920.0),
    (0, 48, 4000, 60, 20000.0),
    (0, 48, 4000, 120, 13200.0),
    (0, 48, 4000, 200, 7700.0),
    (0, 56, 8000, 200, 33000.0),
    (0, 56, 8000, 300, 20000.0),
    (0, 56, 8000, 400, 13200.0),
    (16, 64, 16000, 400, 81000.0),
    (16, 64, 16000, 800, 33000.0),
    (16, 64, 16000, 1000, 20000.0),
    (16, 72, 32000, 800, 180_000.0),
    (16, 72, 32000, 1600, 81_000.0),
    (16, 72, 32000, 2500, 53_000.0),
    (24, 72, 50000, 1500, 320_000.0),
    (24, 72, 50000, 3000, 126_000.0),
    (24, 72, 50000, 4000, 81_000.0),
    (32, 80, 100000, 3000, 550_000.0),
    (32, 80, 100000, 5000, 320_000.0),
    (32, 80, 100000, 7000, 180_000.0),
    (32, 88, 200000, 5000, 1.4e6),
    (32, 88, 200000, 10000, 800e3),
    (32, 88, 200000, 15000, 320e3),
    # Cost = 13.7 B1 + 20 sqrt(B2) log(B2)
    (32, 96, 500000, 9000, 7e6),
    (32, 96, 500000, 15000, 4.5e6),
    (32, 96, 500000, 27000, 1.4e6),
    (32, 96, 500000, 40000, 554e3),
    (48, 104, 1000000, 15000, 30e6),
    (48, 104, 1000000, 25000, 20e6),
    (48, 104, 1000000, 50000, 7.1e6),
    (48, 104, 1000000, 75000, 4.7e6),
    (56, 104, 2000000, 50000, 38e6),
    (56, 104, 2000000, 70000, 19e6),
    (64, 112, 5000000, 120000, 322e6),
    (64, 112, 5000000, 200000, 117e6),
    (64, 112, 5000000, 300000, 38e6),
    (80, 120, 10_000_000, 250000, 1.3e9),
    (80, 120, 10_000_000, 500000, 643e6),
    (80, 120, 10_000_000, 800000, 156e6),
    (88, 128, 20_000_000, 600000, 5.2e9),
    (88, 128, 20_000_000, 1000000, 2.6e9),
    (88, 128, 20_000_000, 1500000, 643e6),
    (96, 136, 50_000_000, 2000000, 32.5e9),
    (96, 136, 50_000_000, 2500000, 21.6e9),
    (96, 136, 50_000_000, 4000000, 5.2e9),
    (104, 144, 100_000_000, 4000000, 136e9),
    (104, 144, 100_000_000, 6000000, 43e9),
    (104, 144, 100_000_000, 8000000, 21e9),
    (108, 160, 200_000_000, 6000000, 543e9),
    (108, 160, 200_000_000, 10000000, 136e9),
    (108, 160, 200_000_000, 15000000, 42e9),
    (108, 160, 500_000_000, 15000000, 3e12),
    (108, 160, 500_000_000, 25000000, 724e9),
    (108, 160, 500_000_000, 40000000, 362e9),
]


def curve_order(l):
    # Rescaled Suyama σ=11 curve
    E = pari.ellinit([0, -679442, 0, 71973621841, 0], l)
    # Rescaled Suyama σ=9/4 curve
    # E = pari.ellinit([0, 429857277, 0, 331776, 0], l)
    order = int(pari.ellcard(E))
    assert order % 12 == 0
    return order


# SAMPLES = 5_000
SAMPLES = 100_000


def main():
    size = int(sys.argv[1])

    inputs = []
    t0 = time.monotonic()
    jobs = []
    if True:
        for _ in range(SAMPLES):
            l = gmpy2.next_prime(random.getrandbits(size - 1) | (1 << (size - 1)))
            order = curve_order(l)
            facs = pymqs.factor(order)
            # FIXME: in case output is not sorted
            facs.sort()
            p1 = facs[-1]
            p2 = facs[-2]
            assert p2 <= p1

            inputs.append((l, order, p1, p2))
    else:
        with ProcessPoolExecutor(max_tasks_per_child=1000) as pool:
            for _ in range(SAMPLES):
                l = gmpy2.next_prime(random.getrandbits(size - 1) | (1 << (size - 1)))
                jobs.append((l, pool.submit(curve_order, l)))
            for l, job in jobs:
                order = job.result()
                facs = pymqs.factor(order)
                # FIXME: in case output is not sorted
                facs.sort()
                p1 = facs[-1]
                p2 = facs[-2]
                assert p2 <= p1

                inputs.append((l, order, p1, p2))
    t1 = time.monotonic()
    print(f"Computed {SAMPLES} curve orders in {t1 - t0:.6}s")

    for minsz, maxsz, cost, b1, b2 in params:
        if size < minsz or size > maxsz:
            continue

        rate = 0
        for l, order, p1, p2 in inputs:
            if p2 <= b1 and p1 < b2:
                rate += 1
        rate /= float(len(inputs))
        # Benchmark
        if cost < 20e6:
            ls = [args[0] for args in inputs[:10]]
        else:
            ls = [args[0] for args in inputs[:4]]
        qsize = size + max(size // 2, 32)
        q = gmpy2.next_prime(random.getrandbits(qsize - 1) | (1 << (qsize - 1)))
        t0 = time.monotonic()
        for l in ls:
            try:
                pymqs.ecm(int(l * q), 1, b1, b2)
            except Exception:
                pass
        t1 = time.monotonic()
        speed = (t1 - t0) / float(len(ls))
        eta = speed / rate if rate else float("inf")

        if size <= 64:
            print(
                cost,
                f"dt={speed * 1000:.3f}ms",
                b1,
                b2,
                f"rate={rate:.3f} eta={eta * 1000:.3f}ms",
            )
        else:
            print(
                cost,
                f"dt={speed * 1000:.1f}ms",
                b1,
                b2,
                f"rate={rate * 100:.3f}% eta={eta:.3f}s",
            )


if __name__ == "__main__":
    main()
