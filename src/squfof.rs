// Copyright 2022 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Implementation of Shanks's square forms factorization
//!
//! References: http://homes.cerias.purdue.edu/~ssw/squfof.pdf

use num_integer::Integer;

pub fn squfof(n: u64) -> Option<(u64, u64)> {
    // Loop over multipliers
    'kloop: for k in 1..=50 {
        let nsqrt = isqrt(n * k);
        if nsqrt * nsqrt == n {
            return Some((nsqrt, nsqrt));
        }

        let iters = 3 * isqrt(nsqrt);

        let mut p_prev = nsqrt;
        let mut q_prev = 1;
        let mut q = n * k - nsqrt * nsqrt;
        let mut q_sqrt = 0;

        for i in 1..=iters {
            if i == iters {
                // Failure
                continue 'kloop;
            }
            let b = (nsqrt + p_prev) / q;
            let p = b * q - p_prev;
            //eprintln!("i={i}, bi={b} Pi={p} Qi={q}");
            let qnext = if p_prev > p {
                q_prev + b * (p_prev - p)
            } else {
                q_prev - b * (p - p_prev)
            };
            if maybe_square(qnext) {
                let qsqrt = isqrt(qnext);
                if qnext == qsqrt * qsqrt && i % 2 == 1 {
                    q_sqrt = qsqrt;
                    p_prev = p;
                    break;
                }
            }
            // next iteration
            p_prev = p;
            q_prev = q;
            q = qnext;
        }

        let b = (nsqrt - p_prev) / q_sqrt;
        let mut p_prev = b * q_sqrt + p_prev;
        let mut q_prev = q_sqrt;
        let mut q = (n * k - p_prev * p_prev) / q_prev;
        for i in 1..=iters {
            if i == iters {
                // Failure
                continue 'kloop;
            }
            let b = (nsqrt + p_prev) / q;
            let p = b * q - p_prev;
            //eprintln!("i={i}, bi={b} Pi={p} Qi={q}");
            let qnext = if p_prev > p {
                q_prev + b * (p_prev - p)
            } else {
                q_prev - b * (p - p_prev)
            };
            if p == p_prev {
                break;
            }
            // next iteration
            p_prev = p;
            q_prev = q;
            q = qnext;
        }
        let f = Integer::gcd(&n, &p_prev);
        //eprintln!("final p={p_prev} gcd={f}");
        if f > 1 {
            assert_eq!(n % f, 0);
            return Some((f, n / f));
        }
    }
    None
}

fn maybe_square(n: u64) -> bool {
    (n & 6 == 0 || n & 7 == 4) && (n + 1) % 5 <= 2
}

// Returns floored square root of n.
fn isqrt(n: u64) -> u64 {
    if n < 4 {
        return std::cmp::min(n, 1);
    }
    let mut r = (n as f64).sqrt() as u64;
    loop {
        let q = n / r;
        if q == r || q == r + 1 {
            return r;
        }
        if q == r - 1 {
            return r - 1;
        }
        r = (r + q) / 2;
        // r^2 ~= n/2 + (r^2 + n/r^2)/2 >= n
    }
}

// Tabulated square roots of 256.

#[test]
fn test_squfof() {
    let ns: &[u64] = &[
        11111,
        235075827453629,
        166130059616737,
        159247921097933,
        224077614412439,
        219669028971857,
    ];
    for &n in ns {
        eprintln!("n={} => {:?}", n, squfof(n).unwrap());
    }

    // Random products
    for i in 0..50 {
        for j in 0..50 {
            let p = 123456789 + i * 2468;
            let q = 198765431 + j * 1590;
            let Some((x,y))  = squfof(p*q)
                else { panic!("failed for {p}*{q}") };
            assert!(x > 1 && y > 1 && x * y == p * q);
        }
    }

    // Primes
    let ps: &[u64] = &[1429332497];
    for &p in ps {
        assert_eq!(squfof(p), None);
    }
}

#[test]
fn test_isqrt() {
    for n in 0..=500_000 {
        let r = isqrt(n);
        assert!(r * r <= n && n < (r + 1) * (r + 1));
    }
    for k in 0..=500_000 {
        let n = 123456789 + 1234 * k;
        let r = isqrt(n);
        assert!(r * r <= n && n < (r + 1) * (r + 1));
    }
}
