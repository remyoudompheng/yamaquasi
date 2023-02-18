// Copyright 2023 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Implementation of Pollard Rho algorithm (Brent variant)
//!
//! The Pollard Rho algorithm can be faster than Pollard P-1
//! for small prime factors:
//! - to find a 24-bit factor with 90% success we need
//!   about B1~20000 B2=2e6 with Pollard P-1
//! - to find a 32-bit factor with 90% success we need
//!   about B1~200e3 B2~200e6 with Pollard P-1
//! whereas Pollard rho runs in heuristic complexity O(sqrt(p))
//! requiring less operations.
//!
//! This implementation runs Pollard Rho with respectively
//! 1000 or 5000 iterations using available threads.
//! Each iteration requires 4 modular multiplications.
//!
//! Find a 20 bit factor, about 1000 iterations are necessary.
//! Find a 24 bit factor, about 4000 iterations are necessary.
//! Find a 28 bit factor, about 20000 iterations are necessary.
//!
//! In practice, it seems that running Pollard's Rho algorithm is only useful
//! when it is expected to completely factor the input number (which must be small)
//! otherwise it will be redundant with P-1 and ECM computations.
//!
//! References:
//! J.M. Pollard, A Monte Carlo method for factorization, 1975
//! R.P. Brent, An improved Monte Carlo factorization algorithm, 1980
//! Peter L. Montgomery, Speeding the Pollard and Elliptic Curve methods of Factorization
//! (Math. Comp. 48, 177, 1987)

use num_integer::Integer;

use crate::arith_montgomery::{gcd_factors, mg_2adic_inv, mg_mul, MInt, ZmodN};
use crate::{Uint, Verbosity};

/// Attempt to factor a 64-bit integer into 2 primes.
/// The argument is expected to have 2 prime factors of similar size.
///
/// The expected use case is to factor cofactors of the quadratic sieve.
/// Since the fallback is SQUFOF
pub fn rho_semiprime(n: u64) -> Option<(u64, u64)> {
    let sz = u64::BITS - u64::leading_zeros(n);
    match sz {
        0..=36 => rho64(n, 2, 500).or_else(|| rho64(n, 3, 500)),
        37..=42 => rho64(n, 2, 2500).or_else(|| rho64(n, 3, 2500)),
        43..=45 => rho64(n, 2, 5000).or_else(|| rho64(n, 3, 5000)),
        46..=49 => rho64(n, 2, 10000),
        50..=54 => rho64(n, 2, 20000),
        55.. => rho64(n, 2, 60000),
    }
}

pub fn rho64(n: u64, seed: u64, iters: u64) -> Option<(u64, u64)> {
    let ninv = mg_2adic_inv(n);
    // Perform x => x^2 + 1 on the Montgomery representation
    // So this is actually: xR => x^2 R + 1 where R=2^64.
    //
    // Invariants:
    // x1 = f^e1(seed) and x2=f^e2(seed)
    // 3/2 e1 <= e2 <= 2 e1 - 1
    let (mut x1, mut x2) = (seed, seed);
    let mut prod = 1;
    let mut next_interval_start = 0;
    let mut next_interval_end = 1;
    for e2 in 1..iters {
        x2 = mg_mul(n, ninv, x2, x2);
        x2 += 1; // we tolerate x2==n
        if e2 < next_interval_start {
            continue;
        }
        // We are in the interval, compare.
        prod = mg_mul(n, ninv, prod, x1.abs_diff(x2));

        if e2 == next_interval_end {
            // Set e1 = e2
            x1 = x2;
            // Next interval is (2^k + 2^(k-1), 2^(k+1) - 1) (length 2^(k-1))
            let pow2k = e2 + 1;
            debug_assert!(pow2k & (pow2k - 1) == 0);
            next_interval_start = pow2k + pow2k / 2;
            next_interval_end = 2 * pow2k - 1;
        }

        if e2 % 128 == 127 {
            let d = Integer::gcd(&n, &prod);
            if d > 1 && d < n {
                return Some((d, n / d));
            }
        }
    }
    let d = Integer::gcd(&n, &prod);
    if d > 1 && d < n {
        return Some((d, n / d));
    }
    None
}

pub fn rho(n: &Uint, verbosity: Verbosity) -> Option<(Vec<Uint>, Uint)> {
    let start = std::time::Instant::now();
    let size = n.bits();
    let n0 = n.digits()[0];
    let iters = match size {
        0..=24 => 100,
        25..=32 => 500,
        33..=48 => 2000,
        49..=64 => 20000,
        // Skip multiprecision inputs: factors will already be found
        // by P-1 and ECM. Even if Pollard rho is quick, it is redundant
        // with other fast methods.
        _ => return None,
    };
    // Use a varying seed to avoid redundancy.
    let seed = (n0 & 31) + 1;
    let (p, q) = rho64(n0, seed, iters)?;
    if verbosity >= Verbosity::Info {
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        eprintln!("Found factor {p} with Pollard rho (seed={seed} iters={iters}) in {ms:.1}ms");
    }
    Some((vec![p.into()], q.into()))
}

#[doc(hidden)]
pub fn rho_impl(
    n: &Uint,
    seed: u64,
    iters: u64,
    verbosity: Verbosity,
) -> Option<(Vec<Uint>, Uint)> {
    let start = std::time::Instant::now();
    assert!(n.bits() >= 64 || seed < n.digits()[0]);
    // The modulus/ring can shrink as we find factors.
    let zn = ZmodN::new(*n);
    let s = zn.from_int(seed.into());
    let mut x1 = s;
    let mut x2 = s;
    // This is not really 1, because we are using Montgomery
    // representation, but it is easier (and it is also a square).
    let one = MInt::from_uint(Uint::ONE);
    let mut prods = Vec::with_capacity(iters as usize);
    let mut prod = zn.one();
    for _ in 0..iters {
        // x1 => x1^2 + one
        x1 = zn.mul(&x1, &x1);
        x1 = zn.add(&x1, &one);
        // Twice x2 => x2^2 + one
        x2 = zn.mul(&x2, &x2);
        x2 = zn.add(&x2, &one);
        x2 = zn.mul(&x2, &x2);
        x2 = zn.add(&x2, &one);
        prod = zn.mul(&prod, &zn.sub(&x2, &x1));
        prods.push(prod);
    }
    let (fs, nred) = gcd_factors(&n, &prods);
    if nred == Uint::ONE || &nred == n {
        return None;
    }
    if verbosity >= Verbosity::Info {
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "Found factors {:?} with Pollard rho (seed={seed} iters={iters}) in {ms:.1}ms",
            fs
        );
    }
    Some((fs, nred))
}

#[test]
fn test_rho_basic() {
    let ns: &[u64] = &[
        235075827453629,
        166130059616737,
        159247921097933,
        224077614412439,
        219669028971857,
    ];
    'nextn: for &n in ns {
        for budget in [500, 1000, 1500, 2000, 4000, 7000, 10000, 20000] {
            if let Some((p, q)) = rho64(n, 2, budget) {
                eprintln!("factored {n} with budget {budget} => {p}*{q}");
                assert_eq!(p * q, n);
                continue 'nextn;
            }
        }
        panic!("failed to factor {n}");
    }
}

#[test]
fn test_rho_random() {
    use crate::fbase;

    for bits in [16, 20, 22, 24, 26, 28] {
        let mut seed = 1234567_u32;
        let mut samples = vec![];
        let mut primes = vec![];
        while samples.len() < 300 {
            seed = seed.wrapping_mul(123456789);
            let p = seed % (1 << bits);
            if !fbase::certainly_composite(p as u64) {
                primes.push(p);
            }
            if primes.len() == 2 {
                let p = primes[0];
                let q = primes[1];
                samples.push(p as u64 * q as u64);
                primes.clear();
            }
        }

        let size = 2 * bits;
        let total = samples.len();
        for budget in [500, 1000, 2000, 5000, 10000, 20000, 40000, 60000] {
            let mut ok = 0;
            let start = std::time::Instant::now();
            for &n in &samples {
                if let Some((x, y)) = rho64(n, 2, budget) {
                    assert_eq!(x * y, n);
                    ok += 1;
                }
            }
            let elapsed = start.elapsed().as_secs_f64() * 1000.;
            eprintln!(
                "{size} bits, budget={budget} factored {ok}/{total} semiprimes in {elapsed:.2}ms"
            );

            ok = 0;
            let start = std::time::Instant::now();
            for &n in &samples {
                if let Some((x, y)) = rho64(n, 2, budget / 2).or_else(|| rho64(n, 3, budget / 2)) {
                    assert_eq!(x * y, n);
                    ok += 1;
                }
            }
            let elapsed = start.elapsed().as_secs_f64() * 1000.;
            eprintln!(
                "{size} bits, budget={budget}(2 half) factored {ok}/{total} semiprimes in {elapsed:.2}ms"
            );
        }
    }
}
