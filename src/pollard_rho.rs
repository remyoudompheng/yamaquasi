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
//! Find a 20 bit factor, about 4000 iterations are necessary.
//! Find a 24 bit factor, about 16000 iterations are necessary.
//! Find a 28 bit factor, about 65000 iterations are necessary.
//!
//! In practice, it seems that running Pollard's Rho algorithm is only useful
//! when it is expected to completely factor the input number (which must be small)
//! otherwise it will be redundant with P-1 and ECM computations.
//! Therefore it is not used in the general factoring logic for large integers.
//!
//! References:
//! J.M. Pollard, A Monte Carlo method for factorization, 1975
//! R.P. Brent, An improved Monte Carlo factorization algorithm, 1980
//! Peter L. Montgomery, Speeding the Pollard and Elliptic Curve methods of Factorization
//! (Math. Comp. 48, 177, 1987)

use num_integer::Integer;

use crate::arith_montgomery::{gcd_factors, mg_2adic_inv, mg_mul, MInt, ZmodN};
use crate::{Uint, Verbosity};

/// Attempt to factor a double large prime from quadratic sieve
///
/// The argument is expected to have 2 large prime factors.
/// The smallest prime factor usually has 19-30 bits.
/// This function focuses on factors under 25 bits.
pub fn rho_semiprime(n: u64) -> Option<(u64, u64)> {
    if n >> 40 == 0 {
        rho64(n, 1, 2048)
            .or_else(|| rho64(n, 2, 2048))
            .or_else(|| rho64(n, 3, 2048))
    } else if n >> 48 == 0 {
        rho64(n, 1, 4096)
            .or_else(|| rho64(n, 2, 4096))
            .or_else(|| rho64(n, 3, 4096))
    } else {
        // Don't try too much
        rho64(n, 1, 8192)
    }
}

// Run Pollard's rho algorithm.
//
// Due to the structure of Brent's cycle finding algorithm,
// iters should be slightly less than a power of 2 (interval [1.5*2^k, 2^k])
#[doc(hidden)]
pub fn rho64(n: u64, c: u64, iters: u64) -> Option<(u64, u64)> {
    let ninv = mg_2adic_inv(n);
    // Perform x => x^2 + c on the Montgomery representation
    // So this is actually: xR => x^2 R + c where R=2^64.
    //
    // The seed is always 2.
    //
    // Invariants:
    // x1 = f^e1(seed) and x2=f^e2(seed)
    // 3/2 e1 <= e2 <= 2 e1 - 1
    let (mut x1, mut x2) = (2_u64, 2_u64);
    let mut prod = 1;
    let mut next_interval_start = 0;
    let mut next_interval_end = 1;
    for e2 in 1..iters {
        x2 = mg_mul(n, ninv, x2, x2);
        x2 += c; // we tolerate x2==n (it will be absorbed by next mg_mul)
        if e2 < next_interval_start {
            continue;
        }
        // We are in the interval, compare.
        let prodnext = mg_mul(n, ninv, prod, x1.abs_diff(x2));
        if prodnext == 0 {
            // Probably the previous value had a nontrivial GCD with n
            // and the remaining factor was multiplied in.
            let d = Integer::gcd(&n, &x1.abs_diff(x2));
            if d > 1 && d < n {
                return Some((d, n / d));
            }
        }
        if e2 >= 512 && e2 % 128 == 127 {
            let d = Integer::gcd(&n, &prod);
            if d > 1 && d < n {
                return Some((d, n / d));
            }
        }
        prod = prodnext;
        if e2 == next_interval_end {
            // Set e1 = e2
            x1 = x2;
            // Next interval is (2^k + 2^(k-1), 2^(k+1) - 1) (length 2^(k-1))
            let pow2k = e2 + 1;
            debug_assert!(pow2k & (pow2k - 1) == 0);
            next_interval_start = pow2k + pow2k / 2;
            next_interval_end = 2 * pow2k - 1;
        }
    }
    let d = Integer::gcd(&n, &prod);
    if d > 1 && d < n {
        return Some((d, n / d));
    }
    None
}

/// Run Pollard's rho as part of general factorization.
///
/// It is both suitable for small prime factors and small semiprimes,
/// so it will attempt to retry several times to find a factor.
pub fn rho(n: &Uint, verbosity: Verbosity) -> Option<(Vec<Uint>, Uint)> {
    let start = std::time::Instant::now();
    let size = n.bits();
    let n0 = n.digits()[0];
    let iters = match size {
        0..=24 => 128,
        25..=32 => 512,
        33..=40 => 2048,
        41..=48 => 8192,
        // 16384 is a bit too small for 55 bits.
        49..=52 => 16384,
        53..=57 => 32768,
        58..=62 => 65536,
        63..=64 => 131072,
        // Skip multiprecision inputs: factors will already be found
        // by P-1 and ECM. Even if Pollard rho is quick, it is redundant
        // with other fast methods.
        _ => return None,
    };
    // Use several functions to avoid large cycles.
    // We don't want to fallback to a slower algorithm.
    // Typical runtime is below 1ms.
    for c in 1..10 {
        if let Some((p, q)) = rho64(n0, c, iters) {
            if verbosity >= Verbosity::Info {
                let ms = start.elapsed().as_secs_f64() * 1000.0;
                eprintln!("Found factor {p} with Pollard rho (iters={c}x{iters}) in {ms:.3}ms");
            }
            return Some((vec![p.into()], q.into()));
        }
    }
    None
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
    let (fs, nred) = gcd_factors(n, &prods);
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
            if let Some((p, q)) = rho64(n, 1, budget) {
                eprintln!("factored {n} with budget {budget} => {p}*{q}");
                assert_eq!(p * q, n);
                continue 'nextn;
            }
        }
        panic!("failed to factor {n}");
    }

    // p=409891 has an usually long cycle (it requires 54724 iterations).
    let n = 409891 * 69890423;
    for budget in [500, 1000, 1500, 2000, 4000, 7000, 10000, 20000, 50000] {
        assert!(rho64(n, 1, budget).is_none());
    }
    let (p, q) = rho64(n, 1, 54725).unwrap();
    assert_eq!(p * q, n);
    // But another function will work.
    let (p, q) = rho64(n, 2, 1000).unwrap();
    assert_eq!(p * q, n);
    // 64-bit integers must work.
    let n = 0xeb67d1ff62bd9f49;
    let (p, q) = rho64(n, 2, 65536).unwrap();
    assert_eq!(p * q, n);
}

#[test]
fn test_rho_small() {
    // Test some numbers observed to fail with some parameters.
    // Especially if cycles happens simultaneously.
    // Requires checking GCD often enough or testing enough values.
    let ns: &[u64] = &[
        281 * 331,
        2179 * 2539,
        2707 * 3821,
        3119 * 3719,
        12011 * 13619,
        14879 * 16229,
        43037 * 59107,
        139801 * 146381,
        611641 * 995651,
        937571 * 917209,
        // Needs many iterations for iters=16384
        114385069 * 94938061,
        168806699 * 197877437,
        173937383 * 240881257,
        946617377 * 892240367,
        693965191 * 1582979039,
        // Needs many iterations for iters=65536
        4088764103 * 3473680711,
        3142887637 * 2807200547,
    ];
    for &n in ns {
        if let Some((p, q)) = rho(&Uint::from(n), Verbosity::Info) {
            assert_eq!(p[0] * q, Uint::from_digit(n));
        } else {
            panic!("failure for {n}");
        }
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
        for budget in [512, 1024, 2000, 4000, 8000, 16000, 32000, 65000] {
            let mut ok = 0;
            let start = std::time::Instant::now();
            for &n in &samples {
                if let Some((x, y)) = rho64(n, 1, budget) {
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
                if let Some((x, y)) = rho64(n, 1, budget / 2).or_else(|| rho64(n, 2, budget / 2)) {
                    assert_eq!(x * y, n);
                    ok += 1;
                }
            }
            let elapsed = start.elapsed().as_secs_f64() * 1000.;
            eprintln!(
                "{size} bits, budget={budget} (2 polys) factored {ok}/{total} semiprimes in {elapsed:.2}ms"
            );

            ok = 0;
            let start = std::time::Instant::now();
            for &n in &samples {
                if let Some((x, y)) = rho64(n, 1, budget / 3)
                    .or_else(|| rho64(n, 2, budget / 3))
                    .or_else(|| rho64(n, 3, budget / 3))
                {
                    assert_eq!(x * y, n);
                    ok += 1;
                }
            }
            let elapsed = start.elapsed().as_secs_f64() * 1000.;
            eprintln!(
                "{size} bits, budget={budget} (3 polys) factored {ok}/{total} semiprimes in {elapsed:.2}ms"
            );
        }
    }
}

#[test]
#[ignore = "takes one hour"]
fn test_rho_allprimes() {
    // Test Brent cycle finding convergence speed
    // for all primes below 2^29. This takes about 1 hour.

    // A tiny 32-bit implementation of rho algorithm.
    // For a prime modulus, the GCD step is not necessary.
    fn inv32(n: u32) -> u32 {
        let mut x = 1u32;
        loop {
            let rem = n.wrapping_mul(x) - 1;
            if rem == 0 {
                break;
            }
            x += 1 << rem.trailing_zeros();
        }
        assert!(n.wrapping_mul(x) == 1);
        1 + !x
    }
    // Function x^2 + c using Montgomery modular multiplication.
    fn f(p: u32, pinv: u32, x: u32, c: u32) -> u32 {
        let xx = x as u64 * x as u64;
        let xred = if xx as u32 == 0 {
            (xx >> 32) as u32
        } else {
            let m = (xx as u32).wrapping_mul(pinv);
            let mn = m as u64 * p as u64;
            (xx >> 32) as u32 + (mn >> 32) as u32 + 1
        };
        let mut res = xred + c;
        while res >= p {
            res -= p;
        }
        res
    }
    fn rho(p: u32, c: u32) -> u32 {
        // The seed is always 2 but we vary c=1,2,3
        let pinv = inv32(p);
        // p * pinv = 2^32 rinv - 1
        let rinv = ((p as u64 * pinv as u64 + 1) >> 32) as u32;
        // Replicate the 64-bit variant using 32-bit arithmetic (R=2^32)
        // x0=seed, x R^2 => x^2 R^2 + c
        // becomes:
        // y0=seed/R, y R => y^2 R + c/R
        let seed = ((2 * rinv as u64) % p as u64) as u32;
        let c = ((c as u64 * rinv as u64) % p as u64) as u32;
        let (mut x1, mut x2) = (seed, seed);
        let mut next_interval_start = 0;
        let mut next_interval_end = 1;
        let mut e2 = 0;
        loop {
            e2 += 1;
            x2 = f(p, pinv, x2, c);
            if e2 < next_interval_start {
                continue;
            }
            if x2 == x1 {
                return e2;
            }
            if e2 == next_interval_end {
                x1 = x2;
                let pow2k = e2 + 1;
                next_interval_start = pow2k + pow2k / 2;
                next_interval_end = 2 * pow2k - 1;
            }
        }
    }
    // This is also the result of 64-bit rho with seed 2
    // and x => mg_mul(x,x)+1
    assert_eq!(rho(2043251411, 1), 129832);

    use crate::fbase;
    let mut s = fbase::PrimeSieve::new();
    let mut done = 0;
    let start = std::time::Instant::now();
    loop {
        let b = s.next();
        if b.is_empty() || b[0] >= 1 << 29 {
            break;
        }
        let mut avgrho = 0;
        let mut bdone = 0;
        for &p in b.iter() {
            if p < 1000 {
                continue;
            }
            let r = rho(p, 1) as u64;
            bdone += 1;
            avgrho += r;
            // Display remarkable primes: x²+1 and x²+2 and x²+3 have large cycles.
            // In this case they cannot simultaneously be very large (more than sqrt(40p))
            // These primes are exceptionally unlucky (they need about 2^k + 2^(k-1) iterations
            // for each polynomial)
            // p=3575609 rho(c=1)=12377 rho(c=2)=12807 rho(c=3)=12764
            // p=40726549 rho(c=1)=50577 rho(c=2)=49608 rho(c=3)=49417
            // p=43789153 rho(c=1)=49207 rho(c=2)=53740 rho(c=3)=49207
            // p=173743873 rho(c=1)=100320 rho(c=2)=100786 rho(c=3)=100504
            // p=187522513 rho(c=1)=106808 rho(c=2)=103979 rho(c=3)=106669
            if r * r > 30 * p as u64 {
                let r2 = rho(p, 2) as u64;
                let r3 = rho(p, 3) as u64;
                let p = p as u64;
                if r2 * r2 > 30 * p && r3 * r3 > 30 * p {
                    eprintln!("Large cycles p={p} rho(c=1)={r} rho(c=2)={r2} rho(c=3)={r3}");
                    if p < 25_000_000 {
                        assert!(r * r < 45 * p || r2 * r2 < 45 * p || r3 * r3 < 45 * p);
                    } else {
                        assert!(r * r < 60 * p || r2 * r2 < 60 * p || r3 * r3 < 60 * p);
                    }
                }
            }
        }
        done += b.len();
        // Average cycle size should be bounded by theoretical limits.
        let avg = avgrho / bdone;
        let maxp = b[b.len() - 1];
        assert!(avg * avg <= 7 * maxp as u64);
        if done % 100 == 0 {
            let elapsed = start.elapsed().as_secs_f64();
            eprintln!("processed {done} primes until {maxp} in {elapsed:.3}s (avg {avg})");
        }
    }
}
