// Copyright 2022 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Yamaquasi is a library of integer factoring algorithms.
//!
//! Bibliography:
//!
//! Carl Pomerance, A Tale of Two Sieves
//! <https://www.ams.org/notices/199612/pomerance.pdf>
//!
//! J. Gerver, Factoring Large Numbers with a Quadratic Sieve
//! <https://www.jstor.org/stable/2007781>
//!
//! Wikipedia
//! <https://en.wikipedia.org/wiki/Quadratic_sieve>

pub mod arith;
pub mod arith_fft;
pub mod arith_gcd;
pub mod arith_montgomery;
pub mod arith_poly;
pub mod fbase;
pub mod matrix;
pub mod params;
pub mod relations;

// Implementations
pub mod ecm;
pub mod ecm128;
pub mod mpqs;
pub mod pollard_pm1;
pub mod pollard_rho;
pub mod pp1;
pub mod qsieve;
pub mod qsieve64;
pub mod sieve;
pub mod siqs;
pub mod squfof;

// Class group computations
pub mod classgroup;
pub mod relationcls;

// We need to perform modular multiplication modulo the input number.
pub type Int = arith::I1024;
pub type Uint = arith::U1024;

// Top-level functions
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};

use arith::Num;
use arith_montgomery::{MInt, ZmodN};
use bnum::cast::CastFrom;
use num_integer::Integer;

const DEBUG: bool = false;

#[derive(Default)]
pub struct Preferences {
    // General parameters
    pub threads: Option<usize>,
    pub verbosity: Verbosity,
    pub should_abort: Option<Box<dyn Fn() -> bool + Sync>>,
    // Quadratic sieve parameters
    pub fb_size: Option<u32>,
    pub interval_size: Option<u32>,
    pub large_factor: Option<u64>,
    pub use_double: Option<bool>,
    // Only for class group computations
    pub outdir: Option<PathBuf>,

    // Yes, storing state variables in a Preferences object
    // is quite awkward.
    pm1_done: AtomicBool,
}

impl Preferences {
    /// Whether preferences specify at least `v` as verbosity level.
    #[doc(hidden)]
    pub fn verbose(&self, v: Verbosity) -> bool {
        self.verbosity >= v
    }

    pub fn abort(&self) -> bool {
        if let Some(f) = &self.should_abort {
            f()
        } else {
            false
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Algo {
    Auto,
    Rho,
    Squfof,
    Qs64,
    Pm1,
    Ecm,
    Ecm128,
    Qs,
    Mpqs,
    Siqs,
}

impl FromStr for Algo {
    type Err = Box<dyn std::error::Error>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "auto" => Ok(Self::Auto),
            "rho" => Ok(Self::Rho),
            "squfof" => Ok(Self::Squfof),
            "pm1" => Ok(Self::Pm1),
            "ecm" => Ok(Self::Ecm),
            "ecm128" => Ok(Self::Ecm128),
            "qs" => Ok(Self::Qs),
            "qs64" => Ok(Self::Qs64),
            "mpqs" => Ok(Self::Mpqs),
            "siqs" => Ok(Self::Siqs),
            _ => Err(format!("invalid algo {}", s).into()),
        }
    }
}

#[derive(Default, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub enum Verbosity {
    /// Must not print anything. Useful when using as a library.
    Silent,
    /// Prints general information about the factoring strategy and results.
    /// Users should understand the outline of algorithms from the output.
    /// Default option for interactive use.
    #[default]
    Info,
    /// Detailed information about algorithm progress.
    Verbose,
    /// Debugging information.
    Debug,
}

impl FromStr for Verbosity {
    type Err = Box<dyn std::error::Error>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "0" | "silent" => Ok(Self::Silent),
            "1" | "info" => Ok(Self::Info),
            "2" | "verbose" => Ok(Self::Verbose),
            "3" | "debug" => Ok(Self::Debug),
            _ => Err(format!("invalid verbosity level {}", s).into()),
        }
    }
}

/// An "error" describing a (small) factor encountered unexpectedly
/// during algorithms. This includes factor bases containing a divisor
/// of N or ECM generating a singular curve.
#[derive(Debug)]
pub struct UnexpectedFactor(u64);

#[derive(Debug)]
pub struct FactoringFailure;

/// Factorizes an integer into a product of factors.
pub fn factor(n: Uint, alg: Algo, prefs: &Preferences) -> Result<Vec<Uint>, FactoringFailure> {
    if n.is_zero() {
        return Ok(vec![n]);
    }
    let mut factors = vec![];
    if prefs.verbose(Verbosity::Info) {
        eprintln!("Testing small prime divisors");
    }
    let mut nred = n;
    for (&p, div) in fbase::SMALL_PRIMES
        .iter()
        .zip(&fbase::SMALL_PRIMES_DIVIDERS)
    {
        loop {
            let (q, r) = div.divmod_uint(&nred);
            if r != 0 {
                break;
            }
            nred = q;
            factors.push(p.into());
            if prefs.verbose(Verbosity::Info) {
                eprintln!("Found small factor {p}");
            }
        }
    }
    if nred != n && prefs.verbose(Verbosity::Info) {
        eprintln!("Factoring {nred}");
    }
    // Create thread pool
    let tpool: Option<rayon::ThreadPool> = match prefs.threads {
        None | Some(1) => None,
        Some(t) => {
            if prefs.verbose(Verbosity::Verbose) {
                eprintln!("Using a pool of {t} threads");
            }
            Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(t)
                    .build()
                    .expect("cannot create thread pool"),
            )
        }
    };
    let tpool = tpool.as_ref();
    factor_impl(nred, alg, prefs, &mut factors, tpool);
    check_factors(&n, &factors)?;
    factors.sort();
    Ok(factors)
}

/// Partially factorize an integer heuristically finding
/// all factors with bit length less than a specified size,
/// with expected probability at least 99%.
///
/// Algorithm selection is automatic and multithreading is not used.
pub fn factor_smooth(n: Uint, factor_bits: usize) -> Vec<Uint> {
    if n.is_zero() {
        return vec![n];
    }
    let mut factors = vec![];
    let mut nred = n;
    for (&p, div) in fbase::SMALL_PRIMES
        .iter()
        .zip(&fbase::SMALL_PRIMES_DIVIDERS)
    {
        loop {
            let (q, r) = div.divmod_uint(&nred);
            if r != 0 {
                break;
            }
            nred = q;
            factors.push(p.into());
        }
    }
    factor_smooth_impl(nred, factor_bits, &mut factors);
    // No primality testing of factors.
    factors.sort();
    factors
}

fn factor_impl(
    n: Uint,
    alg: Algo,
    prefs: &Preferences,
    factors: &mut Vec<Uint>,
    tpool: Option<&rayon::ThreadPool>,
) {
    // Since quadratic sieve methods work by finding non-trivial random
    // elements of multiplicative order 2 modulo n they will fail to resolve
    // prime power factors of n (because Z/p^k Z is cyclic).
    // So prime powers need to be explicitly tested.
    if n.is_one() {
        return;
    }
    let is_perfect_power = {
        if n.bits() <= 64 {
            // Use native arithmetic is possible.
            arith::perfect_power(n.low_u64()).map(|_pk @ (p, k)| (p.into(), k))
        } else {
            arith::perfect_power(n)
        }
    };
    if let Some((p, k)) = is_perfect_power {
        let mut facs = vec![];
        factor_impl(p, alg, prefs, &mut facs, tpool);
        for _ in 0..k {
            factors.extend_from_slice(&facs[..]);
        }
        return;
    } else if pseudoprime(n) {
        factors.push(n);
        return;
    }
    // Apply automatic strategy.
    let alg_real = match alg {
        Algo::Auto => {
            // For small inputs, Pollard rho is faster than ECM and quadratic sieve.
            if n.bits() < 52 {
                if let Some((a_s, b)) = pollard_rho::rho(&n, prefs.verbosity) {
                    for a in a_s {
                        factor_impl(a, alg, prefs, factors, tpool);
                    }
                    if prefs.verbose(Verbosity::Info) {
                        eprintln!("Recursively factor {b}");
                    }
                    factor_impl(b, alg, prefs, factors, tpool);
                    return;
                }
            }
            // Only in automatic mode, for large inputs, Pollard P-1 and ECM can be useful.
            if n.bits() > 64 && !prefs.pm1_done.load(Ordering::Relaxed) {
                let start_pm1 = std::time::Instant::now();
                let pm1_res = pollard_pm1::pm1_quick(&n, prefs.verbosity);
                // P-1 should be only run once.
                prefs.pm1_done.store(true, Ordering::Relaxed);
                if let Some((a_s, b)) = pm1_res {
                    if prefs.verbose(Verbosity::Info) {
                        eprintln!(
                            "Pollard P-1 success with factors {a_s:?} in {:.3}s",
                            start_pm1.elapsed().as_secs_f64()
                        );
                    }
                    for a in a_s {
                        factor_impl(a, alg, prefs, factors, tpool);
                    }
                    if prefs.verbose(Verbosity::Info) {
                        eprintln!("Recursively factor {b}");
                    }
                    factor_impl(b, alg, prefs, factors, tpool);
                    return;
                } else if prefs.verbose(Verbosity::Info) {
                    eprintln!(
                        "Pollard P-1 failure in {:.3}s",
                        start_pm1.elapsed().as_secs_f64()
                    );
                }
            }
            // ECM, in its 128-bit variant, is already efficient for 52-bit numbers.
            let ecm_res = if matches!(n.bits(), 52..=128) {
                ecm128::ecm128(n, false, prefs)
            } else {
                ecm::ecm_auto(n, prefs, tpool)
            };
            if let Some((a, b)) = ecm_res {
                factor_impl(a, alg, prefs, factors, tpool);
                if prefs.verbose(Verbosity::Info) {
                    eprintln!("Recursively factor {b}");
                }
                factor_impl(b, alg, prefs, factors, tpool);
                return;
            }
            // Select fallback algorithm
            // The above Rho and ECM128 steps should not fail.
            // Select ECM128 as the fallback for small integers.
            if n.bits() <= 80 {
                Algo::Ecm128
            } else {
                Algo::Siqs
            }
        }
        _ => alg,
    };
    // Now the strategy cannot be Auto.
    match alg_real {
        Algo::Auto => unreachable!("impossible"),
        Algo::Pm1 => {
            // Pure Pollard P-1
            let start_pm1 = std::time::Instant::now();
            if let Some((a_s, b)) = pollard_pm1::pm1_only(&n, prefs.verbosity) {
                if prefs.verbose(Verbosity::Info) {
                    eprintln!(
                        "Pollard P-1 success with factors p={a_s:?} in {:.3}s",
                        start_pm1.elapsed().as_secs_f64()
                    );
                }
                for a in a_s {
                    factor_impl(a, alg, prefs, factors, tpool);
                }
                if prefs.verbose(Verbosity::Info) {
                    eprintln!("Recursively factor {b}");
                }
                factor_impl(b, alg, prefs, factors, tpool);
                return;
            } else if prefs.verbose(Verbosity::Info) {
                eprintln!(
                    "Pollard P-1 failure in {:.3}s",
                    start_pm1.elapsed().as_secs_f64()
                );
            }
            factors.push(n);
            return;
        }
        Algo::Ecm => {
            if let Some((a, b)) = ecm::ecm_only(n, prefs, tpool) {
                factor_impl(a, alg, prefs, factors, tpool);
                if prefs.verbose(Verbosity::Info) {
                    eprintln!("Recursively factor {b}");
                }
                factor_impl(b, alg, prefs, factors, tpool);
                return;
            }
            if prefs.verbose(Verbosity::Info) {
                eprintln!("Factorization is incomplete.");
            }
            factors.push(n);
            return;
        }
        Algo::Ecm128 => {
            // "Small" ECM
            // However due to determinism the recursion will go through the
            // same curves, which is not very useful.
            if let Some((a, b)) = ecm128::ecm128(n, true, prefs) {
                factor_impl(a, alg, prefs, factors, tpool);
                if prefs.verbose(Verbosity::Info) {
                    eprintln!("Recursively factor {b}");
                }
                factor_impl(b, alg, prefs, factors, tpool);
                return;
            }
            if prefs.verbose(Verbosity::Info) {
                eprintln!("Factorization is incomplete.");
            }
            factors.push(n);
            return;
        }
        Algo::Qs64 => {
            assert!(n.bits() <= 64);
            if let Some((a, b)) = qsieve64::qsieve(n.low_u64(), prefs.verbosity) {
                // Recurse
                factor_impl(a.into(), alg, prefs, factors, tpool);
                factor_impl(b.into(), alg, prefs, factors, tpool);
            } else {
                if prefs.verbose(Verbosity::Info) {
                    eprintln!("qsieve64 failed");
                }
                factors.push(n);
            }
            return;
        }
        Algo::Rho => {
            assert!(n.bits() <= 64);
            if let Some((a_s, b)) = pollard_rho::rho(&n, prefs.verbosity) {
                for a in a_s {
                    factor_impl(a, alg, prefs, factors, tpool);
                }
                if prefs.verbose(Verbosity::Info) {
                    eprintln!("Recursively factor {b}");
                }
                factor_impl(b, alg, prefs, factors, tpool);
                return;
            } else {
                if prefs.verbose(Verbosity::Info) {
                    eprintln!("Rho algorithm failed");
                }
            }
        }
        Algo::Squfof => {
            assert!(n.bits() <= 64);
            if let Some((a, b)) = squfof::squfof(n.low_u64()) {
                factor_impl(a.into(), alg, prefs, factors, tpool);
                factor_impl(b.into(), alg, prefs, factors, tpool);
            } else {
                if prefs.verbose(Verbosity::Info) {
                    eprintln!("SQUFOF failed");
                }
                factors.push(n);
            }
            return;
        }
        // Otherwise it is a quadratic sieve.
        Algo::Qs | Algo::Mpqs | Algo::Siqs => {}
    }
    if prefs.abort() {
        factors.push(n);
        return;
    }

    let (k, score) = fbase::select_multiplier(n);
    if prefs.verbose(Verbosity::Info) {
        eprintln!("Selected multiplier {k} (score {score:.2}/10)");
    }
    let divs = match alg_real {
        Algo::Qs => Ok(qsieve::qsieve(n, k, prefs, tpool)),
        Algo::Mpqs => Ok(mpqs::mpqs(n, k, prefs, tpool)),
        Algo::Siqs => siqs::siqs(&n, k, prefs, tpool),
        _ => unreachable!("impossible"),
    };
    let divs = match divs {
        Ok(divs) => {
            if divs.len() == 0 {
                // Failure or interrupted.
                factors.push(n);
                return;
            } else {
                divs
            }
        }
        Err(UnexpectedFactor(d)) => {
            factor_impl(d.into(), alg, prefs, factors, tpool);
            factor_impl(n / Uint::from(d), alg, prefs, factors, tpool);
            return;
        }
    };
    // Use non trivial divisors to factor n.
    // A factorization of n.
    let mut facs = vec![n];
    for d in divs {
        // is it combined with existing divisors?
        let mut residue = d;
        let mut splits = vec![];
        facs.retain(|&f| {
            let gcd: Uint = Integer::gcd(&f, &residue);
            let split = gcd != f && !gcd.is_one();
            if split {
                splits.push(f)
            }
            residue /= gcd;
            !split
        });
        assert!(residue.is_one());
        let mut residue = d;
        for f in splits {
            let gcd: Uint = Integer::gcd(&f, &residue);
            if gcd != f && !gcd.is_one() {
                facs.push(f / gcd);
                facs.push(gcd);
            } else {
                // Can this happen?
                facs.push(f);
            }
            residue /= gcd;
        }
    }
    for f in facs {
        if f == n {
            if prefs.verbose(Verbosity::Info) {
                eprintln!("Factorization failure");
            }
            factors.push(f);
        } else if !pseudoprime(f) {
            if prefs.verbose(Verbosity::Info) {
                eprintln!("Recursively factor {f}");
            }
            factor_impl(f, alg, prefs, factors, tpool);
        } else {
            factors.push(f);
        }
    }
}

fn factor_smooth_impl(n: Uint, factor_bits: usize, factors: &mut Vec<Uint>) {
    if n.is_one() {
        return;
    }
    let is_perfect_power = {
        if n.bits() <= 64 {
            // Use native arithmetic is possible.
            arith::perfect_power(n.low_u64()).map(|_pk @ (p, k)| (p.into(), k))
        } else {
            arith::perfect_power(n)
        }
    };
    if let Some((p, k)) = is_perfect_power {
        let mut facs = vec![];
        factor_smooth_impl(p, factor_bits, &mut facs);
        for _ in 0..k {
            factors.extend_from_slice(&facs[..]);
        }
        return;
    } else if pseudoprime(n) {
        factors.push(n);
        return;
    }
    let mut prefs = Preferences::default();
    prefs.verbosity = Verbosity::Silent;
    // Always try Pollard rho to clear very small factors
    // 1024 iterations will catch 99% of 16-bit factors.
    let mut nred: Uint = n;
    let rho_iters = 1024;
    if n.bits() <= 63 {
        while let Some((a, b)) = pollard_rho::rho64(nred.low_u64(), 2, rho_iters) {
            factor_impl(a.into(), Algo::Auto, &prefs, factors, None);
            nred = b.into();
        }
    } else {
        if let Some((a_s, b)) = pollard_rho::rho_impl(&nred, 2, rho_iters, prefs.verbosity) {
            // Force rho for recursion
            for a in a_s {
                factor_impl(a, Algo::Auto, &prefs, factors, None);
            }
            nred = b;
        }
    }
    if factor_bits <= 16 || pseudoprime(nred) {
        factors.push(nred);
        return;
    }
    // Settings finding ~99% of 32-bit factors (see ecm128::ecm128)
    let (curves, b1, b2) = match factor_bits {
        0..=19 => (6, 40, 1080.),
        20..=21 => (8, 50, 1920.),
        22..=23 => (10, 70, 1920.),
        24..=25 => (16, 100, 3000.),
        26..=27 => (20, 150, 3000.),
        _ => (20, 180, 7.7e3), // 28..32
    };
    if nred.bits() < 128 {
        while let Some((a, b)) = ecm128::ecm(u128::cast_from(nred), curves, b1, b2, prefs.verbosity)
        {
            factor_impl(a.into(), Algo::Auto, &prefs, factors, None);
            nred = b.into();
            if pseudoprime(nred) {
                factors.push(nred);
                return;
            }
        }
    } else {
        // Try ECM
        while let Some((a, b)) = ecm::ecm(nred, curves, b1 as usize, b2, &prefs, None) {
            factor_impl(a.into(), Algo::Auto, &prefs, factors, None);
            nred = b.into();
            if pseudoprime(nred) {
                factors.push(nred);
                return;
            }
        }
    }
    if factor_bits < 32 {
        factors.push(nred);
        return;
    }
    if factor_bits as f64 / nred.bits() as f64 > 0.4 {
        // This is equivalent to full factoring
        factor_impl(nred, Algo::Auto, &prefs, factors, None);
        return;
    }
    // Otherwise keep using ECM
    let (curves, b1, b2) = match factor_bits {
        0..=35 => (24, 350, 13.2e3),
        36..=39 => (60, 600, 20e3),
        40..=43 => (30, 1500, 81e3),
        44..=47 => (50, 2500, 126e3),
        48..=51 => (70, 4000, 126e3),
        52..=55 => (80, 6000, 181e3),
        56..=59 => (100, 10000, 323e3),
        // Parameters from ecm::ecm_only
        60..=63 => (200, 15_000, 554e3),
        64..=79 => (100, 100_000, 19e6),
        80..=95 => (200, 300_000, 156e6),
        96..=119 => (600, 3_000_000, 10e9),
        120..=143 => (2000, 15_000_000, 136e9),
        144..=167 => (10000, 60_000_000, 1500e9),
        _ => (15000, 350_000_000, 49e12),
    };
    while let Some((a, b)) = ecm::ecm(nred, curves, b1 as usize, b2, &prefs, None) {
        factor_impl(a.into(), Algo::Rho, &prefs, factors, None);
        nred = b.into();
        if pseudoprime(nred) {
            factors.push(nred);
            return;
        }
    }
    factors.push(nred);
}

fn check_factors(n: &Uint, factors: &[Uint]) -> Result<(), FactoringFailure> {
    if let &[p] = &factors {
        assert_eq!(n, p);
        if !pseudoprime(*p) {
            return Err(FactoringFailure);
        }
    }
    assert_eq!(*n, factors.iter().product::<Uint>());
    Ok(())
}

/// Primality test of u64 using a Miller test for small bases.
///
/// It is known that bases until 11 are enough for a 40-bit integer.
/// It is known that testing bases until 37 is enough for a 64-bit integer.
pub fn isprime64(p: u64) -> bool {
    if p < *fbase::SMALL_PRIMES.last().unwrap() {
        return fbase::SMALL_PRIMES[..].contains(&p);
    }
    // Compute auxiliary numbers for modular arithmetic.
    let pinv = arith_montgomery::mg_2adic_inv(p);
    let r1 = 0_u64.wrapping_sub(p) % p; // 2^64 % p == (2^64-p) % p
    let r2 = ((r1 as u128 * r1 as u128) % (p as u128)) as u64;

    let tz = (p - 1).trailing_zeros();
    let podd = p >> tz;

    let one = r1;
    let pm1 = p - r1;
    let mul = |x, y| arith_montgomery::mg_mul(p, pinv, x, y);
    // Performs the Miller test for base b.
    let miller = |b: u64| {
        let b = mul(b, r2);
        // Compute b^podd
        let mut pow = {
            let mut x = one;
            let mut sq = b;
            let mut exp = podd;
            while exp > 0 {
                if exp & 1 == 1 {
                    x = mul(x, sq);
                }
                sq = mul(sq, sq);
                exp /= 2;
            }
            x
        };
        let mut ok = pow == one || pow == pm1;
        for _ in 0..tz {
            pow = mul(pow, pow);
            if pow == pm1 {
                ok = true;
                break;
            } else if pow == one {
                break;
            }
        }
        ok
    };
    // Bases for 20-bit integers.
    for b in [2, 3] {
        if !miller(b) {
            return false;
        }
    }
    if p >> 20 != 0 {
        // Bases for 40-bit integers.
        for b in [5, 7, 11] {
            if !miller(b) {
                return false;
            }
        }
    }
    if p >> 40 != 0 {
        // Bases for 64-bit integers.
        for b in [13, 17, 19, 23, 29, 31, 37] {
            if !miller(b) {
                return false;
            }
        }
    }
    true
}

/// Probabilistic primality test using a Miller test for small bases.
pub fn pseudoprime(p: Uint) -> bool {
    // Montgomery arithmetic is only for odd numbers.
    if !p.bit(0) {
        return p.try_into() == Ok(2_u64);
    }
    if p.bits() <= 64 {
        return isprime64(p.low_u64());
    }
    pub fn pow_mod(zp: &ZmodN, x: MInt, exp: &Uint) -> MInt {
        let mut res = zp.one();
        let mut x = x;
        for b in 0..exp.bits() {
            if exp.bit(b) {
                res = zp.mul(&res, &x);
            }
            x = zp.mul(&x, &x);
        }
        res
    }

    let zp = ZmodN::new(p);
    let s = (p.low_u64() - 1).trailing_zeros();
    let p_odd = p >> s;
    for &b in &fbase::SMALL_PRIMES {
        let mut pow = pow_mod(&zp, zp.from_int(b.into()), &p_odd);
        let pm1 = zp.sub(&zp.zero(), &zp.one());
        let mut ok = pow == zp.one() || pow == pm1;
        for _ in 0..s {
            pow = zp.mul(&pow, &pow);
            if pow == pm1 {
                ok = true;
                break;
            } else if pow == zp.one() {
                break;
            }
        }
        if !ok {
            return false;
        }
    }
    true
}

#[test]
fn test_factor() -> Result<(), bnum::errors::ParseIntError> {
    let fs = factor(Uint::ZERO, Algo::Auto, &Preferences::default());
    assert_eq!(fs.unwrap(), vec![Uint::ZERO]);

    let fs = factor(Uint::ONE, Algo::Auto, &Preferences::default()).unwrap();
    assert_eq!(fs, vec![]);

    // semiprime
    eprintln!("=> test semiprime");
    let n = Uint::from_str("404385851501206046375042621")?;
    factor(n, Algo::Auto, &Preferences::default()).unwrap();

    // small factor (2003 * 665199750163226410868760173)
    eprintln!("=> test small factor 1");
    let n = Uint::from_str("1332395099576942500970126626519")?;
    factor(n, Algo::Auto, &Preferences::default()).unwrap();

    // 2 small factors (443 * 1151 * 172633679917074861804179395686166722361211)
    let n = Uint::from_str("88024704953957052509918444604606608564924960423")?;
    factor(n, Algo::Auto, &Preferences::default()).unwrap();

    // 148-bit number, no small factors
    // Requires P-1 with B1=1000 B2=2100 or ECM with enough curves
    // SIQS factor base is very sparse: [2, 3, 7, 13, 37, 71, 73, 103, 107, 109]
    let n = Uint::from_str("223986131066668467510118179315296601602386513")?;
    factor(n, Algo::Auto, &Preferences::default()).unwrap();

    // Could trigger an infinite loop after failing to factor due to
    // 223 being in the factor base.
    // 223*28579484042221159639852413780078523
    eprintln!("=> small factor 3");
    let n = Uint::from_str("317546892790192732050746209")?;
    factor(n, Algo::Siqs, &Preferences::default()).unwrap();

    // All small factors until MAX_MULTIPLIER must be properly tested
    // to avoid QS failures.
    // 199 * 18011383943879611828742161
    eprintln!("=> small factor 199");
    let n = Uint::from_str("3584265404832042753919690039")?;
    factor(n, Algo::Siqs, &Preferences::default()).unwrap();

    // When n has a small factor, it can appear in relations, creating pairs
    // (x,y) such that x=±y but the factorization (x-y)(x+y) is still interesting.
    eprintln!("=> small factor 5047");
    let n = Uint::from_str("9416412050459436444341141867167")?;
    factor(n, Algo::Siqs, &Preferences::default()).unwrap();

    // perfect square (17819845476047^2)
    eprintln!("=> test square");
    let n = Uint::from_str("317546892790192732050746209")?;
    factor(n, Algo::Auto, &Preferences::default()).unwrap();
    // square of a composite number
    // (211*499)^2 * 10271
    let n = Uint::from_str("113861979834191")?;
    factor(n, Algo::Auto, &Preferences::default()).unwrap();
    // perfect cube
    eprintln!("=> test cube");
    let n = Uint::from_str("350521251909490182639506149")?;
    factor(n, Algo::Auto, &Preferences::default()).unwrap();
    eprintln!("=> test 6th power");
    let n = Uint::from_str("1000018000135000540001215001458000729")?;
    assert_eq!(
        factor(n, Algo::Auto, &Preferences::default())
            .unwrap()
            .len(),
        6
    );

    // not squarefree (839322217^2 * 705079549)
    eprintln!("=> test not squarefree");
    let n = Uint::from_str("496701596915056959994534861")?;
    factor(n, Algo::Auto, &Preferences::default()).unwrap();

    // Observed failure: n=981572983530105943
    // is 60-bit and unlucky for SQUFOF.
    eprintln!("=> unlucky SQUFOF");
    let n = Uint::from_digit(981572983530105943);
    factor(n, Algo::Auto, &Preferences::default()).unwrap();

    // Basic classical QS sanity check.
    eprintln!("=> simple classical QS");
    let n = Uint::from_str("144145963608905891153").unwrap();
    let fs = factor(n, Algo::Qs, &Preferences::default()).unwrap();
    assert_eq!(fs[0] * fs[1], n);

    // MPQS sanity check
    // n = 1 mod 4 for optimal multiplier
    let n = Uint::from_digit(2028822982549217551);
    factor(n, Algo::Mpqs, &Preferences::default()).unwrap();
    // n = 3 mod 4 for optimal multiplier
    let n = Uint::from_digit(966218335873381319);
    factor(n, Algo::Mpqs, &Preferences::default()).unwrap();

    Ok(())
}

#[test]
fn test_factor_qs_edgecases() -> Result<(), bnum::errors::ParseIntError> {
    // Multiplier 1, factor base 2, 3, 7, 11, 13, 23, 43, 71, 97, 139...
    // has only 16 out of the 48 smallest primes, so the smooth bound
    // must be high enough to collect more primes.
    let n = Uint::from_digit(325434172177);
    let fs = factor(n, Algo::Qs, &Preferences::default()).unwrap();
    assert_eq!(fs[0] * fs[1], n);

    Ok(())
}

#[test]
fn test_factor_mpqs_edgecases() -> Result<(), bnum::errors::ParseIntError> {
    // Very small integers with 2 factors.
    let smalls: &[u64] = &[654949849, 2468912671, 20152052489];
    for &n in smalls {
        let n = Uint::from_digit(n);
        let fs = factor(n, Algo::Mpqs, &Preferences::default()).unwrap();
        assert_eq!(fs[0] * fs[1], n);
    }

    // Multiplier=1, n mod 4 = 3
    let n = Uint::from_str("188568530916066130831")?;
    let fs = factor(n, Algo::Mpqs, &Preferences::default()).unwrap();
    assert_eq!(fs[0] * fs[1], n);

    Ok(())
}

#[test]
fn test_factor_siqs_edgecases() -> Result<(), bnum::errors::ParseIntError> {
    // SIQS with a small number: A needs 3 factors.
    // Used to fail due to selecting 2 factors or too many As.
    eprintln!("=> SIQS 60-75 bits");
    let n = Uint::from_digit(1231055495188530589);
    factor(n, Algo::Siqs, &Preferences::default()).unwrap();
    let n = Uint::from_digit(1939847356913363213);
    factor(n, Algo::Siqs, &Preferences::default()).unwrap();
    let n = Uint::from_digit(9173516735614600627);
    factor(n, Algo::Siqs, &Preferences::default()).unwrap();
    let n = Uint::from_str("10847815350861015899809")?;
    factor(n, Algo::Siqs, &Preferences::default()).unwrap();

    // SIQS does not generate many relations (not #fbase + 64) but still enough.
    let n = Uint::from_digit(4954670127929);
    factor(n, Algo::Siqs, &Preferences::default()).unwrap();
    // SIQS generates enough relations but they are fewer than the factor base (ok).
    let n = Uint::from_str("495751324548272090616278443938858471242622233")?;
    factor(n, Algo::Siqs, &Preferences::default()).unwrap();

    // Numbers with small/sparse factor bases: they make it difficult to generate
    // optimal A values. Most examples are failures during random testing.
    // To succeed they need a large enough factor base or interval.

    // Multiplier 1, factor base [2, 3, 7, 11, 19, 31, 59, 67]
    let n = Uint::from_digit(314534861617);
    factor(n, Algo::Siqs, &Preferences::default()).unwrap();
    // Another small number.
    let n = Uint::from_digit(157261665529);
    factor(n, Algo::Siqs, &Preferences::default()).unwrap();
    // Factor base [2, 3, 7, 13, 19, 79, 89, 107, 131, ...]
    let n = Uint::from_str("72231484786704818233")?;
    factor(n, Algo::Siqs, &Preferences::default()).unwrap();
    // 72 bit example, multiplier 1: factor base gap 47, 61, 103, 131, 137, 163, 223
    let n = Uint::from_str("212433504133480536121")?;
    factor(n, Algo::Siqs, &Preferences::default()).unwrap();
    // Factor base gap between 47 and 97:
    let n = Uint::from_str("232159658536337208497609")?;
    factor(n, Algo::Siqs, &Preferences::default()).unwrap();
    // 87-bit number, multiplier 43, factor base gap 79, 173, 223
    // Can create very suboptimal values of A.
    let n = Uint::from_str("145632526168873091762826187")?;
    factor(n, Algo::Siqs, &Preferences::default()).unwrap();
    // Large gap between 127 and 211: selection of A was stuck
    // because only 1 candidate is found.
    let n = Uint::from_str("774227958313673793204983642345821")?;
    factor(n, Algo::Siqs, &Preferences::default()).unwrap();

    // SIQS with 90-100 bit numbers: A needs 4 factors (5 is too many)
    let n = Uint::from_str("13819541643362998561057402169")?;
    factor(n, Algo::Siqs, &Preferences::default()).unwrap();
    let n = Uint::from_str("34084481733943226418420736441")?;
    factor(n, Algo::Siqs, &Preferences::default()).unwrap();
    Ok(())
}

#[test]
fn test_factor_ecm_edgecases() -> Result<(), bnum::errors::ParseIntError> {
    // ECM with very small factors.
    // A number with 2 very close small factors can be difficult to fully factor with ECM.
    // This could cause an infinite loop or a crash.

    let n = Uint::from_str("149765065983515097066869381115702138825777596")?;
    let fs = factor(n, Algo::Ecm, &Preferences::default()).unwrap();
    assert_eq!(fs.len(), 18);

    // Products of small primes
    #[rustfmt::skip]
    let smalls = [
        211 * 251, 211 * 263, 229 * 283, 653 * 821,
        769 * 797, 1399 * 1559, 1433 * 1613,
    ];
    for n in smalls {
        let n = Uint::from_digit(n);
        let fs = factor(n, Algo::Ecm, &Preferences::default()).unwrap();
        assert_eq!(fs.len(), 2);
        assert!(fs[0] * fs[1] == n);
    }
    let smalls = [1621 * 1709 * 1733, 1697 * 1787 * 1831];
    for n in smalls {
        let n = Uint::from_digit(n);
        let fs = factor(n, Algo::Ecm, &Preferences::default()).unwrap();
        assert_eq!(fs.len(), 3);
        assert!(fs[0] * fs[1] * fs[2] == n);
    }

    // Exceptional primes for the Suyama-11 family
    // The generator (12,24) of the modular curve has order 17, 19, 23
    // for the prime factors of this number:
    // 4076109244408937005985066831
    // 6705163953466859722567402034832167
    // 6164159587123652872394951179302664763814309093309197470628088016151
    // As a consequence ECM must find all factors quickly "accidentally".
    let n = Uint::from_str("168472527175896339170265431590477670742294002583447818106088193178591923790722222786096568695941644944559115227634107731116901327")?;
    let fs = factor(n, Algo::Ecm, &Preferences::default()).unwrap();
    assert_eq!(fs.len(), 3);
    assert_eq!(fs[0] * fs[1] * fs[2], n);

    // Similarly for the small variant.
    let n = Uint::from_str("35095808598940323061")?;
    let fs = factor(n, Algo::Ecm128, &Preferences::default()).unwrap();
    assert_eq!(fs.len(), 2);
    assert_eq!(fs[0] * fs[1], n);

    Ok(())
}

#[test]
fn test_factor_smooth() -> Result<(), bnum::errors::ParseIntError> {
    let n = Uint::from_str(
        "12201879382649676470194887538548324244326362954552609354733068712098984100",
    )?;
    let fs = factor_smooth(n, 55);
    assert_eq!(fs.len(), 13);

    let n = Uint::from_str(
        "24091791282868184848805353591599648198248879500538357141333569815331594788",
    )?;
    let fs = factor_smooth(n, 55);
    assert_eq!(fs.len(), 14);

    let n = Uint::from_str("2491972896696220875515999140704747439734476676")?;
    let fs = factor_smooth(n, 20);
    // n = 2*2*3*23*229*443*14341*34883*98221*104123*17396053561176129467
    assert_eq!(fs.len(), 11);
    Ok(())
}

#[test]
fn test_factor_smooth_random() {
    // Test that factor_smooth catches almost all factor
    // of advertised size.
    // In debug mode, it is very slow, 1 sample per size is good enough.
    const SAMPLES: usize = 1;
    //const SAMPLES: usize = 20;
    //const SAMPLES: usize = 1000;
    let mut seed = 1234567_u128;
    for pbits in [12, 16, 20, 24, 32, 40, 48, 56, 63] {
        for qbits in [std::cmp::min(pbits + 8, 63), 64, 127] {
            let mut ps = vec![];
            let mut qs = vec![];
            while ps.len() < SAMPLES || qs.len() < SAMPLES {
                seed = seed.wrapping_mul(123456789_123456789);
                let p = seed % (1 << pbits);
                if isprime64(p as u64) {
                    ps.push(p as u64);
                }
                seed = seed.wrapping_mul(123456789_123456789);
                let q = seed % (1 << qbits);
                if pseudoprime(q.into()) {
                    qs.push(q);
                }
            }
            let mut ok = 0;
            let start = std::time::Instant::now();
            for i in 0..SAMPLES {
                let n = Uint::from(ps[i]) * Uint::from(qs[i]);
                let facs = factor_smooth(n, pbits);
                let mut expect = [ps[i] as u128, qs[i]].map(Uint::from);
                expect.sort();
                if facs.len() == 2 {
                    assert!(facs == expect, "facs={facs:?} p={} q={}", ps[i], qs[i]);
                    ok += 1;
                }
            }
            let elapsed = start.elapsed().as_secs_f64() * 1000.;
            eprintln!("{pbits}+{qbits} bits, factored {ok}/{SAMPLES} semiprimes in {elapsed:.2}ms");
            assert!(ok >= SAMPLES - SAMPLES / 6);

            // Check that adding tiny factors does not affect the success rate.
            let mut ok = 0;
            let start = std::time::Instant::now();
            for i in 0..SAMPLES {
                let n = Uint::from(ps[i]) * Uint::from(qs[i]) * Uint::from(79608931_u64);
                let facs = factor_smooth(n, pbits);
                let mut expect = [67, 733, 1621, ps[i] as u128, qs[i]].map(Uint::from);
                expect.sort();
                if facs.len() == 5 {
                    assert!(facs == expect, "facs={facs:?} p={} q={}", ps[i], qs[i]);
                    ok += 1;
                }
            }
            let elapsed = start.elapsed().as_secs_f64() * 1000.;
            eprintln!(
                "small+{pbits}+{qbits} bits, factored {ok}/{SAMPLES} semiprimes in {elapsed:.2}ms"
            );
            assert!(ok >= SAMPLES - SAMPLES / 6);
        }
    }
}

#[test]
fn test_pseudoprime() {
    assert!(!pseudoprime(1_u64.into()));
    assert!(pseudoprime(2_u64.into()));
    assert!(!pseudoprime(4_u64.into()));
    assert!(pseudoprime(17_u64.into()));
    // Large prime
    assert!(pseudoprime(Uint::from_str("1515019151549030796823931666316891543876480618160148234227332522965297454091879022905608234715852754566536639937").unwrap()));
    // Some composite number.
    assert!(!pseudoprime(
        Uint::from_str("893439027234689082677874957196339924735254319540").unwrap()
    ));
    // Carmichael number.
    assert!(!pseudoprime(9746347772161_u64.into()));

    assert!(!isprime64(1));
    assert!(isprime64(2));
    assert!(isprime64(17));
    assert!(!isprime64(9746347772161));
    // Some prime number
    assert!(isprime64(9938261980284378737));
    // Some composite number
    assert!(!isprime64(11775166524998067797));
}
