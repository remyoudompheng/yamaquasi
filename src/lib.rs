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
pub mod primality;
pub mod relations;

// Implementations
pub mod ecm;
pub mod mpqs;
pub mod pollard_pm1;
pub mod pollard_rho;
pub mod pp1;
pub mod qsieve;
pub mod qsieve64;
pub mod sieve;
pub mod siqs;
pub mod squfof;
pub mod trialdiv;

// Class group computations
pub mod classgroup;
pub mod relationcls;

pub use primality::{isprime64, pseudoprime};

// We need to perform modular multiplication modulo the input number.
pub type Int = arith::I1024;
pub type Uint = arith::U1024;

// Top-level functions
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::atomic::AtomicU32;

use arith::Num;
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

    // Hidden state variable
    pub(crate) ecm_seed: AtomicU32,
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

    pub fn make_thread_pool(&self) -> Option<rayon::ThreadPool> {
        match self.threads {
            None | Some(1) => None,
            Some(t) => {
                if self.verbose(Verbosity::Verbose) {
                    eprintln!("Using a pool of {t} threads");
                }
                Some(
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(t)
                        .build()
                        .expect("cannot create thread pool"),
                )
            }
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
    if alg == Algo::Auto {
        return factor_auto(n, prefs);
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
    let tpool = prefs.make_thread_pool();
    factor_impl(nred, alg, prefs, &mut factors, tpool.as_ref());
    check_factors(&n, &factors)?;
    factors.sort();
    Ok(factors)
}

/// Partially factorize an integer heuristically finding
/// all factors with bit length less than a specified size,
/// with expected probability at least 99%.
///
/// Algorithm selection is automatic and multithreading is not used.
/// Parameters are optimized for the case where many small factors
/// are expected.
pub fn find_small_factors(n: Uint, factor_bits: usize) -> Vec<Uint> {
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
    factor_small_impl(nred, factor_bits, &mut factors);
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
    assert!(alg != Algo::Auto);
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
        } else if n.bits() <= 128 {
            arith::perfect_power(u128::cast_from(n)).map(|_pk @ (p, k)| (p.into(), k))
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
    // Now the strategy cannot be Auto.
    match alg {
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
            if let Some(a_s) = ecm::ecm_only(n, prefs, tpool) {
                let mut nred = n;
                for a in a_s {
                    if (nred % a).is_zero() {
                        nred /= a;
                        if prefs.verbose(Verbosity::Info) {
                            eprintln!("Recursively factor {a}");
                        }
                        factor_impl(a, alg, prefs, factors, tpool);
                    }
                }
                if nred.bits() > 1 {
                    if prefs.verbose(Verbosity::Info) {
                        eprintln!("Recursively factor {nred}");
                    }
                    factor_impl(nred, alg, prefs, factors, tpool);
                }
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
            if let Some((a, b)) = ecm::ecm_small_auto(n, true, prefs) {
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
    let divs = match alg {
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

fn factor_auto(n: Uint, prefs: &Preferences) -> Result<Vec<Uint>, FactoringFailure> {
    // Eliminate special cases.
    let mut factors = vec![];
    if n.is_one() {
        return Ok(factors);
    }

    if prefs.verbose(Verbosity::Info) {
        eprintln!("Testing small prime divisors");
    }
    if n.bits() <= 32 {
        let mut n32 = u32::cast_from(n);
        let tz = n32.trailing_zeros();
        for _ in 0..tz {
            factors.push(Uint::from_digit(2));
        }
        n32 >>= tz;
        trialdiv::factor_u32(n32, &mut factors);
        return Ok(factors);
    }

    let mut n = n;
    let mut nred = n;

    let tz = nred.trailing_zeros();
    for _ in 0..tz {
        factors.push(Uint::from_digit(2));
    }
    nred >>= tz;

    if nred.bits() <= 128 {
        trialdiv::factor4096(&mut nred, &mut factors);
    } else {
        trialdiv::factor128k(&mut nred, &mut factors);
    }
    if nred != n && prefs.verbose(Verbosity::Info) {
        eprintln!("Factoring {nred}");
    }
    if nred.is_one() {
        return Ok(factors);
    }

    n = nred;
    let is_perfect_power = {
        if n.bits() <= 64 {
            // Use native arithmetic is possible.
            arith::perfect_power(n.low_u64()).map(|_pk @ (p, k)| (p.into(), k))
        } else if n.bits() <= 128 {
            arith::perfect_power(u128::cast_from(n)).map(|_pk @ (p, k)| (p.into(), k))
        } else {
            arith::perfect_power(n)
        }
    };
    if let Some((p, k)) = is_perfect_power {
        let facs = factor_auto(p, prefs)?;
        for _ in 0..k {
            factors.extend_from_slice(&facs[..]);
        }
        return Ok(factors);
    }

    let tpool = prefs.make_thread_pool();
    factor_auto_impl(n, prefs, &mut factors, tpool.as_ref())?;
    Ok(factors)
}

/// Current state of factorization.
struct FactorState {
    // The original number to be factored.
    //n: Uint,
    // The current composite factor to process.
    nred: Uint,

    // A list of prime factors of n.
    primes: Vec<Uint>,
    // A list of composite factors of n.
    composites: Vec<Uint>,
    // Invariants
    // n = product(primes) * product(composites)
    // nred = product(composites)
}

impl FactorState {
    fn new(n: Uint) -> Self {
        FactorState {
            nred: n,
            primes: vec![],
            composites: vec![n],
        }
    }

    /// Update state using a decomposition nred = product(factors) * cofactor.
    fn declare_factors(&mut self, factors: Vec<Uint>) {
        for f in factors {
            if pseudoprime(f) {
                for c in self.composites.iter_mut() {
                    while *c % f == Uint::ZERO {
                        self.primes.push(f);
                        *c /= f;
                        self.nred /= f;
                    }
                }
            } else {
                let mut i = 0;
                while i < self.composites.len() {
                    let c = self.composites[i];
                    let d = arith_gcd::big_gcd(&c, &f);
                    if d != Uint::ONE && d != c {
                        // Found a divisor of c
                        // Product of composites is unchanged.
                        self.composites[i] = c / d;
                        self.composites.push(d);
                    }
                    i += 1;
                }
            }
        }
        // Move primes to other vector.
        for c in self.composites.iter_mut() {
            if pseudoprime(*c) {
                self.primes.push(*c);
                self.nred /= *c;
                *c = Uint::ONE;
            }
        }
        self.composites.retain(|c| !c.is_one());
    }

    fn done(&self) -> bool {
        self.composites.is_empty()
    }
}

/// Implementation of automatic algorithm selection.
fn factor_auto_impl(
    n: Uint,
    prefs: &Preferences,
    factors: &mut Vec<Uint>,
    tpool: Option<&rayon::ThreadPool>,
) -> Result<(), FactoringFailure> {
    if pseudoprime(n) {
        factors.push(n);
        return Ok(());
    }
    if n.bits() <= 32 {
        trialdiv::factor_u32(u32::cast_from(n), factors);
        return Ok(());
    }
    let mut st = FactorState::new(n);
    if st.done() {
        factors.append(&mut st.primes);
        return Ok(());
    }

    // See [factor_auto], n is larger than 32 bits and has no factor below 12 bits.

    // Start with algorithms depending on prime size (trial factoring, Rho, P-1, ECM)
    // There is no need for recursion: running them twice does not bring benefits.
    // 90% of random integers have second largest factor O(n^1/3)

    // For small inputs, P-1 covers most primes and is faster than ECM and quadratic sieve.
    let start_pm1 = std::time::Instant::now();
    let pm1_res = pollard_pm1::pm1_auto(&st.nred, prefs.verbosity);
    // P-1 should be only run once.
    if let Some((a_s, _b)) = pm1_res {
        if prefs.verbose(Verbosity::Info) {
            eprintln!(
                "Pollard P-1 success with factors {a_s:?} in {:.3}s",
                start_pm1.elapsed().as_secs_f64()
            );
        }
        st.declare_factors(a_s);
        if st.done() {
            factors.append(&mut st.primes);
            return Ok(());
        }
        if prefs.verbose(Verbosity::Info) {
            eprintln!("Recursively factor {}", st.nred);
        }
    } else if prefs.verbose(Verbosity::Info) {
        eprintln!(
            "Pollard P-1 failure in {:.3}s",
            start_pm1.elapsed().as_secs_f64()
        );
    }
    loop {
        let ecm_res = ecm::ecm_auto(st.nred, prefs, tpool);
        if let Some(fs) = ecm_res {
            st.declare_factors(fs);
            if st.done() {
                factors.append(&mut st.primes);
                return Ok(());
            }
        } else {
            break;
        }
    }

    while !st.composites.is_empty() {
        let chk: Uint = st.composites.iter().sum();
        let facs = st.composites.clone();
        for f in &facs {
            if prefs.verbose(Verbosity::Info) {
                eprintln!("Recursively factor {f}");
            }
            if f.bits() <= 80 {
                if let Some(fs) = ecm::ecm_only(*f, prefs, tpool) {
                    st.declare_factors(fs);
                } else {
                    return Err(FactoringFailure);
                }
            } else {
                // Never call SIQS with a perfect power.
                let is_perfect_power = {
                    if f.bits() <= 64 {
                        // Use native arithmetic is possible.
                        arith::perfect_power(f.low_u64()).map(|_pk @ (p, k)| (p.into(), k))
                    } else if f.bits() <= 128 {
                        arith::perfect_power(u128::cast_from(*f)).map(|_pk @ (p, k)| (p.into(), k))
                    } else {
                        arith::perfect_power(*f)
                    }
                };
                if let Some((p, k)) = is_perfect_power {
                    let mut facs = vec![];
                    factor_auto_impl(p, prefs, &mut facs, tpool)?;
                    for _ in 0..k {
                        st.declare_factors(facs.clone());
                    }
                    continue;
                }

                let (k, score) = fbase::select_multiplier(*f);
                if prefs.verbose(Verbosity::Info) {
                    eprintln!("Selected multiplier {k} (score {score:.2}/10)");
                }
                //eprintln!("{n} => SIQS {f}");
                let divs = siqs::siqs(f, k, prefs, tpool);
                match divs {
                    Ok(divs) => st.declare_factors(divs),
                    Err(UnexpectedFactor(d)) => st.declare_factors(vec![d.into()]),
                };
            }
        }
        let chk2: Uint = st.composites.iter().sum();
        if chk == chk2 {
            break;
        }
        if st.done() {
            factors.append(&mut st.primes);
            return Ok(());
        }
    }
    if prefs.verbose(Verbosity::Info) {
        eprintln!("Factorization is incomplete.");
    }
    Err(FactoringFailure)
}

fn factor_small_impl(n: Uint, factor_bits: usize, factors: &mut Vec<Uint>) {
    if n.is_one() {
        return;
    }
    if pseudoprime(n) {
        factors.push(n);
        return;
    }
    let mut prefs = Preferences::default();
    prefs.verbosity = Verbosity::Silent;
    // Always try Pollard rho to clear very small factors
    // 1024 iterations will catch 99% of 16-bit factors.
    let mut nred: Uint = n;
    if nred.bits() >= 64 {
        // Quick Rho run to eliminate small factors.
        let rho_fast_iters: u64 = 128;
        if let Some((a_s, b)) = pollard_rho::rho_impl(&nred, 2, rho_fast_iters, prefs.verbosity) {
            // Force rho for recursion
            for a in a_s {
                factor_auto_impl(a, &prefs, factors, None).unwrap();
            }
            nred = b;
            if pseudoprime(nred) {
                factors.push(nred);
                return;
            }
        }
    }
    let rho_iters = 512;
    if let Some((a_s, b)) = pollard_rho::rho_impl(&nred, 3, rho_iters, prefs.verbosity) {
        // Force rho for recursion
        for a in a_s {
            factor_auto_impl(a, &prefs, factors, None).unwrap();
        }
        nred = b;
        if pseudoprime(nred) {
            factors.push(nred);
            return;
        }
    }
    if nred.is_one() {
        return;
    }
    if factor_bits <= 14 {
        factors.push(nred);
        return;
    }
    // Settings finding ~99% of 32-bit factors (see ecm128::ecm128)
    let (curves, b1, b2) = match factor_bits {
        0..=15 => (4, 20, 660.),
        16..=17 => (4, 30, 660.),
        18..=19 => (6, 40, 1080.),
        20..=21 => (8, 60, 1080.),
        22..=23 => (12, 80, 1920.),
        24..=25 => (12, 100, 3000.),
        26..=27 => (16, 150, 3000.),
        28..=29 => (20, 150, 5040.),
        _ => (24, 250, 7.7e3), // 30..32
    };
    // Try ECM
    while let Some(a_s) = ecm::ecm(nred, curves, b1 as usize, b2, &prefs, None) {
        for a in a_s {
            let d = Integer::gcd(&a, &nred);
            if Uint::ONE < d && d < nred {
                factor_auto_impl(d, &prefs, factors, None).unwrap();
                nred /= d;
            }
        }
        if pseudoprime(nred) {
            factors.push(nred);
            return;
        }
    }
    if factor_bits < 32 {
        if (nred.bits() as usize) < 2 * factor_bits && !pseudoprime(nred) {
            // If nred is composite and small, it has necessarily
            // a small factor.
            factor_auto_impl(nred, &prefs, factors, None).unwrap();
        } else {
            factors.push(nred);
        }
        return;
    }
    if factor_bits as f64 / nred.bits() as f64 > 0.4 {
        // This is equivalent to full factoring
        factor_auto_impl(nred, &prefs, factors, None).unwrap();
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
    while let Some(a_s) = ecm::ecm(nred, curves, b1 as usize, b2, &prefs, None) {
        for a in a_s {
            let d = Integer::gcd(&a, &nred);
            if Uint::ONE < d && d < nred {
                factor_auto_impl(d, &prefs, factors, None).unwrap();
                nred /= d;
            }
        }
        if pseudoprime(nred) {
            factors.push(nred);
            return;
        }
    }
    factors.push(nred);
}

/// Factor an integer which known to be completely factorable
/// using prime factors < 2^bits.
pub fn factor_smooth(n: Uint, factor_bits: usize) -> Vec<Uint> {
    if n.is_zero() {
        return vec![n];
    }
    let mut factors = vec![];
    let mut nred = n;
    let tz = nred.trailing_zeros();
    for _ in 0..tz {
        factors.push(Uint::from(2_u64));
    }
    nred >>= tz;
    // Perform trial division like factor_auto.
    if nred.bits() <= 32 {
        trialdiv::factor_u32(u32::cast_from(nred), &mut factors);
        return factors;
    } else if nred.bits() > 128 {
        trialdiv::factor128k(&mut nred, &mut factors);
    } else {
        trialdiv::factor4096(&mut nred, &mut factors);
    }
    //eprintln!("{n} trialdiv => {} {:?}", t0.elapsed().as_secs_f32() * 1e6, &factors);
    factor_smooth_impl(nred, factor_bits, &mut factors);
    // No primality testing of factors.
    factors.sort();
    factors
}

fn factor_smooth_impl(n: Uint, factor_bits: usize, factors: &mut Vec<Uint>) {
    let mut st = FactorState::new(n);
    if st.done() {
        factors.append(&mut st.primes);
        return;
    }

    // P-1 should be only run once.
    let facs = pollard_pm1::pm1_small(&st.nred, factor_bits);
    st.declare_factors(facs);
    if st.done() {
        factors.append(&mut st.primes);
        return;
    }

    if factor_bits < 16 || factor_bits > 96 {
        // No strategy: fallback to generic methods.
        factors.append(&mut st.primes);
        let mut prefs = Preferences::default();
        prefs.verbosity = Verbosity::Silent;
        for &c in &st.composites {
            factor_auto_impl(c, &prefs, factors, None).unwrap();
        }
        return;
    }

    // Use ECM (only 128 bits): usually 5-10 curves are enough.
    let (tries, b1, b2) = match factor_bits {
        16..=20 => (10, 50, 1920),     // cost=1000
        21..=24 => (20, 50, 1920),     // cost=1000
        25..=28 => (30, 75, 5040),     // cost=2000
        29..=30 => (30, 120, 13200),   // cost=4000
        31..=36 => (30, 280, 33000),   // cost=8000
        37..=44 => (30, 400, 126000),  // cost=16000
        45..=48 => (30, 2000, 181000), // cost=50000
        // Reuse parameters from ecm::ecm_only
        49..=64 => (100, 10_000, 554000),
        65..=80 => (100, 100_000, 19_000_000),
        81..=96 => (200, 300_000, 156_000_000),
        _ => unreachable!(),
    };

    // FIXME??
    let mut prefs = Preferences::default();
    prefs.verbosity = Verbosity::Silent;

    // Use general ECM
    while let Some(a_s) = ecm::ecm(st.nred, tries as usize, b1, b2 as f64, &prefs, None) {
        let old_nred = st.nred;
        st.declare_factors(a_s);
        if st.done() {
            factors.append(&mut st.primes);
            return;
        }
        if st.nred == old_nred {
            // No change, curves are pseudo-random, abort.
            break;
        }
    }

    factors.append(&mut st.primes);
    for &c in &st.composites {
        factor_auto_impl(c, &prefs, factors, None).unwrap();
    }
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

    // Many small factors
    let n = Uint::from_digit(412682899978);
    factor(n, Algo::Auto, &Preferences::default()).unwrap();

    // Triggered ecm128(1) in Auto mode.
    let n = Uint::from_str("116234185885542836921546102271413189262848299811057")?;
    factor(n, Algo::Auto, &Preferences::default()).unwrap();
    let n = Uint::from_str("35263301580611604739963993408020422755244243071902")?;
    factor(n, Algo::Auto, &Preferences::default()).unwrap();

    // Triggered infinite loop
    let n = Uint::from_str("29652120026020591938541823457")?;
    factor(n, Algo::Auto, &Preferences::default()).unwrap();

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
fn test_find_small_factors() -> Result<(), bnum::errors::ParseIntError> {
    let n = Uint::from_str(
        "12201879382649676470194887538548324244326362954552609354733068712098984100",
    )?;
    let fs = find_small_factors(n, 55);
    assert_eq!(fs.len(), 13);

    let n = Uint::from_str(
        "24091791282868184848805353591599648198248879500538357141333569815331594788",
    )?;
    let fs = find_small_factors(n, 55);
    assert_eq!(fs.len(), 14);

    let n = Uint::from_str("2491972896696220875515999140704747439734476676")?;
    let fs = find_small_factors(n, 20);
    // n = 2*2*3*23*229*443*14341*34883*98221*104123*17396053561176129467
    assert_eq!(fs.len(), 11);

    // Some primes below 1000 can escape all tests:
    let n = Uint::from(242153041166900_u64);
    let fs = find_small_factors(n, 19);
    // n = 2*2*5*5*199*281*853*50767
    assert_eq!(fs.len(), 8);

    let n = Uint::from(902002007847990_u64);
    let fs = find_small_factors(n, 19);
    assert_eq!(fs.len(), 9, "{fs:?}");

    let fs = find_small_factors(Uint::from(70054262122659589807_u128), 18);
    assert_eq!(fs.len(), 7, "{fs:?}");
    let fs = find_small_factors(Uint::from(3977516652591602527_u128), 18);
    assert_eq!(fs.len(), 8, "{fs:?}");
    let fs = find_small_factors(Uint::from(3452559913281303_u64), 18);
    assert_eq!(fs.len(), 10, "{fs:?}");
    let fs = find_small_factors(Uint::from(56891511204557826214_u128), 18);
    assert_eq!(fs.len(), 9, "{fs:?}");

    Ok(())
}

#[test]
fn test_factor_smooth() -> Result<(), bnum::errors::ParseIntError> {
    let n = Uint::from_str(
        "24091791282868184848805353591599648198248879500538357141333569815331594788",
    )?;
    let fs = factor_smooth(n, 28);
    assert_eq!(fs.len(), 14);

    let n = Uint::from_str("2491972896696220875515999140704747439734476676")?;
    let fs = factor_smooth(n, 20);
    // n = 2*2*3*23*229*443*14341*34883*98221*104123*1115111329*15600284123
    assert_eq!(fs.len(), 12, "{fs:?}");

    // Some primes below 1000 can escape all tests:
    let n = Uint::from(242153041166900_u64);
    let fs = factor_smooth(n, 19);
    // n = 2*2*5*5*199*281*853*50767
    assert_eq!(fs.len(), 8, "{fs:?}");

    let n = Uint::from(902002007847990_u64);
    let fs = factor_smooth(n, 19);
    assert_eq!(fs.len(), 9, "{fs:?}");

    let fs = factor_smooth(Uint::from(70054262122659589807_u128), 18);
    assert_eq!(fs.len(), 7, "{fs:?}");
    let fs = factor_smooth(Uint::from(3977516652591602527_u128), 18);
    assert_eq!(fs.len(), 8, "{fs:?}");
    let fs = factor_smooth(Uint::from(3452559913281303_u64), 18);
    assert_eq!(fs.len(), 10, "{fs:?}");
    let fs = factor_smooth(Uint::from(56891511204557826214_u128), 18);
    assert_eq!(fs.len(), 9, "{fs:?}");

    // Triggered infinite loop
    let n = Uint::from_str("7977858279185817933259895587935864322507169424708")?;
    let fs = factor_smooth(n, 24);
    assert_eq!(fs.len(), 11, "{fs:?}");

    Ok(())
}

#[test]
fn test_find_small_factors_random() {
    // Test that find_small_factors catches almost all factor
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
                let facs = find_small_factors(n, pbits);
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
                let facs = find_small_factors(n, pbits);
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
