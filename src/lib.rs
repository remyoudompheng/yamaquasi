// Copyright 2022 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

pub mod arith;
pub mod arith_montgomery;
pub mod fbase;
pub mod matrix;
pub mod params;
pub mod relations;

// Implementations
pub mod ecm;
pub mod mpqs;
pub mod pollard_pm1;
pub mod qsieve;
pub mod qsieve64;
pub mod sieve;
pub mod siqs;
pub mod squfof;

// We need to perform modular multiplication modulo the input number.
pub type Int = arith::I1024;
pub type Uint = arith::U1024;

// Top-level functions
use std::str::FromStr;

use arith::Num;
use bnum::cast::CastFrom;
use num_integer::Integer;

const DEBUG: bool = false;

#[derive(Default)]
pub struct Preferences {
    pub fb_size: Option<u32>,
    pub large_factor: Option<u64>,
    pub use_double: Option<bool>,
    pub threads: Option<usize>,
    pub verbose: bool,
}

#[derive(PartialEq, Eq, Clone, Copy)]
pub enum Algo {
    Auto,
    Squfof,
    Qs64,
    Pm1,
    Ecm,
    Qs,
    Mpqs,
    Siqs,
}

impl FromStr for Algo {
    type Err = Box<dyn std::error::Error>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "auto" => Ok(Self::Auto),
            "squfof" => Ok(Self::Squfof),
            "pm1" => Ok(Self::Pm1),
            "ecm" => Ok(Self::Ecm),
            "qs" => Ok(Self::Qs),
            "qs64" => Ok(Self::Qs64),
            "mpqs" => Ok(Self::Mpqs),
            "siqs" => Ok(Self::Siqs),
            _ => Err(format!("invalid algo {}", s).into()),
        }
    }
}

/// An "error" describing a (small) factor encountered unexpectedly
/// during algorithms. This includes factor bases containing a divisor
/// of N or ECM generating a singular curve.
#[derive(Debug)]
pub struct UnexpectedFactor(u64);

/// Factorizes an integer into a product of factors.
pub fn factor(n: Uint, alg: Algo, prefs: &Preferences) -> Vec<Uint> {
    if n.is_zero() {
        return vec![n];
    }
    let mut factors = vec![];
    if prefs.verbose {
        eprintln!("Testing small prime divisors");
    }
    let mut nred = n;
    for &p in fbase::SMALL_PRIMES {
        while nred % (p as u64) == 0 {
            let p = Uint::from(p);
            factors.push(p);
            nred /= p;
            if prefs.verbose {
                eprintln!("Found small factor {p}");
            }
        }
    }
    if nred != n && prefs.verbose {
        eprintln!("Factoring {nred}");
    }
    // Create thread pool
    let tpool: Option<rayon::ThreadPool> = prefs.threads.map(|t| {
        if prefs.verbose {
            eprintln!("Using a pool of {t} threads");
        }
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build()
            .expect("cannot create thread pool")
    });
    let tpool = tpool.as_ref();
    factor_impl(nred, alg, prefs, &mut factors, tpool);

    check_factors(&n, &factors);
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
    } else if let Some((p, k)) = arith::perfect_power(n) {
        for _ in 0..k {
            factors.push(p)
        }
        return;
    } else if pseudoprime(n) {
        factors.push(n);
        return;
    }
    // Do we need to try an ECM step?
    match alg {
        Algo::Auto => {
            // Only in automatic mode, for large inputs, Pollard P-1 and ECM can be useful.
            if n.bits() >= 150 {
                let start_pm1 = std::time::Instant::now();
                if let Some((a, b)) = pollard_pm1::pm1_quick(n) {
                    eprintln!(
                        "Pollard P-1 success with factor p={a} in {:.3}s",
                        start_pm1.elapsed().as_secs_f64()
                    );
                    factor_impl(a.into(), alg, prefs, factors, tpool);
                    eprintln!("Recursively factor {b}");
                    factor_impl(b.into(), alg, prefs, factors, tpool);
                    return;
                } else {
                    eprintln!(
                        "Pollard P-1 failure in {:.3}s",
                        start_pm1.elapsed().as_secs_f64()
                    );
                }
            }
            if n.bits() > 190 {
                if let Some((a, b)) = ecm::ecm_auto(n, tpool) {
                    factor_impl(a.into(), alg, prefs, factors, tpool);
                    eprintln!("Recursively factor {b}");
                    factor_impl(b.into(), alg, prefs, factors, tpool);
                    return;
                }
            }
        }
        Algo::Pm1 => {
            // Pure Pollard P-1
            let start_pm1 = std::time::Instant::now();
            if let Some((a, b)) = pollard_pm1::pm1_only(n) {
                eprintln!(
                    "Pollard P-1 success with factor p={a} in {:.3}s",
                    start_pm1.elapsed().as_secs_f64()
                );
                factor_impl(a.into(), alg, prefs, factors, tpool);
                eprintln!("Recursively factor {b}");
                factor_impl(b.into(), alg, prefs, factors, tpool);
                return;
            } else {
                eprintln!(
                    "Pollard P-1 failure in {:.3}s",
                    start_pm1.elapsed().as_secs_f64()
                );
            }
            factors.push(n);
            return;
        }
        Algo::Ecm => {
            // Pure ECM is requested.
            // However due to determinism the recursion will go through the
            // same curves, which is not very useful.
            if let Some((a, b)) = ecm::ecm_only(n, tpool) {
                factor_impl(a.into(), alg, prefs, factors, tpool);
                eprintln!("Recursively factor {b}");
                factor_impl(b.into(), alg, prefs, factors, tpool);
                return;
            }
            eprintln!("Factorization is incomplete.");
            factors.push(n);
            return;
        }
        _ => {}
    }
    // Select algorithm
    let alg_real = if let Algo::Auto = alg {
        if n.bits() <= 60 {
            Algo::Squfof
        } else {
            Algo::Siqs
        }
    } else {
        alg
    };
    match alg_real {
        Algo::Qs64 => {
            assert!(n.bits() <= 64);
            if let Some((a, b)) = qsieve64::qsieve(n.low_u64()) {
                // Recurse
                factor_impl(a.into(), alg, prefs, factors, tpool);
                factor_impl(b.into(), alg, prefs, factors, tpool);
            } else {
                eprintln!("qsieve64 failed");
                factors.push(n);
            }
            return;
        }
        Algo::Squfof => {
            assert!(n.bits() <= 64);
            if let Some((a, b)) = squfof::squfof(n.low_u64()) {
                factor_impl(a.into(), alg, prefs, factors, tpool);
                factor_impl(b.into(), alg, prefs, factors, tpool);
            } else {
                eprintln!("SQUFOF failed");
                factors.push(n);
            }
            return;
        }
        _ => {}
    }

    let (k, score) = fbase::select_multiplier(n);
    if prefs.verbose {
        eprintln!("Selected multiplier {k} (score {score:.2}/10)");
    }
    let nk = n * Uint::from(k);
    // TODO: handle the case where n is not coprime to the factor base
    // TODO: handle the case of prime powers
    let rels = match alg_real {
        Algo::Auto => unreachable!("impossible"),
        Algo::Qs64 => unreachable!("impossible"),
        Algo::Squfof => unreachable!("impossible"),
        Algo::Pm1 => unreachable!("impossible"),
        Algo::Ecm => unreachable!("impossible"),
        Algo::Qs => Ok(qsieve::qsieve(nk, &prefs, tpool)),
        Algo::Mpqs => Ok(mpqs::mpqs(nk, &prefs, tpool)),
        Algo::Siqs => siqs::siqs(&nk, &prefs, tpool),
    };
    let rels = match rels {
        Ok(rels) => rels,
        Err(UnexpectedFactor(d)) => {
            factor_impl(d.into(), alg, prefs, factors, tpool);
            factor_impl(n / Uint::from(d), alg, prefs, factors, tpool);
            return;
        }
    };
    // Determine non trivial divisors.
    let divs = relations::final_step(&n, &rels, true);
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
        assert_eq!(residue.to_u64(), Some(1));
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
        if let Some((p, k)) = arith::perfect_power(f) {
            for _ in 0..k {
                factors.push(p)
            }
        } else if !pseudoprime(f) {
            eprintln!("Recursively factor {f}");
            factor_impl(f, alg, prefs, factors, tpool);
        } else {
            factors.push(f);
        }
    }
}

fn check_factors(n: &Uint, factors: &[Uint]) {
    if let &[p] = &factors {
        assert_eq!(n, p);
        assert!(pseudoprime(*p));
    }
    assert_eq!(*n, factors.iter().product::<Uint>());
}

pub fn pseudoprime(p: Uint) -> bool {
    // A basic Miller test with small bases.
    let s = (p.low_u64() - 1).trailing_zeros();
    for &b in fbase::SMALL_PRIMES {
        if p.to_u64() == Some(b) {
            return true;
        }
        if p.bits() <= 64 && b > 37 {
            // Bases up to 37 are enough for 64-bit integers.
            break;
        }
        let mut pow = arith::pow_mod(Uint::from(b), Uint::cast_from(p) >> s, Uint::cast_from(p));
        let p = Uint::cast_from(p);
        let pm1 = p - Uint::from(1u64);
        let mut ok = pow.to_u64() == Some(1) || pow == pm1;
        for _ in 0..s {
            pow = (pow * pow) % p;
            if pow == pm1 {
                ok = true;
                break;
            } else if pow.to_u64() == Some(1) {
                break;
            }
        }
        if !ok {
            return false;
        }
    }
    return true;
}

#[test]
fn test_factor() -> Result<(), bnum::errors::ParseIntError> {
    // semiprime
    eprintln!("=> test semiprime");
    let n = Uint::from_str("404385851501206046375042621")?;
    factor(n, Algo::Auto, &Preferences::default());

    // small factor (2003 * 665199750163226410868760173)
    eprintln!("=> test small factor 1");
    let n = Uint::from_str("1332395099576942500970126626519")?;
    factor(n, Algo::Auto, &Preferences::default());

    // 2 small factors (443 * 1151 * 172633679917074861804179395686166722361211)
    let n = Uint::from_str("88024704953957052509918444604606608564924960423")?;
    factor(n, Algo::Auto, &Preferences::default());

    // Could trigger an infinite loop after failing to factor due to
    // 223 being in the factor base.
    // 223*28579484042221159639852413780078523
    eprintln!("=> small factor 3");
    let n = Uint::from_str("317546892790192732050746209")?;
    factor(n, Algo::Siqs, &Preferences::default());

    // perfect square (17819845476047^2)
    eprintln!("=> test square");
    let n = Uint::from_str("317546892790192732050746209")?;
    factor(n, Algo::Auto, &Preferences::default());
    // perfect cube
    eprintln!("=> test cube");
    let n = Uint::from_str("350521251909490182639506149")?;
    factor(n, Algo::Auto, &Preferences::default());
    // not squarefree (839322217^2 * 705079549)
    eprintln!("=> test not squarefree");
    let n = Uint::from_str("496701596915056959994534861")?;
    factor(n, Algo::Auto, &Preferences::default());
    Ok(())
}
