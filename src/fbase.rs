// Copyright 2022 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Routines related to the quadratic sieve factor base.
//! This file is common to all variants (QS, MPQS, SIQS).

use crate::arith;
use crate::arith::Num;
use crate::pollard_pm1;
use crate::{Int, Uint, UnexpectedFactor};

/// A factor base consisting of 24-bit primes related to an input number N,
/// along with useful precomputed data.
/// To help with memory locality, each additional information is held
/// in a separate vector.
#[derive(Clone, Debug)]
pub struct FBase {
    pub primes: Vec<u32>,
    // Square roots of N.
    pub sqrts: Vec<u32>,
    pub divs: Vec<arith::Dividers>,
    // idx_by_log[i] is the index of the first prime
    // such that bit_length >= i.
    pub idx_by_log: [usize; 24 + 2],
}

impl FBase {
    pub fn new(n: Uint, size: u32) -> Self {
        let ps = primes(2 * size);
        let mut primes = vec![];
        let mut sqrts = vec![];
        let mut divs = vec![];
        let mut idx_by_log = [0; 24 + 2];
        let mut log = 0;
        let mut prepared = prepare_factor_base(&n, &ps);
        // Align to a multiple of 8.
        // This is required for SIMD code.
        prepared.truncate(8 * (prepared.len() / 8));
        for (p, r, div) in prepared {
            let p = p as u32;
            let l = 32 - u32::leading_zeros(p) as usize;
            if l >= log {
                for idx in log..=l {
                    idx_by_log[idx] = primes.len();
                }
                log = l + 1;
            }
            primes.push(p as u32);
            sqrts.push(r as u32);
            divs.push(div);
        }
        assert!(primes.len() % 8 == 0);
        for idx in log..idx_by_log.len() {
            idx_by_log[idx] = primes.len();
        }
        FBase {
            primes,
            sqrts,
            divs,
            idx_by_log,
        }
    }

    pub fn smalls(&self) -> &[u32] {
        if self.len() >= 10 {
            &self.primes[..10]
        } else {
            &self.primes
        }
    }

    pub(crate) fn check_divisors(&self) -> Result<(), UnexpectedFactor> {
        if let Some(idx) = self.sqrts.iter().rposition(|&r| r == 0) {
            let p = self.primes[idx];
            if p > MAX_MULTIPLIER {
                return Err(UnexpectedFactor(p as u64));
            }
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.primes.len()
    }

    pub fn bound(&self) -> u32 {
        *self.primes.last().unwrap()
    }

    pub fn p(&self, idx: usize) -> u32 {
        self.primes[idx]
    }

    pub fn r(&self, idx: usize) -> u32 {
        self.sqrts[idx]
    }

    pub fn div(&self, idx: usize) -> &arith::Dividers {
        &self.divs[idx]
    }

    // Returns a (large) structure for a given prime.
    pub fn prime<'a>(&'a self, idx: usize) -> Prime<'a> {
        Prime {
            p: self.primes[idx] as u64,
            r: self.sqrts[idx] as u64,
            div: &self.divs[idx],
        }
    }
}

#[derive(Clone, Debug)]
pub struct Prime<'a> {
    pub p: u64, // prime number
    pub r: u64, // square root of N
    pub div: &'a arith::Dividers,
}

pub const SMALL_PRIMES: &[u64] = &[
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
    197, 199,
];

const MAX_MULTIPLIER: u32 = 200;

/// Selects k such kn is a quadratic residue modulo many small primes.
/// The scoring system is the average bit length of the smooth factor
/// of sieved numbers.
/// It is usually possible to obtain a score close to 10 with
/// a reasonably small multiplier.
pub fn select_multiplier(n: Uint) -> (u32, f64) {
    let mut best: u32 = 1;
    let mut best_score = 0.0;
    for k in 1..MAX_MULTIPLIER {
        let mag = expected_smooth_magnitude(&(n * Uint::from(k)));
        // A multiplier k increases the size of P(x) by sqrt(k)
        let mag = (mag - 0.5 * (k as f64).ln()) / std::f64::consts::LN_2;
        if mag > best_score {
            best_score = mag;
            best = k;
        }
    }
    (best, best_score)
}

/// The optimization criterion is the Knuth-Schroeppel formula
/// giving the expected number of small prime factors.
///
/// Reference: [Silverman, section 5]
///
/// Formula is corrected for the weight of 2 (1 instead of 2)
/// and denominator p-1 instead of p to account for prime
/// powers.
pub fn expected_smooth_magnitude(n: &Uint) -> f64 {
    let mut res: f64 = 0.0;
    for &p in SMALL_PRIMES {
        let np: u64 = *n % p;
        let exp = if p == 2 {
            match *n % 8u64 {
                // Modulo 8:
                // n has 4 square roots modulo 8 but also modulo 16, 32 etc.
                // 3/2 + 1/4 + 1/8 + ... = 2
                //
                // When using type 2 polynomials (2Ax+B)^2-n / 4:
                // - we only use odd (2Ax+B) => multiply score by 2
                // - we eliminate a factor 4 => subtract 2
                // - values are 2x smaller => add 1
                // Final score: 2*2 - 2 + 1 = 3
                1 => 3.0,
                // x²-n can be divisible by 4 (half of the time) but never by 8.
                // When using a type 2 polynomial, the factor 4 is already eliminated
                // but there is an additional score 1 because polynomial values are 2x smaller.
                5 => 1.0,
                // x²-n can never be divisible by 4.
                // It is divisible by 2 half of the time.
                3 | 7 => 0.5,
                _ => 0.0,
            }
        } else if np == 0 {
            1.0 / (p - 1) as f64
        } else if let Some(_) = arith::sqrt_mod(np, p) {
            2.0 / (p - 1) as f64
        } else {
            0.0
        };
        res += exp * (p as f64).ln();
    }
    res
}

pub fn primes(n: u32) -> Vec<u32> {
    let bound = (n * 2 * (32 - n.leading_zeros())) as usize;
    let mut sieve = vec![0; bound];

    let mut primes = vec![];
    for p in 2..sieve.len() {
        if sieve[p] == 0 {
            primes.push(p as u32);
            if primes.len() == n as usize {
                break;
            }
            let mut k = 2 * p as usize;
            while k < bound {
                sieve[k] = 1;
                k += p
            }
        }
    }
    primes
}

// An iterator over blocks of prime numbers until 2^32.
// It is initialized with 6542 primes under 2^16 and sieves intervals
// of size 2^16 to keep low memory footprint.
//
// The main usage is Pollard P-1 with large B2.
pub struct PrimeSieve {
    smallprimes: Box<[u32]>,
    block: Vec<u32>,
    block_count: usize,
    offsets: Vec<u32>,
    sieve: [bool; 1 << 16],
}

impl PrimeSieve {
    pub fn new() -> Self {
        let smalls = primes(6542);
        assert_eq!(smalls.last(), Some(&65521));
        // If 65535%p = k, 65536 + (p-1-k) is a multiple of p
        let offsets = smalls.iter().map(|&p| p - 1 - 65535 % p).collect();
        Self {
            smallprimes: smalls.into_boxed_slice(),
            block: vec![],
            block_count: 0,
            offsets,
            sieve: [false; 1 << 16],
        }
    }

    pub fn next(&mut self) -> &[u32] {
        if self.block_count == 65536 {
            // The end
            self.block.clear();
            &self.block
        } else if self.block_count == 0 {
            self.block_count += 1;
            &self.smallprimes
        } else {
            // sieve a block
            self.sieve.fill(false);
            for (&p, off) in self.smallprimes.iter().zip(self.offsets.iter_mut()) {
                let mut o = *off as usize;
                while o < self.sieve.len() {
                    self.sieve[o] = true;
                    o += p as usize;
                }
                *off = o as u32 - 65536;
            }
            self.block.clear();
            for (idx, b) in self.sieve.iter().enumerate() {
                if !b {
                    self.block.push(((self.block_count << 16) + idx) as u32);
                }
            }
            self.block_count += 1;
            &self.block
        }
    }
}

fn prepare_factor_base(nk: &Uint, primes: &[u32]) -> Vec<(u64, u64, arith::Dividers)> {
    primes
        .iter()
        .filter_map(|&p| {
            // All factor base elements are required to fit in 24 bits.
            if p >= 1 << 24 {
                return None;
            }
            let nk: u64 = *nk % (p as u64);
            let r = arith::sqrt_mod(nk, p as u64)?;
            Some((p as u64, r, arith::Dividers::new(p)))
        })
        .collect()
}

// Returns whether n is composite through an Euler witness.
// The use case is a product of 2 odd primes (these are never
// Carmichael numbers).
//
// Random testing on 48-bit semiprimes show that 2 is almost
// never an Euler liar (probability < 1e-6), but for example
// 2^(n-1) = 1 mod n for n = 173142166387457
pub fn certainly_composite(n: u64) -> bool {
    if n > 2 && n % 2 == 0 {
        return true;
    }
    let n128 = n as u128;
    let x = arith::pow_mod(2u128, n128 >> 1, n128) as u64;
    x != 1 && x != n - 1
}

// Try to factor a possible "double large prime".
// A number of assumptions are made, in particular
// than composites are necessary more than 24 bit wide.
pub fn try_factor64(fb: Option<&pollard_pm1::PM1Base>, n: u64) -> Option<(u64, u64)> {
    if n >> 24 == 0 || !certainly_composite(n) {
        return None;
    }
    let pm1_budget = match n.bits() {
        0..=42 => 5000,
        43..=45 => 10000,
        46..=48 => 20000,
        49.. => 30000,
    };
    if let Some(pq) = fb.and_then(|fb| fb.factor(n, pm1_budget)) {
        return Some(pq);
    }
    crate::squfof::squfof(n)
}

// Performs trial division of x by fbase[idx] for prime indices in facs.
// Returns the remaining cofactor as a product pq and a list of factors with exponents.
// If the cofactor is too large, return None.
#[inline]
pub fn cofactor(
    fbase: &FBase,
    x: &Int,
    facs: &[usize],
    maxlarge: u64,
    max_cofactor: u64,
    pm1_base: Option<&pollard_pm1::PM1Base>,
) -> Option<((u64, u64), Vec<(i64, u64)>)> {
    let mut factors: Vec<(i64, u64)> = Vec::with_capacity(20);
    if x.is_negative() {
        factors.push((-1, 1));
    }
    let xabs = x.abs().to_bits();
    let mut cofactor = xabs;
    for &pidx in facs {
        let pp = fbase.p(pidx);
        let pdiv = fbase.div(pidx);
        let mut exp = 0;
        loop {
            let (q, r) = pdiv.divmod_uint(&cofactor);
            if r == 0 {
                cofactor = q;
                exp += 1;
            } else {
                break;
            }
        }
        // FIXME: we should have exp > 0
        if exp > 0 {
            factors.push((pp as i64, exp));
        }
    }
    let cofactor = cofactor.to_u64()?;
    if cofactor > max_cofactor {
        return None;
    }
    let maxprime = fbase.bound() as u64;
    let pq = if cofactor > maxprime * maxprime {
        // Possibly a double large prime
        let pq = try_factor64(pm1_base, cofactor);
        match pq {
            Some((p, q)) if p > maxlarge || q > maxlarge => None,
            None if cofactor > maxlarge => None,
            _ => pq,
        }
    } else {
        // Must be prime
        debug_assert!(!certainly_composite(cofactor));
        if cofactor > maxlarge {
            None
        } else {
            Some((cofactor, 1))
        }
    };
    Some((pq?, factors))
}

#[test]
fn test_primesieve() {
    let mut s = PrimeSieve::new();
    loop {
        let block = s.next();
        if block[0] >= 2 << 20 {
            // There are 4533 primes between 2<<20 and 2<<20 + 65536
            // from 2097169 to 2162681
            assert_eq!(block.len(), 4533);
            assert_eq!(block[0], 2097169);
            assert_eq!(block[4532], 2162681);
            eprintln!("{:?} .. {:?}", &block[..5], &block[block.len() - 5..]);
            break;
        }
    }
}
