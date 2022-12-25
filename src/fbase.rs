// Copyright 2022 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Routines related to the quadratic sieve factor base.
//! This file is common to all variants (QS, MPQS, SIQS).

use crate::arith;
use crate::Uint;

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
    // such that bit_length == i.
    //pub idx_by_log: [usize; 24+1],
}

impl FBase {
    pub fn new(n: Uint, size: u32) -> Self {
        let ps = primes(2 * size);
        let mut primes = vec![];
        let mut sqrts = vec![];
        let mut divs = vec![];
        for (p, r, div) in prepare_factor_base(&n, &ps) {
            primes.push(p as u32);
            sqrts.push(r as u32);
            divs.push(div);
        }
        FBase {
            primes,
            sqrts,
            divs,
        }
    }

    pub fn smalls(&self) -> &[u32] {
        if self.len() >= 10 {
            &self.primes[..10]
        } else {
            &self.primes
        }
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
];

/// Selects k such kn is a quadratic residue modulo many small primes.
/// The scoring system is the average bit length of the smooth factor
/// of sieved numbers.
pub fn select_multiplier(n: Uint) -> (u32, f64) {
    let mut best = 1;
    let mut best_score = 0.0;
    for k in 1..100 {
        let mag = expected_smooth_magnitude(&(n * Uint::from(k)));
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
                // square root modulo every power of 2
                // score is 1/2 + 1/4 + ...
                1 => 1.0,
                // square root modulo 2 and 4, score is 1/2 + 1/4
                5 => 0.75,
                // square root modulo 2, score 1/2
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

fn prepare_factor_base(nk: &Uint, primes: &[u32]) -> Vec<(u64, u64, arith::Dividers)> {
    primes
        .iter()
        .filter_map(|&p| {
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
pub fn try_factor64(n: u64) -> Option<(u64, u64)> {
    if n >> 24 == 0 || !certainly_composite(n) {
        return None;
    }
    crate::squfof::squfof(n)
}
