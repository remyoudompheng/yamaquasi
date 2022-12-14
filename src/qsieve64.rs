// Copyright 2022 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! A specialized version of ordinary quadratic sieve for 64-bit integers.
//!
//! Most parameters are hardcoded.
//! Factor base is ~20 primes, block size is 16k

use std::collections::HashMap;

use crate::arith::{self, Num};
use crate::fbase::{Prime, SMALL_PRIMES};
use crate::relations::{combine_large_relation, final_step, Relation};
use crate::Uint;

const DEBUG: bool = false;

pub fn qsieve(n: u64) -> Option<(u64, u64)> {
    // Prepare factor base
    let mut primes = Vec::with_capacity(25);
    let (k, score) = select_multiplier(n);
    if DEBUG {
        eprintln!("Selected multiplier {} (score {:.2})", k, score);
    }
    let nk = n * k as u64;
    for &p in SMALL_PRIMES {
        if let Some(r) = arith::sqrt_mod(nk, p as u64) {
            primes.push(Prime {
                p: p as u64,
                r: r,
                div: arith::Dividers::new(p as u32),
            });
        }
    }
    if DEBUG {
        eprintln!(
            "Selected factor base {:?}",
            primes.iter().map(|p| p.p).collect::<Vec<_>>()
        );
    }

    let nsqrt = arith::isqrt(nk);
    // (nsqrt+x)^2 - n
    let (b, c) = (2 * nsqrt, nk - nsqrt * nsqrt);
    if DEBUG {
        eprintln!("Polynomial x^2 + {} x - {}", b, c);
    }

    let mut relations = vec![];
    let mut larges = HashMap::new();
    let mut extras = 0;
    let maxlarge = 5000;

    let block_size = match n.bits() {
        1..=50 => 4096,
        _ => 16384,
    };
    // Run sieve
    let mut interval = vec![0u8; 2 * block_size];
    for blk in 0..64 {
        // -1, 1, -3, 3, -5, 5, etc.
        let blk = if blk % 2 == 0 { -(blk + 1) as i64 } else { blk };
        let offset = blk * block_size as i64;
        for p in &primes {
            if p.p <= 3 {
                continue;
            }
            let logp = 32 - u32::leading_zeros(p.p as u32) as u8;
            for rt in [p.r, p.p - p.r] {
                // nsqrt + offset + x is a root?
                let mut off = p.div.modi64(rt as i64 - offset - nsqrt as i64) as usize;
                while off < interval.len() {
                    interval[off] += logp;
                    off += p.p as usize;
                }
            }
        }
        // Factor smooths
        let target = n.bits() / 2 + 16 - maxlarge.bits() - 2;
        for (i, &sz) in interval.iter().enumerate() {
            if sz >= target as u8 {
                let x = i as i64 + offset;
                let u = nsqrt as i64 + x;
                let mut v = (x + b as i64) * x - c as i64;
                let mut factors: Vec<(i64, u64)> = vec![];
                if v < 0 {
                    factors.push((-1, 1));
                    v = -v;
                }
                let mut v = v as u64;
                // Factor candidate
                for p in &primes {
                    let mut exp = 0;
                    loop {
                        let (q, r) = p.div.divmod64(v);
                        if r == 0 {
                            v = q;
                            exp += 1;
                        } else {
                            break;
                        }
                    }
                    if exp > 0 {
                        factors.push((p.p as i64, exp));
                    }
                }
                let cofactor = v;
                if cofactor >= maxlarge {
                    continue;
                }
                //eprintln!("relation {}^2 = {} * product({:?})", u, cofactor, &factors);
                // Process relation
                let rel = Relation {
                    x: Uint::from(u as u64),
                    cofactor,
                    factors,
                };
                if cofactor == 1 {
                    relations.push(rel)
                } else {
                    if let Some(rr) = combine_large_relation(&mut larges, &rel, &Uint::from(n)) {
                        relations.push(rr);
                        extras += 1;
                    }
                }
            }
        }
        if relations.len() > primes.len() + 10 {
            break;
        }
        if DEBUG {
            eprintln!(
                "Found {} smooths (cofactors: {} combined, {} pending)",
                relations.len(),
                extras,
                larges.len(),
            );
        }
        interval.fill(0u8);
    }

    final_step(&Uint::from(n), &relations, DEBUG).map(|(a, b)| {
        let (a, b) = (a.low_u64(), b.low_u64());
        assert_eq!(n, a * b);
        (a, b)
    })
}

pub fn select_multiplier(n: u64) -> (u32, f64) {
    let mut best = 1;
    let mut best_score = 0.0;
    let mut nk = n;
    for k in 1..30 {
        let mag = expected_smooth_magnitude(nk);
        let mag = (mag - 0.5 * (k as f64).ln()) / std::f64::consts::LN_2;
        if mag > best_score {
            best_score = mag;
            best = k;
        }
        if let Some(x) = nk.checked_add(n) {
            nk = x
        } else {
            break;
        }
    }
    (best, best_score)
}

fn expected_smooth_magnitude(n: u64) -> f64 {
    let mut res: f64 = 0.0;
    for &p in &SMALL_PRIMES[..20] {
        let np: u64 = n % p;
        let exp = if p == 2 {
            match n % 8u64 {
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
            1 as f64 / (p - 1) as f64
        } else if let Some(_) = arith::sqrt_mod(np, p) {
            2 as f64 / (p - 1) as f64
        } else {
            0.0
        };
        res += exp * (p as f64).ln();
    }
    res
}
