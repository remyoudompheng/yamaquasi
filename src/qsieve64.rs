// Copyright 2022 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! A specialized version of ordinary quadratic sieve
//! for smaller than 64-bit integers.
//!
//! Most parameters are hardcoded.
//! Factor base is ~20 primes, block size is 16k

use std::collections::HashMap;

use crate::arith::{self, Num};
use crate::fbase::{FBase, SMALL_PRIMES};
use crate::relations::{self, Relation};
use crate::{Uint, Verbosity};

pub fn qsieve(n: u64, v: Verbosity) -> Option<(u64, u64)> {
    // Prepare factor base
    let (k, score) = select_multiplier(n);
    if v >= Verbosity::Debug {
        eprintln!("Selected multiplier {} (score {:.2})", k, score);
    }
    // Handle perfect squares.
    let nsqrt = arith::isqrt(n);
    if n == nsqrt * nsqrt {
        return Some((nsqrt, nsqrt));
    }
    let nk = n * k as u64;
    let fb = FBase::new64(nk);
    if v >= Verbosity::Debug {
        eprintln!("Selected factor base {:?}", fb.primes);
    }
    let (primes, sqrts, divs) = (&fb.primes, &fb.sqrts, &fb.divs);

    let nsqrt = arith::isqrt(nk);
    if nk == nsqrt * nsqrt {
        if n == (nsqrt / k as u64) * nsqrt {
            return Some((nsqrt / k as u64, nsqrt));
        }
    }
    // (nsqrt+x)^2 - n
    let (b, c) = (2 * nsqrt, nk - nsqrt * nsqrt);
    if v >= Verbosity::Debug {
        eprintln!("Polynomial x^2 + {} x - {}", b, c);
    }

    let dummy_rset = relations::RelationSet::new(Uint::from(n), fb.len(), 0);
    let mut rels = vec![];
    let mut larges = HashMap::new();
    let mut extras = 0;
    let maxlarge = 5000;

    let block_size = match n.bits() {
        0..=50 => 4096,
        _ => 16384,
    };
    // Run sieve
    let mut interval = vec![0u8; 2 * block_size];
    for blk in 0..64 as i64 {
        // -1, 1, -3, 3, -5, 5, etc.
        let blk = if blk % 2 == 0 { -(blk + 1) } else { blk };
        let offset = blk * block_size as i64;
        for idx in 0..primes.len() {
            let p = primes[idx];
            let r = sqrts[idx];
            let div = &divs[idx];
            if p <= 3 {
                continue;
            }
            let logp = 32 - u32::leading_zeros(p) as u8;
            for rt in [r, p - r] {
                // nsqrt + offset + x is a root?
                let mut off = div.modi64(rt as i64 - offset - nsqrt as i64) as usize;
                while off < interval.len() {
                    unsafe {
                        *interval.get_unchecked_mut(off) += logp;
                    }
                    off += p as usize;
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
                for (pidx, &p) in primes.iter().enumerate() {
                    let mut exp = 0;
                    let div = &divs[pidx];
                    loop {
                        let (q, r) = div.divmod64(v);
                        if r == 0 {
                            v = q;
                            exp += 1;
                        } else {
                            break;
                        }
                    }
                    if exp > 0 {
                        factors.push((p as i64, exp));
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
                    cyclelen: 1,
                };
                if cofactor == 1 {
                    rels.push(rel)
                } else {
                    if let Some(r0) = larges.get(&cofactor) {
                        let rr = dummy_rset.combine(&rel, r0);
                        rels.push(rr);
                        extras += 1;
                    } else {
                        larges.insert(cofactor, rel);
                    }
                }
            }
        }
        // We cannot require too many relations for such small integers.
        if rels.len() > primes.len() + 8 {
            break;
        }
        if v >= Verbosity::Debug {
            eprintln!(
                "Found {} smooths (cofactors: {} combined, {} pending)",
                rels.len(),
                extras,
                larges.len(),
            );
        }
        interval.fill(0u8);
    }

    let divs = relations::final_step(&Uint::from(n), &fb, &rels, v);
    if let Some(p) = divs.first() {
        let p = p.low_u64();
        assert_eq!(n % p, 0);
        Some((p, n / p))
    } else {
        None
    }
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

#[test]
fn test_qsieve() {
    assert_eq!(
        qsieve(781418872441, Verbosity::Silent),
        Some((883979, 883979))
    );
    let mut factored = 0;
    for n in 1 << 49..(1 << 49) + 2000 {
        if n % 2 == 0 {
            // Modular arithmetic will fail.
            continue;
        }
        if crate::fbase::certainly_composite(n) {
            let Some((p, q)) = qsieve(n, Verbosity::Silent) else {
                panic!("FAIL {}", n);
            };
            assert_eq!(p * q, n);
            assert!(p > 1 && q > 1);
            factored += 1;
        }
    }
    eprintln!("{} numbers factored", factored);
}
