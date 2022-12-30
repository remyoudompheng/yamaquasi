// Copyright 2022 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! A specialized version of Pollard P-1 for double large primes.
//! The target is to factor numbers of size 40-52 bits which are products
//! of primes above the factor base, slightly biased so that one factor
//! remains small.
//!
//! For example, while factoring RSA-100 half of numbers to be factored
//! have a factor such that p-1 is 5000-smooth.
//! The expected cost of SQUFOF is O(N^1/4) divisions so Pollard P-1
//! is interesting if it works in less than ~10000 multiplications.

use num_integer::Integer;

use crate::fbase;

/// A factor base for Pollard P-1.
/// Pairs of factors are multiplied into u32.
pub struct PM1Base {
    factors: Box<[u32]>,
    avg_size: usize,
}

impl PM1Base {
    pub fn new() -> Self {
        let primes = fbase::primes(8000);
        let mut factors = vec![];
        let mut buffer = 1_u64;
        let mut total_size = 0 as usize;
        for p in primes {
            // Small primes are raised to some power.
            let p = p as u64;
            let mut pow = p;
            while pow * p < 1024 {
                pow *= p;
            }
            if buffer * pow >= 1 << 32 {
                total_size += 32 - u32::leading_zeros(buffer as u32) as usize;
                factors.push(buffer as u32);
                buffer = 1;
            }
            buffer *= pow;
            if p == 65537 {
                break;
            }
        }
        if buffer > 1 {
            total_size += 32 - u32::leading_zeros(buffer as u32) as usize;
            factors.push(buffer as u32)
        }
        let avg_size = total_size / factors.len();
        PM1Base {
            factors: factors.into_boxed_slice(),
            avg_size,
        }
    }

    // Tentatively factor number n with a budget of squarings.
    // budget=40000 will factor about 75% of 50-bit semiprimes
    // for about 6x-8x less time than SQUFOF.
    pub fn factor(&self, n: u64, budget: usize) -> Option<(u64, u64)> {
        assert!(n % 2 == 1);
        // We have a lot of modular reductions to compute,
        // so we use Montgomery forms.
        // Precompute opposite inverse of n mod R (R=2^64)
        let ninv = {
            // Invariant: nx = 1 + 2^k s, k increasing
            let mut x = 1u64;
            loop {
                let rem = n.wrapping_mul(x) - 1;
                if rem == 0 {
                    break;
                }
                x += 1 << rem.trailing_zeros();
            }
            assert!(n.wrapping_mul(x) == 1);
            1 + !x
        };

        // Compute 2^K-1 mod n where K bit length is <= budget
        // 2R mod N,
        let one_r = ((1u128 << 64) % (n as u128)) as u64;
        let mut xr = (2 * one_r) % n;
        debug_assert!(mg_redc(n, ninv, xr as u128) == 2);
        let fmax = std::cmp::min(self.factors.len(), budget / self.avg_size + 1);
        for block in self.factors[..fmax].chunks(8) {
            for &f in block {
                // Compute x^f
                let mut res = one_r;
                let mut sq = xr;
                let mut exp = f;
                while exp > 0 {
                    if exp & 1 == 1 {
                        res = mg_mul(n, ninv, res, sq);
                    }
                    sq = mg_mul(n, ninv, sq, sq);
                    exp /= 2;
                }
                xr = res;
            }
            // Maybe we have finished?
            // No need to reduce out of Montgomery form, subtract R
            // to get 2^K R - R = (2^K-1)R
            let d = Integer::gcd(&n, &(xr + (n - one_r)));
            if d > 1 && d < n {
                return Some((d, n / d));
            }
        }
        let d = Integer::gcd(&n, &(xr + n - one_r));
        if d > 1 && d < n {
            Some((d, n / d))
        } else {
            None
        }
    }
}

// Montgomery form arithmetic

#[inline(always)]
fn mg_mul(n: u64, ninv: u64, x: u64, y: u64) -> u64 {
    mg_redc(n, ninv, (x as u128) * (y as u128))
}

#[inline(always)]
fn mg_redc(n: u64, ninv: u64, x: u128) -> u64 {
    // Montgomery reduction (x/R mod n).
    // compute -x/N mod R
    let mul: u64 = (x as u64).wrapping_mul(ninv);
    // reduce
    let m = mul as u128 * n as u128;
    let res = ((x + m) >> 64) as u64;
    if res >= n {
        res - n
    } else {
        res
    }
}

#[test]
fn test_pm1() {
    let ns: &[u64] = &[
        235075827453629, // max divisor 29129
        166130059616737, // max divisor 467
        159247921097933, // max divisor 3251
        224077614412439, // max divisor 1759
        219669028971857, // max divisor 193
    ];
    let pb = PM1Base::new();
    'nextn: for &n in ns {
        for budget in [1000, 2000, 5000, 10000, 20000, 50000, 80000] {
            if let Some((p, q)) = pb.factor(n, budget) {
                eprintln!("factored {n} with budget {budget} => {p}*{q}");
                assert_eq!(p * q, n);
                continue 'nextn;
            }
        }
        panic!("failed to factor {n}");
    }
}

#[test]
fn test_pm1_random() {
    for bits in [20, 22, 24, 26, 28] {
        for budget in [2000, 5000, 10000, 20000, 40000, 80000] {
            let mut seed = 1234567_u32;
            let mut primes = vec![];
            let mut ok = 0;
            let mut attempts = 0;
            let pb = PM1Base::new();
            for _ in 0..10000 {
                seed = seed.wrapping_mul(123456789);
                let p = seed % (1 << bits);
                if !fbase::certainly_composite(p as u64) {
                    primes.push(p);
                }
                if primes.len() == 2 {
                    let p = primes[0];
                    let q = primes[1];
                    attempts += 1;
                    if let Some((x, y)) = pb.factor(p as u64 * q as u64, budget) {
                        assert_eq!(x * y, p as u64 * q as u64);
                        ok += 1;
                    }
                    primes.clear();
                }
            }
            eprintln!(
                "{} bits, budget={budget} factored {ok}/{attempts} semiprimes",
                2 * bits
            );
        }
    }
}
