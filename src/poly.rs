//! Polynomial selection for Multiple Polynomial Quadratic Sieve
//!
//! Bibliography:
//! Robert D. Silverman, The multiple polynomial quadratic sieve
//! Math. Comp. 48, 1987, https://doi.org/10.1090/S0025-5718-1987-0866119-8

use crate::arith::{inv_mod, isqrt, pow_mod, Num};
use crate::Uint;
use num_traits::One;

#[cfg(test)]
use std::str::FromStr;

// Helpers for polynomial selection

#[derive(Copy, Clone, Debug)]
pub struct Prime {
    pub p: u64, // prime number
    pub r: u64, // square root of N
}

pub struct SievePrime {
    pub p: u64,          // prime number
    pub r: u64,          // square root of N
    pub roots: [u64; 2], // polynomial roots
}

/// Compute sieving parameters for polynomial (X+offset)^2 - n
pub fn simple_prime(p: Prime, offset: Uint) -> SievePrime {
    let shift: u64 = p.p - (offset % Uint::from(p.p)).to_u64().unwrap();
    SievePrime {
        p: p.p,
        r: p.r,
        roots: [(p.r + shift) % p.p, (p.p - p.r + shift) % p.p],
    }
}

#[test]
fn test_simple_prime() {
    let p = Prime { p: 10223, r: 4526 };
    let nsqrt = Uint::from_str("13697025762053691031049747437678526773503028576").unwrap();
    let sp = simple_prime(p, nsqrt);
    let rr = nsqrt + Uint::from(sp.roots[0]);
    assert_eq!((rr * rr) % p.p, (p.r * p.r) % p.p);
    let rr = nsqrt + Uint::from(sp.roots[1]);
    assert_eq!((rr * rr) % p.p, (p.r * p.r) % p.p);
}

/// Selects k such kn is a quadratic residue modulo many small primes.
pub fn select_multipliers(n: Uint) -> Vec<u32> {
    let mut res = vec![];
    let bases: &[u32] = &[8, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31];
    let residues: Vec<u32> = bases
        .iter()
        .map(|&b| (n % Uint::from(b)).to_u64().unwrap() as u32)
        .collect();
    for i in 1..30000 {
        let k = 2 * i + 1;
        let mut all_squares = true;
        for (idx, &b) in bases.iter().enumerate() {
            let mut square = false;
            for x in 1..b {
                if (x * x) % b == (k * residues[idx]) % b {
                    square = true;
                    break;
                }
            }
            if !square {
                all_squares = false;
                break;
            }
        }
        if all_squares {
            res.push(k)
        }
    }
    res
}
