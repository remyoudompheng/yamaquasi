//! Polynomial selection for Multiple Polynomial Quadratic Sieve
//!
//! Bibliography:
//! Robert D. Silverman, The multiple polynomial quadratic sieve
//! Math. Comp. 48, 1987, https://doi.org/10.1090/S0025-5718-1987-0866119-8

use crate::arith::{self, inv_mod, pow_mod, Num};
use crate::Uint;
use num_traits::One;

#[cfg(test)]
use std::str::FromStr;

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

pub fn prepare_factor_base(nk: &Uint, primes: &[u32]) -> Vec<Prime> {
    primes
        .into_iter()
        .filter_map(|&p| {
            let nk: u64 = *nk % (p as u64);
            let r = arith::sqrt_mod(nk, p as u64)?;
            Some(Prime {
                p: p as u64,
                r: r,
                div: arith::Dividers::new(p),
            })
        })
        .collect()
}

// Helpers for polynomial selection

#[derive(Clone, Debug)]
pub struct Prime {
    pub p: u64, // prime number
    pub r: u64, // square root of N
    pub div: arith::Dividers,
}

pub struct SievePrime {
    pub p: u64,          // prime number
    pub roots: [u64; 2], // polynomial roots
}

/// Compute sieving parameters for polynomial (X+offset)^2 - n
pub fn simple_prime(p: &Prime, offset: Uint) -> SievePrime {
    let shift: u64 = p.p - (offset % Uint::from(p.p)).to_u64().unwrap();
    SievePrime {
        p: p.p,
        roots: [(p.r + shift) % p.p, (p.p - p.r + shift) % p.p],
    }
}

#[test]
fn test_simple_prime() {
    let p = Prime {
        p: 10223,
        r: 4526,
        div: arith::Dividers::new(10223),
    };
    let nsqrt = Uint::from_str("13697025762053691031049747437678526773503028576").unwrap();
    let sp = simple_prime(&p, nsqrt);
    let rr = nsqrt + Uint::from(sp.roots[0]);
    assert_eq!((rr * rr) % p.p, (p.r * p.r) % p.p);
    let rr = nsqrt + Uint::from(sp.roots[1]);
    assert_eq!((rr * rr) % p.p, (p.r * p.r) % p.p);
}

/// Selects k such kn is a quadratic residue modulo many small primes.
/// The scoring system is the average bit length of the smooth factor
/// of sieved numbers.
pub fn select_multiplier(n: Uint) -> (u32, f64) {
    let mut best = 1;
    let mut best_score = 0.0;
    for k in 1..100 {
        let mag = expected_smooth_magnitude(&(n * Uint::from(k)));
        let mag = (mag - 0.5 * (k as f64).ln()) / std::f64::consts::LN_2;
        if mag > best_score && k % 2 == 1 {
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

/// A polynomial is an omitted quadratic Ax^2 + Bx + C
/// such that (ax+b)^2-n = d^2(Ax^2 + Bx + C) and A=d^2
/// and the polynomial values are small.
///
/// The polynomial values are divisible by p iff
/// ax+b is a square root of n modulo p
#[derive(Debug)]
pub struct Poly {
    pub a: Uint,
    pub b: Uint,
    pub d: Uint,
}

impl Poly {
    pub fn prepare_prime(&self, p: &Prime) -> SievePrime {
        // If p == 2, (2A+B)^2 is always equal to n
        if p.p == 2 {
            return SievePrime {
                p: p.p,
                roots: [0, 1],
            };
        }
        // Transform roots as: r -> (r - B) / 2A
        let a: u64 = p.div.divmod_uint(&self.a).1;
        let b: u64 = p.div.divmod_uint(&self.b).1;
        let ainv = p.div.inv(a).unwrap();
        SievePrime {
            p: p.p,
            roots: [
                p.div.divmod64((p.p + p.r - b) * ainv).1,
                p.div.divmod64((p.p - p.r + p.p - b) * ainv).1,
            ],
        }
    }
}

#[test]
fn test_poly_prime() {
    let p = Prime {
        p: 10223,
        r: 4526,
        div: arith::Dividers::new(10223),
    };
    let poly = Poly {
        a: Uint::from_str("13628964805482736048449433716121").unwrap(),
        b: Uint::from_str("2255304218805619815720698662795").unwrap(),
        d: Uint::from(3691742787015739u64),
    };
    let sp = poly.prepare_prime(&p);
    let x1: Uint = poly.a * Uint::from(sp.roots[0]) + poly.b;
    let x1p: u64 = (x1 % Uint::from(p.p)).to_u64().unwrap();
    assert_eq!(pow_mod(x1p, 2, p.p), pow_mod(p.r, 2, p.p));
    let x2: Uint = poly.a * Uint::from(sp.roots[1]) + poly.b;
    let x2p: u64 = (x2 % Uint::from(p.p)).to_u64().unwrap();
    assert_eq!(pow_mod(x2p, 2, p.p), pow_mod(p.r, 2, p.p));
}

/// Returns a polynomial suitable for sieving across ±2^sievebits
/// The offset is a seed for prime generation.
pub fn select_poly(base: Uint, offset: u64, n: Uint) -> Poly {
    // Select an appropriate pseudoprime. It is enough to be able
    // to compute a modular square root of n.
    let (d, r) = sieve_poly(base, offset, n);
    make_poly(d, r, &n)
}

pub fn select_polys(base: Uint, width: usize, n: &Uint) -> Vec<Poly> {
    sieve_for_polys(base, width, &n)
        .into_iter()
        .map(|(d, r)| make_poly(d, r, &n))
        .collect()
}

pub fn sieve_for_polys(bmin: Uint, width: usize, n: &Uint) -> Vec<(Uint, Uint)> {
    let mut composites = vec![false; width as usize];
    for &p in SMALL_PRIMES {
        let off = bmin % (p as u64);
        let mut idx = -(off as isize);
        while idx < composites.len() as isize {
            if idx >= 0 {
                composites[idx as usize] = true
            }
            idx += p as isize
        }
    }
    let base4 = bmin.low_u64() % 4;
    let mut result = vec![];
    for i in 0..width {
        if !composites[i] && (base4 + i as u64) % 4 == 3 {
            // No small factor, 3 mod 4
            let p = bmin + Uint::from(i);
            let r = pow_mod(*n, (p >> 2) + Uint::one(), p);
            if (r * r) % p == n % p {
                result.push((p, r));
            }
        }
    }
    result
}

fn make_poly(d: Uint, r: Uint, n: &Uint) -> Poly {
    // Lift square root mod D^2
    // Since D*D < N, computations can be done using the same integer width.
    let h1 = r;
    let c = ((n - h1 * h1) / d) % d;
    let h2 = (c * inv_mod(h1 << 1, d).unwrap()) % d;
    // (h1 + h2*D)**2 = n mod D^2
    let mut b = (h1 + h2 * d) % (d * d);

    // If kn = 1 mod 4:
    // A = D^2, B = sqrt(n) mod D^2, C = (B^2 - kn) / 4A
    // (2Ax + B)^2 - kn = 4A (Ax^2 + B x + C)
    // Ax^2 + Bx + C = ((2Ax + B)/2D)^2 mod n
    //
    // otherwise:
    // A = D^2, B = sqrt(n) mod D^2, C = (4B^2 - kn) / A
    // (Ax+2B)^2 - kn = A (Ax^2 + 2Bx + C)
    // Ax^2 + 2Bx + C = ((Ax+2B)/D)^2 mod n
    if n.low_u64() % 4 == 1 {
        // want an odd b
        if b.low_u64() % 2 == 0 {
            b = d * d - b;
        }
        Poly {
            a: d * d << 1,
            b: b,
            d: d << 1,
        }
    } else {
        // want even b
        if b.low_u64() % 2 == 1 {
            b = d * d - b;
        }
        Poly {
            a: d * d,
            b: b,
            d: d,
        }
    }
}

const SMALL_PRIMES: &[u64] = &[
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
];

pub const POLY_STRIDE: usize = 32768;

fn sieve_at(base: Uint, offset: u64) -> [u64; 128] {
    if base.bits() > 1024 {
        panic!("not expecting to sieve {}-bit primes", base.bits())
    }
    let mut composites = [0u8; POLY_STRIDE];
    for &p in SMALL_PRIMES {
        let off = (base + Uint::from(offset)) % p;
        let mut idx = -(off as isize);
        while idx < composites.len() as isize {
            if idx >= 0 {
                composites[idx as usize] = 1
            }
            idx += p as isize
        }
    }
    let mut res = [0u64; 128];
    let mut idx = 0;
    let base4 = base.low_u64() % 4;
    for i in 0..POLY_STRIDE {
        if composites[i] == 0 && (base4 + i as u64) % 4 == 3 {
            res[idx] = offset + i as u64;
            idx += 1;
        }
        if idx == 128 {
            break;
        }
    }
    res
}

fn sieve_poly(base: Uint, offset: u64, n: Uint) -> (Uint, Uint) {
    let offs = sieve_at(base, offset);
    for o in offs {
        if o == 0 {
            continue;
        }
        let base = base + Uint::from(o);
        if base.low_u64() % 4 == 3 {
            // Compute pow(n, (d+1)/4, d)
            let r = pow_mod(n, (base >> 2) + Uint::one(), base);
            if (r * r) % base == n % base {
                return (base, r);
            }
        }
    }
    panic!(
        "impossible! failed to find a pseudoprime {} {}=>{:?}",
        base, offset, offs
    )
}

#[test]
fn test_select_poly() {
    use crate::arith::{isqrt, sqrt_mod};
    use crate::Int;

    let n = Uint::from_str(
        "104567211693678450173299212092863908236097914668062065364632502155864426186497",
    )
    .unwrap();
    let mut polybase: Uint = isqrt(n >> 1) >> 24;
    polybase = isqrt(polybase);
    let Poly { a, b, d } = select_poly(polybase, 0, n);
    // D = 3 mod 4, 2D = 6 mod 8
    assert_eq!(d.low_u64() % 8, 6);
    // N is a square modulo D
    assert!(sqrt_mod(n, d >> 1).is_some());
    // A = D^2, 2A = (2D)^2
    assert_eq!(a << 1, d * d);
    // B^2 = N mod 4D^2
    assert_eq!(pow_mod(b, Uint::from(2u64), d * d), n % (d * d));
    println!("D={} A={} B={}", d, a, b);

    let c = (n - (b * b)) / (a << 2);

    // Check that:
    // Ax²+Bx+C is small
    // 4A(Ax²+Bx+C) = (2Ax+B)^2 mod N
    let x = Uint::from(1_234_567u64);
    let num = ((a << 1) * x + b) % n;
    let den = inv_mod(d << 1, n).unwrap();
    println!("1/2D={}", den);
    let q = (num * den) % n;
    println!("{}", q);
    let q2 = pow_mod(q, Uint::from(2u64), n);
    println!("{}", q2);
    let px = Int::from_bits(a * x * x + b * x) - Int::from_bits(c);
    println!("Ax²+Bx+C = {}", px);
    assert!(px.abs().bits() <= 128 + 24);
}
