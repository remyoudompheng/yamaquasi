use crate::Uint;

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
    let shift = p.p - (offset % p.p).low_u64();
    SievePrime {
        p: p.p,
        r: p.r,
        roots: [(p.r + shift) % p.p, (p.p - p.r + shift) % p.p],
    }
}

#[test]
fn test_simple_prime() {
    let p = Prime { p: 10223, r: 4526 };
    let nsqrt = Uint::from_dec_str("13697025762053691031049747437678526773503028576").unwrap();
    let sp = simple_prime(p, nsqrt);
    let rr = nsqrt + sp.roots[0];
    assert_eq!(((rr * rr) % p.p).low_u64(), (p.r * p.r) % p.p);
    let rr = nsqrt + sp.roots[1];
    assert_eq!(((rr * rr) % p.p).low_u64(), (p.r * p.r) % p.p);
}

/// Selects k such kn is a quadratic residue modulo many small primes.
pub fn select_multipliers(n: Uint) -> Vec<u32> {
    let mut res = vec![];
    let bases: &[u32] = &[8, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31];
    let residues: Vec<u32> = bases.iter().map(|&b| (n % b).low_u32()).collect();
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
