//! The classical quadratic sieve (using polynomial (x + Nsqrt)^2 - N).

use std::collections::HashMap;

use crate::arith::{isqrt, Num};
use crate::fbase::{Prime, SievePrime};
use crate::params::{self, large_prime_factor, BLOCK_SIZE};
use crate::relations::{combine_large_relation, relation_gap, Relation};
use crate::{Int, Uint};

pub fn qsieve(n: Uint, primes: &[Prime]) -> Vec<Relation> {
    // Prepare sieve
    let nsqrt = isqrt(n);
    let sprimes: Vec<SievePrime> = primes.iter().map(|p| simple_prime(p, nsqrt)).collect();
    let maxlarge: u64 = primes.last().unwrap().p * large_prime_factor(&n);
    eprintln!("Max cofactor {}", maxlarge);

    // Naïve quadratic sieve with polynomial x²-n (x=-M..M)
    // Max value is X = sqrt(n) * M
    // Smooth bound Y = exp(1/2 sqrt(log X log log X))
    // If input number is 512 bits, generated values are less then 300 bits

    // There are at most 10 prime factors => 8-bit precision is enough
    // Blocks optimized to fit 512kB cache memory per core
    // Work unit is 16 blocks

    let mut relations = vec![];
    let mut target = primes.len() * 8 / 10;
    let nsqrt = isqrt(n);
    let mut i = 0;
    let mut larges = HashMap::<u64, Relation>::new();
    let mut extras = 0;
    let block_size = params::qs_blocksize(&n);
    let sieve = SieveQS {
        n,
        nsqrt,
        primes,
        sprimes: &sprimes,
    };
    loop {
        let isign: i64 = if i % 2 == 0 {
            i as i64 / 2
        } else {
            -1 - (i as i64) / 2
        };
        let (mut found, foundlarge) = sieve_block(&sieve, isign * block_size as i64, block_size);
        relations.append(&mut found);
        for r in foundlarge {
            if let Some(rr) = combine_large_relation(&mut larges, &r, &n) {
                relations.push(rr);
                extras += 1;
            }
        }
        if (i + 1) % (1024 * 1024 / BLOCK_SIZE) == 0 {
            println!(
                "Sieved {}M found {} smooths ({} using cofactors)",
                (i + 1) * BLOCK_SIZE / 1024 / 1024,
                relations.len(),
                extras,
            );
        }
        // For small n the sieve must stop quickly:
        // test whether we already have enough relations.
        if n.bits() < 64 || relations.len() >= target {
            let gap = relation_gap(n, &relations);
            if gap == 0 {
                println!("Found enough relations");
                i += 1;
                break;
            } else {
                println!("Need {} additional relations", gap);
                target = relations.len() + gap + 10;
            }
        }
        i += 1;
    }
    println!(
        "Sieved {}{} found {} smooths ({} using cofactors)",
        if i * BLOCK_SIZE > 1024 * 1024 {
            i * BLOCK_SIZE / 1024 / 1024
        } else {
            i * BLOCK_SIZE / 1024
        },
        if i * BLOCK_SIZE > 1024 * 1024 {
            "M"
        } else {
            "k"
        },
        relations.len(),
        extras,
    );
    relations
}

struct SieveQS<'a> {
    n: Uint,
    nsqrt: Uint,
    primes: &'a [Prime],
    sprimes: &'a [SievePrime],
}

/// Compute sieving parameters for polynomial (X+offset)^2 - n
fn simple_prime(p: &Prime, offset: Uint) -> SievePrime {
    let shift: u64 = p.p - (offset % Uint::from(p.p)).to_u64().unwrap();
    SievePrime {
        p: p.p,
        roots: [(p.r + shift) % p.p, (p.p - p.r + shift) % p.p],
    }
}

// Compute sieving offsets i,j such that offset+i,j are roots
fn sieve_starts(p: &Prime, sp: &SievePrime, offset: i64) -> [u64; 2] {
    let [r1, r2] = sp.roots;
    let off: u64 = if offset < 0 {
        sp.p - p.div.divmod64((-offset) as u64).1
    } else {
        p.div.divmod64(offset as u64).1
    };
    [
        if r1 < off { r1 + p.p - off } else { r1 - off },
        if r2 < off { r2 + p.p - off } else { r2 - off },
    ]
}

#[test]
fn test_simple_prime() {
    use crate::arith;
    use std::str::FromStr;

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

fn sieve_block(s: &SieveQS, offset: i64, len: usize) -> (Vec<Relation>, Vec<Relation>) {
    let mut blk = vec![0u8; len];
    let mut starts = vec![[0u64, 0u64]; s.primes.len()];
    for (idx, (p, sp)) in s.primes.iter().zip(s.sprimes).enumerate() {
        starts[idx] = sieve_starts(p, sp, offset);
    }
    for (idx, item) in s.sprimes.iter().enumerate() {
        let size = (u64::BITS - u64::leading_zeros(item.p)) as u8;
        let p = item.p as usize;
        let [s1, s2] = starts[idx];
        if p == 2 {
            let mut i = s1 as usize;
            while i < blk.len() {
                blk[i] += 1;
                i += 2;
            }
        } else {
            let (s1, s2) = (s1 as usize, s2 as usize);
            let (s1, s2) = if s1 < s2 { (s1, s2) } else { (s2, s1) };
            let mut ip: usize = 0;
            while ip + s2 < blk.len() {
                blk[ip + s1] += size;
                blk[ip + s2] += size;
                ip += p;
            }
            if ip + s1 < blk.len() {
                blk[ip + s1] += size;
            }
            if ip + s2 < blk.len() {
                blk[ip + s2] += size;
            }
        }
    }

    let maxprime = s.primes.last().unwrap().p;
    let maxlarge = maxprime * large_prime_factor(&s.n);
    let mut result = vec![];
    let mut extras = vec![];
    let magnitude = u64::BITS - u64::leading_zeros(std::cmp::max(offset.abs() as u64, len as u64));
    let target = s.n.bits() / 2 + magnitude - maxlarge.bits();
    let (n, nsqrt) = (&s.n, &s.nsqrt);
    for i in 0..len {
        if blk[i] as u32 >= target {
            let mut factors: Vec<(i64, u64)> = Vec::with_capacity(20);
            let x = Int::from_bits(*nsqrt) + Int::from(offset + (i as i64));
            let candidate: Int = x * x - Int::from_bits(*n);
            let cabs = candidate.abs().to_bits() % n;
            if candidate.is_negative() {
                factors.push((-1, 1));
            }
            let mut cofactor: Uint = cabs;
            for (idx, item) in s.primes.iter().enumerate() {
                if starts[idx].contains(&item.div.divmod64(i as u64).1) {
                    let mut exp = 0;
                    loop {
                        let (q, r) = item.div.divmod_uint(&cofactor);
                        if r == 0 {
                            cofactor = q;
                            exp += 1;
                        } else {
                            break;
                        }
                    }
                    factors.push((item.p as i64, exp));
                }
            }
            let Some(cofactor) = cofactor.to_u64() else { continue };
            if cofactor == 1 {
                //println!("i={} smooth {}", i, cabs);
                let sabs = (x.abs().to_bits()) % n;
                result.push(Relation {
                    x: if candidate.is_negative() {
                        n - sabs
                    } else {
                        sabs
                    },
                    cofactor: 1,
                    factors,
                });
            } else if cofactor < maxlarge {
                //println!("i={} smooth {} cofactor {}", i, cabs, cofactor);
                let sabs = (x.abs().to_bits()) % n;
                extras.push(Relation {
                    x: if candidate.is_negative() {
                        n - sabs
                    } else {
                        sabs
                    },
                    cofactor: cofactor,
                    factors,
                });
            }
        }
    }
    (result, extras)
}
