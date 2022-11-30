//! Bibliography:
//!
//! Carl Pomerance, A Tale of Two Sieves
//! https://www.ams.org/notices/199612/pomerance.pdf
//!
//! J. Gerver, Factoring Large Numbers with a Quadratic Sieve
//! https://www.jstor.org/stable/2007781
//!
//! https://en.wikipedia.org/wiki/Quadratic_sieve

use std::collections::HashMap;
use std::str::FromStr;

use yamaquasi::arith::{inv_mod, isqrt, sqrt_mod, Num, U1024};
use yamaquasi::params::{self, large_prime_factor, BLOCK_SIZE};
use yamaquasi::poly::{self, Poly, Prime, SievePrime};
use yamaquasi::relations::{combine_large_relation, final_step, relation_gap, Relation};
use yamaquasi::{Int, Uint};

use num_integer::div_rem;

const OPT_MULTIPLIERS: bool = true;

fn main() {
    let arg = arguments::parse(std::env::args()).unwrap();
    if arg.orphans.len() != 1 {
        println!("Usage: ymqs [--mode qs|mpqs] NUMBER");
    }
    let mode = arg.get::<String>("mode").unwrap_or("mpqs".into());
    let number = &arg.orphans[0];
    let n = U1024::from_str(number).expect("could not read decimal number");
    const MAXBITS: u32 = 2 * (256 - 30);
    if n.bits() > MAXBITS {
        panic!(
            "Number size ({} bits) exceeds {} bits limit",
            n.bits(),
            MAXBITS
        )
    }
    let n = Uint::from_str(number).unwrap();
    eprintln!("Input number {}", n);
    let ks = poly::select_multipliers(n);
    let k: u32 = if OPT_MULTIPLIERS { ks } else { 1 };
    eprintln!("Multiplier {}", k);
    // Choose factor base. Sieve twice the number of primes
    // (n will be a quadratic residue for only half of them)
    let b = params::factor_base_size(n);
    let primes = poly::primes(std::cmp::max(2 * b, 1000));
    eprintln!("Testing small prime divisors");
    let mut n = n;
    for &p in &primes {
        while n % (p as u64) == 0 {
            n /= Uint::from(p);
            println!("Factor {}", p)
        }
    }
    let primes = &primes[..2 * b as usize];
    eprintln!("Smoothness bound {}", primes.last().unwrap());
    eprintln!("All primes {}", primes.len());
    // Prepare factor base
    let primes: Vec<Prime> = primes
        .into_iter()
        .filter_map(|&p| {
            let nk: u64 = (n * Uint::from(k)) % (p as u64);
            let r = sqrt_mod(nk, p as u64)?;
            Some(Prime { p: p as u64, r: r })
        })
        .collect();
    let smallprimes: Vec<u64> = primes.iter().map(|f| f.p).take(10).collect();
    eprintln!("Factor base size {} ({:?})", primes.len(), smallprimes);

    if &mode == "qs" {
        sieve(n * Uint::from(k), &primes)
    } else {
        let mut target = primes.len() * 8 / 10;
        let mut relations = vec![];
        let mut larges = HashMap::<u64, Relation>::new();
        let mut extras = 0;
        let mlog = params::mpqs_interval_logsize(&n);
        if mlog >= 20 {
            println!("Sieving interval size {}M", 1 << (mlog - 20));
        } else {
            println!("Sieving interval size {}k", 1 << (mlog - 10));
        }
        // Precompute starting point for polynomials
        // See [Silverman, Section 3]
        // Look for A=D^2 such that (2A M/2)^2 ~= N/2
        // D is less than 64-bit for a 256-bit n
        let mut polybase: Uint = isqrt(n >> 1) >> mlog;
        polybase = isqrt(polybase);
        let maxprime = primes.last().unwrap().p;
        if polybase < Uint::from(maxprime) {
            polybase = Uint::from(maxprime + 1000);
        }
        let polystride = if n.bits() < 100 {
            // Block of 20 polynomials
            20 * 20 / 7 * polybase.bits()
        } else {
            100 * 20 / 7 * polybase.bits()
        };
        let nk: Uint = n * Uint::from(k);
        let mut polys = poly::select_polys(polybase, polystride as usize, &nk);
        let mut polyidx = 0;
        println!("Generated {} polynomials", polys.len());
        let maxlarge: u64 = maxprime * large_prime_factor(&n);
        println!("Max cofactor {}", maxlarge);
        for idx in 0u64..1_000_000 {
            // Pop next polynomial.
            if polyidx == polys.len() {
                polybase += Uint::from(polystride);
                polys = poly::select_polys(polybase, polystride as usize, &nk);
                polyidx = 0;
                println!("Generated {} polynomials", polys.len());
            }
            let pol = &polys[polyidx];
            polyidx += 1;
            let (mut found, foundlarge) = sieve_poly(pol, nk, &primes);
            relations.append(&mut found);
            for r in foundlarge {
                if let Some(rr) = combine_large_relation(&mut larges, &r, &n) {
                    relations.push(rr);
                    extras += 1;
                }
            }
            println!(
                "Sieved {}M {} polys found {} smooths ({} using cofactors)",
                ((idx + 1) << (mlog + 1 - 10)) >> 10,
                idx + 1,
                relations.len(),
                extras,
            );
            if relations.len() >= target {
                let gap = relation_gap(n, &relations);
                if gap == 0 {
                    println!("Found enough relations");
                    break;
                } else {
                    println!("Need {} additional relations", gap);
                    target += gap + 10;
                }
            }
        }
        final_step(n, &relations);
    }
}

fn sieve(n: Uint, primes: &[Prime]) {
    // Prepare sieve
    let nsqrt = isqrt(n);
    let sprimes: Vec<SievePrime> = primes
        .iter()
        .map(|&p| poly::simple_prime(p, nsqrt))
        .collect();
    let maxlarge: u64 = primes.last().unwrap().p * large_prime_factor(&n);
    println!("Max cofactor {}", maxlarge);

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
    loop {
        let isign: i64 = if i % 2 == 0 {
            i as i64 / 2
        } else {
            -1 - (i as i64) / 2
        };
        let (mut found, foundlarge) = sieve_block(
            Block {
                offset: isign * block_size as i64,
                len: block_size,
                n: n,
                nsqrt: nsqrt,
            },
            &sprimes,
        );
        relations.append(&mut found);
        for r in foundlarge {
            if let Some(rr) = combine_large_relation(&mut larges, &r, &n) {
                relations.push(rr);
                extras += 1;
            }
        }
        println!(
            "Sieved {}M found {} smooths ({} using cofactors)",
            (i + 1) * BLOCK_SIZE / 1024 / 1024,
            relations.len(),
            extras,
        );
        // For small n the sieve must stop quickly:
        // test whether we already have enough relations.
        if n.bits() < 64 || relations.len() >= target {
            let gap = relation_gap(n, &relations);
            if gap == 0 {
                println!("Found enough relations");
                break;
            } else {
                println!("Need {} additional relations", gap);
                target = relations.len() + gap + 10;
            }
        }
        i += 1;
    }
    final_step(n, &relations);
}

struct Block {
    offset: i64,
    len: usize,
    n: Uint,
    nsqrt: Uint,
}

impl Block {
    // Returns smallest i,j such that:
    // nsqrt + offset + {i,j} are roots modulo p
    fn starts(&self, pr: &SievePrime) -> [u64; 2] {
        let p = pr.p;
        // offset modulo p
        let off: u64 = if self.offset < 0 {
            p - (-self.offset as u64) % p
        } else {
            self.offset as u64 % p
        };
        let [r1, r2] = pr.roots;
        [
            if r1 < off { r1 + p - off } else { r1 - off },
            if r2 < off { r2 + p - off } else { r2 - off },
        ]
    }

    fn matches(&self, i: usize, pr: &SievePrime) -> bool {
        let [a, b] = self.starts(pr);
        i as u64 % pr.p == a || i as u64 % pr.p == b
    }
}

fn sieve_block(b: Block, primes: &[SievePrime]) -> (Vec<Relation>, Vec<Relation>) {
    let mut blk = vec![0u8; b.len];
    for item in primes {
        let p = item.p;
        let size = u64::BITS - u64::leading_zeros(p);
        let starts = b.starts(&item);
        let starts = if p == 2 { &starts[..1] } else { &starts[..] };
        for &s in starts {
            let mut i = s as usize;
            while i < blk.len() {
                blk[i] += size as u8;
                i += p as usize;
            }
        }
    }

    let maxprime = primes.last().unwrap().p;
    let maxlarge = maxprime * large_prime_factor(&b.n);
    let mut result = vec![];
    let mut extras = vec![];
    const EXTRABITS: u32 = 8;
    let target = b.nsqrt.bits() + EXTRABITS - maxlarge.bits();
    for i in 0..b.len {
        if blk[i] as u32 >= target {
            let mut factors: Vec<(i64, u64)> = Vec::with_capacity(20);
            let s = Int::from_bits(b.nsqrt) + Int::from(b.offset + (i as i64));
            let candidate: Int = s * s - Int::from_bits(b.n);
            let cabs = candidate.abs().to_bits() % b.n;
            if candidate.is_negative() {
                factors.push((-1, 1));
            }
            let mut cofactor: Uint = cabs;
            for item in primes {
                if b.matches(i, item) {
                    let mut exp = 0;
                    loop {
                        let (q, r) = div_rem(cofactor, Uint::from(item.p));
                        if r.is_zero() {
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
                let sabs = (s.abs().to_bits()) % b.n;
                result.push(Relation {
                    x: if candidate.is_negative() {
                        b.n - sabs
                    } else {
                        sabs
                    },
                    cofactor: 1,
                    factors,
                });
            } else if cofactor < maxlarge {
                //println!("i={} smooth {} cofactor {}", i, cabs, cofactor);
                let sabs = (s.abs().to_bits()) % b.n;
                extras.push(Relation {
                    x: if candidate.is_negative() {
                        b.n - sabs
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

// MPQS implementation

// One MPQS unit of work, identified by an integer 'idx'.
fn sieve_poly(pol: &Poly, n: Uint, primes: &[Prime]) -> (Vec<Relation>, Vec<Relation>) {
    let mlog = params::mpqs_interval_logsize(&n);
    let nblocks = (1u64 << (mlog - 10)) / (BLOCK_SIZE as u64 / 1024);
    println!(
        "Sieving polynomial A={} B={} M=2^{} blocks={}",
        pol.a, pol.b, mlog, nblocks
    );
    // Precompute inverse of 4A
    let ainv = inv_mod(pol.a << 2, n).unwrap();
    // Precompute inverse of 2D
    let dinv = inv_mod(pol.d << 1, n).unwrap();

    // Precompute factor base extra information.
    let sprimes: Vec<_> = primes.into_iter().map(|&p| pol.prepare_prime(p)).collect();

    // Sieve from -M to M
    let nsqrt = isqrt(n);
    if nblocks == 0 {
        return sieve_block_poly(
            Block {
                offset: -(1 << mlog),
                len: 2 << mlog,
                n: n,
                nsqrt: nsqrt,
            },
            &sprimes,
            &pol,
            ainv,
            dinv,
        );
    }
    let mut result: Vec<Relation> = vec![];
    let mut extras: Vec<Relation> = vec![];
    let nblocks = nblocks as i64;
    for i in -nblocks..nblocks {
        let (mut x, mut y) = sieve_block_poly(
            Block {
                offset: i * BLOCK_SIZE as i64,
                len: BLOCK_SIZE,
                n: n,
                nsqrt: nsqrt,
            },
            &sprimes,
            &pol,
            ainv,
            dinv,
        );
        result.append(&mut x);
        extras.append(&mut y);
    }
    (result, extras)
}

// Sieve using a selected polynomial
fn sieve_block_poly(
    b: Block,
    primes: &[SievePrime],
    pol: &Poly,
    ainv: Uint,
    dinv: Uint,
) -> (Vec<Relation>, Vec<Relation>) {
    let mut blk = vec![0u8; b.len];
    for item in primes {
        let size = (u64::BITS - u64::leading_zeros(item.p)) as u8;
        let p = item.p as usize;
        let [s1, s2] = &b.starts(&item);
        if p == 2 {
            let mut i = *s1 as usize;
            while i < blk.len() {
                blk[i] += 1;
                i += 2;
            }
        } else {
            let (s1, s2) = (*s1 as usize, *s2 as usize);
            let mut ip: usize = 0;
            while ip < blk.len() {
                if ip + s1 < blk.len() {
                    blk[ip + s1] += size;
                }
                if ip + s2 < blk.len() {
                    blk[ip + s2] += size;
                }
                ip += p;
            }
        }
    }

    let maxprime = primes.last().unwrap().p;
    let maxlarge = maxprime * large_prime_factor(&b.n);
    let mut result = vec![];
    let mut extras = vec![];

    const EXTRABITS: u32 = 8;
    let target = b.nsqrt.bits() + EXTRABITS - maxlarge.bits() / 3;
    for i in 0..b.len {
        if blk[i] as u32 >= target {
            let mut factors: Vec<(i64, u64)> = Vec::with_capacity(20);
            // Evaluate polynomial
            let s: Int = Int::from_bits(pol.a) * Int::from(b.offset + (i as i64)) << 1;
            let s = s + Int::from_bits(pol.b);
            let candidate: Int = s * s - Int::from_bits(b.n);
            let cabs = (candidate.abs().to_bits() * ainv) % b.n;
            if candidate.is_negative() {
                factors.push((-1, 1));
            }
            let mut cofactor: Uint = cabs;
            for item in primes {
                if b.matches(i, item) {
                    let mut exp = 0;
                    loop {
                        let (q, r) = div_rem(cofactor, Uint::from(item.p));
                        if r.is_zero() {
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
            if cofactor > maxlarge {
                continue;
            }
            if cofactor == 1 {
                //println!("i={} smooth {}", i, cabs);
                let sabs = (s.abs().to_bits() * dinv) % b.n;
                result.push(Relation {
                    x: if candidate.is_negative() {
                        b.n - sabs
                    } else {
                        sabs
                    },
                    cofactor: 1,
                    factors,
                });
            } else {
                //println!("i={} smooth {} cofactor {}", i, cabs, cofactor);
                let sabs = (s.abs().to_bits() * dinv) % b.n;
                extras.push(Relation {
                    x: if candidate.is_negative() {
                        b.n - sabs
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
