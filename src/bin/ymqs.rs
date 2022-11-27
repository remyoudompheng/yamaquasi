//! Bibliography:
//!
//! Carl Pomerance, A Tale of Two Sieves
//! https://www.ams.org/notices/199612/pomerance.pdf
//!
//! J. Gerver, Factoring Large Numbers with a Quadratic Sieve
//! https://www.jstor.org/stable/2007781
//!
//! https://en.wikipedia.org/wiki/Quadratic_sieve

use std::str::FromStr;

use yamaquasi::arith::{isqrt, sqrt_mod, inv_mod, Num, U1024};
use yamaquasi::poly::{self, Poly, Prime, SievePrime, select_polys, POLY_STRIDE};
use yamaquasi::{Int, Uint};
use yamaquasi::params::{BLOCK_SIZE, smooth_bound, sieve_interval_logsize};

use num_integer::div_rem;

const OPT_MULTIPLIERS: bool = true;
const OPT_MULTIPOLY: bool = true;

fn main() {
    let arg = std::env::args().nth(1).expect("Usage: ymqs NUMBER");
    let n = U1024::from_str(&arg).expect("could not read decimal number");
    const MAXBITS: u32 = 2 * (256 - 30);
    if n.bits() > MAXBITS {
        panic!(
            "Number size ({} bits) exceeds {} bits limit",
            n.bits(),
            MAXBITS
        )
    }
    let n = Uint::from_str(&arg).unwrap();
    eprintln!("Input number {}", n);
    let ks = poly::select_multipliers(n);
    eprintln!("Multipliers {:?}", ks);
    let b = smooth_bound(n);
    eprintln!("Smoothness bound {}", b);
    let primes = compute_primes(b);
    eprintln!("All primes {}", primes.len());
    let k: u32 = if OPT_MULTIPLIERS {
        *ks.first().unwrap()
    } else {
        1
    };
    // Prepare factor base
    let primes: Vec<Prime> = primes
        .into_iter()
        .filter_map(|p| {
            let nk: Uint = (n * Uint::from(k)) % Uint::from(p);
            let r = sqrt_mod(nk.low_u64(), p as u64)?;
            Some(Prime { p: p as u64, r: r })
        })
        .collect();
    let smallprimes: Vec<u64> = primes.iter().map(|f| f.p).take(10).collect();
    eprintln!("Factor base size {} ({:?})", primes.len(), smallprimes);

    if !OPT_MULTIPOLY {
        sieve(n * Uint::from(k), &primes)
    } else {
        let mut found = 0usize ;
        let mlog = sieve_interval_logsize(n);
        println!("Sieving interval size {}M", 1<<(mlog - 20));
        for idx in 0u64..1000 {
            found += sieve_poly(idx, n * Uint::from(k), &primes);
            println!(
                "Sieved {}M {} polys found {} smooths",
                ((idx + 1) << (mlog+1 - 10)) >> 10,
                idx +1 ,
                found
            );

        }
    }
}

fn compute_primes(bound: u32) -> Vec<u32> {
    // Eratosthenes
    let mut sieve = vec![0; bound as usize];
    let mut p = 2;
    while p * p < bound {
        let mut k = 2 * p;
        while k < bound {
            sieve[k as usize] = 1;
            k += p
        }
        p += 1
    }

    let mut primes = vec![];
    for p in 2..sieve.len() {
        if sieve[p] == 0 {
            primes.push(p as u32)
        }
    }
    primes
}

fn sieve(n: Uint, primes: &[Prime]) {
    // Prepare sieve
    let nsqrt = isqrt(n);
    let sprimes: Vec<SievePrime> = primes
        .iter()
        .map(|&p| poly::simple_prime(p, nsqrt))
        .collect();

    // Naïve quadratic sieve with polynomial x²-n (x=-M..M)
    // Max value is X = sqrt(n) * M
    // Smooth bound Y = exp(1/2 sqrt(log X log log X))
    // If input number is 512 bits, generated values are less then 300 bits

    // There are at most 10 prime factors => 8-bit precision is enough
    // Blocks optimized to fit 512kB cache memory per core
    // Work unit is 16 blocks

    let mut found: usize = 0;
    let nsqrt = isqrt(n);
    for i in 0..(16 * 1024 * 1024 / (BLOCK_SIZE / 1024)) {
        let isign: i64 = if i % 2 == 0 {
            i as i64 / 2
        } else {
            -1 - (i as i64) / 2
        };
        let x = sieve_block(
            Block {
                offset: isign * BLOCK_SIZE as i64,
                n: n,
                nsqrt: nsqrt,
            },
            &sprimes,
        );
        found += x;
        if i % 4 == 3 {
            println!(
                "Sieved {}M found {} smooths",
                (i + 1) * BLOCK_SIZE / 1024 / 1024,
                found
            );
        }
    }
}

struct Block {
    offset: i64,
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

fn sieve_block(b: Block, primes: &[SievePrime]) -> usize {
    println!("block offset {}", b.offset);
    let mut blk = vec![0u8; BLOCK_SIZE];
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
    const EXTRABITS: u8 = 8;
    let target = b.nsqrt.bits() as u8 + EXTRABITS;
    let mut found = 0usize;
    for i in 0..BLOCK_SIZE {
        if blk[i] >= target {
            let s = Int::from_bits(b.nsqrt) + Int::from(b.offset + (i as i64));
            let candidate = s * s - Int::from_bits(b.n);
            let mut cofactor: Uint = candidate.abs().to_bits();
            for item in primes {
                if b.matches(i, item) {
                    loop {
                        let (q, r) = div_rem(cofactor, Uint::from(item.p));
                        if r.is_zero() {
                            cofactor = q
                        } else {
                            break;
                        }
                    }
                }
            }
            if cofactor.bits() < 64 && cofactor.low_u64() < maxprime {
                println!("i={} smooth {}", i, candidate);
                found += 1;
            } else {
                println!("i={} smooth {} cofactor {}", i, candidate, cofactor);
            }
        }
    }
    found
}

// MPQS implementation

// One MPQS unit of work, identified by an integer 'idx'.
fn sieve_poly(idx: u64, n: Uint, primes: &[Prime]) -> usize {
    let mlog = sieve_interval_logsize(n);
    // a suitable prime will be found (hopefully) in
    // a size (2 log n) interval
    let pol = select_polys(mlog as usize, idx * POLY_STRIDE as u64, n);
    let nblocks = (1u64 << (mlog-10)) / (BLOCK_SIZE as u64 / 1024);
    println!("Sieving polynomial A={} B={} M=2^{} blocks={}", pol.0, pol.1, mlog,nblocks);
    // Precompute inverse of 4A
    let ainv = inv_mod(pol.0 << 2, n).unwrap();

    // Precompute factor base extra information.
    let sprimes: Vec<_> = primes.into_iter().map(|&p| pol.prepare_prime(p)).collect();

    let mut found: usize = 0;
    let nsqrt = isqrt(n);
    // Sieve from -M to M
    let nblocks = nblocks as i64;
    for i in -nblocks..nblocks {
        let x = sieve_block_poly(
            Block {
                offset: i * BLOCK_SIZE as i64,
                n: n,
                nsqrt: nsqrt,
            },
            &sprimes,
            &pol,
            ainv
        );
        found += x;
    }
    found
}

// Sieve using a selected polynomial
fn sieve_block_poly(b: Block, primes: &[SievePrime], pol: &Poly, ainv: Uint) -> usize {
    let mut blk = vec![0u8; BLOCK_SIZE];
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
    const EXTRABITS: u8 = 8;
    let target = b.nsqrt.bits() as u8 + EXTRABITS;
    let mut found = 0usize;
    for i in 0..BLOCK_SIZE {
        if blk[i] >= target {
            // Evaluate polynomial
            let s = Int::from_bits(pol.0) * Int::from(b.offset + (i as i64)) << 1;
            let s = s + Int::from_bits(pol.1);
            let candidate: Int = s * s - Int::from_bits(b.n);
            let cabs = (candidate.abs().to_bits() * ainv) % b.n;
            let sign = candidate.is_negative();
            let mut cofactor: Uint = cabs;
            for item in primes {
                if b.matches(i, item) {
                    loop {
                        let (q, r) = div_rem(cofactor, Uint::from(item.p));
                        if r.is_zero() {
                            cofactor = q
                        } else {
                            break;
                        }
                    }
                }
            }
            if cofactor.bits() < 64 && cofactor.low_u64() < maxprime {
                println!("i={} smooth {}", i, cabs);
                found += 1;
            } else {
                println!("i={} smooth {} cofactor {}", i, cabs, cofactor);
            }
        }
    }
    found
}

