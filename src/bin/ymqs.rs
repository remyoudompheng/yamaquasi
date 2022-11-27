// Bibliography:
//
// Carl Pomerance, A Tale of Two Sieves
// https://www.ams.org/notices/199612/pomerance.pdf
//
// J. Gerver, Factoring Large Numbers with a Quadratic Sieve
// https://www.jstor.org/stable/2007781
//
// https://en.wikipedia.org/wiki/Quadratic_sieve
//
use yamaquasi::poly::{self, Prime, SievePrime};
use yamaquasi::{isqrt, sqrt_mod, Uint, U1024};

const OPT_MULTIPLIERS: bool = true;
const OPT_SELECTPOLY: bool = false;
const OPT_MULTIPOLY: bool = false;

fn main() {
    let arg = std::env::args().nth(1).expect("Usage: ymqs NUMBER");
    let n = U1024::from_dec_str(&arg).expect("could not read decimal number");
    const MAXBITS: usize = 2 * (256 - 30);
    if n.bits() > MAXBITS {
        panic!(
            "Number size ({} bits) exceeds {} bits limit",
            n.bits(),
            MAXBITS
        )
    }
    let n = Uint::from_dec_str(&arg).unwrap();
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
            let r = sqrt_mod(n * k, p)?;
            Some(Prime {
                p: p as u64,
                r: r as u64,
            })
        })
        .collect();
    let smallprimes: Vec<u64> = primes.iter().map(|f| f.p).take(10).collect();
    eprintln!("Factor base size {} ({:?})", primes.len(), smallprimes);
    // Prepare sieve
    let nsqrt = isqrt(n * k);
    let sprimes: Vec<SievePrime> = primes
        .iter()
        .map(|&p| poly::simple_prime(p, nsqrt))
        .collect();
    sieve(n * k, sprimes)
}

fn smooth_bound(n: Uint) -> u32 {
    // x => sqrt(2)/4 * sqrt(x * log(2) * log(x * log(2)))
    let bits = n.bits() / 2 + 30;
    let bits_f = bits as f64;
    let mut sqrt: f64 = bits_f / 8.;
    sqrt = sqrt / 2. + bits_f / (2. * sqrt);
    sqrt = sqrt / 2. + bits_f / (2. * sqrt);
    (1.08 * sqrt).exp() as u32
}

const BLOCK_SIZE: usize = 4 * 1024 * 1024;

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

fn sieve(n: Uint, primes: Vec<SievePrime>) {
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
            &primes,
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

struct Relation {
    x: Uint,
    x2: Uint,                     // x*x mod n
    factors: Vec<(isize, usize)>, // -1 for the sign
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
            let candidate = if b.offset < 0 {
                let s = b.nsqrt - (-b.offset - i as i64);
                b.n - s * s
            } else {
                let s = b.nsqrt + b.offset + i;
                s * s - b.n
            };
            let mut cofactor = candidate;
            for item in primes {
                if b.matches(i, item) {
                    loop {
                        let (q, r) = cofactor.div_mod(Uint::from(item.p));
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
