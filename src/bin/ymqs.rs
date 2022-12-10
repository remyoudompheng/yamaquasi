// Copyright 2022 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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

use yamaquasi::arith::U1024;
use yamaquasi::fbase::{self, prepare_factor_base, Prime};
use yamaquasi::relations::final_step;
use yamaquasi::Uint;
use yamaquasi::{mpqs, params, qsieve, siqs};

fn main() {
    let arg = arguments::parse(std::env::args()).unwrap();
    if arg.orphans.len() != 1 {
        println!("Usage: ymqs [--mode qs|mpqs] [--threads N] NUMBER");
    }
    let mode = arg.get::<String>("mode").unwrap_or("mpqs".into());
    let threads = arg.get::<usize>("threads");
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
    let (k, score) = fbase::select_multiplier(n);
    eprintln!("Selected multiplier {} (score {:.2}/8)", k, score);
    // Choose factor base. Sieve twice the number of primes
    // (n will be a quadratic residue for only half of them)
    let fb = params::factor_base_size(&n);
    let primes = fbase::primes(std::cmp::max(2 * fb, 1000));
    eprintln!("Testing small prime divisors");
    let mut n = n;
    for &p in &primes {
        while n % (p as u64) == 0 {
            n /= Uint::from(p);
            eprintln!("Found small factor");
            println!("{}", p);
        }
    }
    if n.is_one() {
        return;
    }
    let primes = &primes[..2 * fb as usize];
    eprintln!("Smoothness bound {}", primes.last().unwrap());
    eprintln!("All primes {}", primes.len());
    // Prepare factor base
    let nk = n * Uint::from(k);
    let primes: Vec<Prime> = prepare_factor_base(&nk, primes);
    let smallprimes: Vec<u64> = primes.iter().map(|f| f.p).take(10).collect();
    eprintln!("Factor base size {} ({:?})", primes.len(), smallprimes);

    let rels = match &mode[..] {
        "qs" => qsieve::qsieve(nk, &primes),
        "mpqs" => mpqs::mpqs(nk, &primes, threads),
        "siqs" => siqs::siqs(&nk, &primes),
        _ => {
            eprintln!("Invalid operation mode {:?}", mode);
            return;
        }
    };
    final_step(&n, &rels);
}
