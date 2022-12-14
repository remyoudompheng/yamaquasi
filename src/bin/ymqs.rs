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

use yamaquasi::arith::{Num, U1024};
use yamaquasi::fbase;
use yamaquasi::relations::final_step;
use yamaquasi::Uint;
use yamaquasi::{mpqs, qsieve, qsieve64, siqs};

fn main() {
    let arg = arguments::parse(std::env::args()).unwrap();
    if arg.get::<bool>("help").is_some() || arg.orphans.len() != 1 {
        eprintln!("Usage: ymqs [OPTIONS] NUMBER");
        eprintln!("");
        eprintln!("Options:");
        eprintln!("  --help                    show this help");
        eprintln!("  --mode qs|qs64|mpqs|siqs: force algorithm selection");
        eprintln!("  --threads N:              enable up to N computation threads");
        eprintln!("  --fb F:                   override automatic factor base size");
        return;
    }
    let mode = arg.get::<String>("mode").unwrap_or("mpqs".into());
    let threads = arg.get::<usize>("threads");
    let fb_user = arg.get::<u32>("fb");
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
    if mode == "qs64" {
        assert!(n.bits() <= 64);
        if let Some((a, b)) = qsieve64::qsieve(n.low_u64()) {
            println!("{}", a);
            println!("{}", b);
        }
        return;
    }
    let (k, score) = fbase::select_multiplier(n);
    eprintln!("Selected multiplier {} (score {:.2}/8)", k, score);
    eprintln!("Testing small prime divisors");
    let mut n = n;
    for &p in fbase::SMALL_PRIMES {
        while n % (p as u64) == 0 {
            n /= Uint::from(p);
            eprintln!("Found small factor");
            println!("{}", p);
        }
    }
    if n.is_one() {
        return;
    }
    // Prepare factor base
    let nk = n * Uint::from(k);

    let tpool: Option<rayon::ThreadPool> = threads.map(|t| {
        eprintln!("Using a pool of {} threads", t);
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build()
            .expect("cannot create thread pool")
    });
    let tpool = tpool.as_ref();

    let rels = match &mode[..] {
        "qs" => qsieve::qsieve(nk, fb_user, tpool),
        "mpqs" => mpqs::mpqs(nk, fb_user, tpool),
        "siqs" => siqs::siqs(&nk, fb_user, tpool),
        _ => {
            eprintln!("Invalid operation mode {:?}", mode);
            return;
        }
    };
    final_step(&n, &rels, true);
}
