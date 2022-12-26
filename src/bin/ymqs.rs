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
use yamaquasi::Uint;
use yamaquasi::{factor, pseudoprime, Algo, Preferences};

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
        eprintln!("  --large B1:               multiplier for large primes");
        eprintln!("  --use-double true|false:  use double large prime");
        return;
    }
    let mode = arg.get::<String>("mode").unwrap_or("auto".into());
    let threads = arg.get::<usize>("threads");
    let fb_user = arg.get::<u32>("fb");
    let large = arg.get::<u64>("large");
    let double = arg.get::<bool>("use-double");
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

    let prefs = Preferences {
        fb_size: fb_user,
        large_factor: large,
        use_double: double,
        threads,
        verbose: true,
    };
    let alg = Algo::from_str(&mode).unwrap();
    let factors = factor(n, alg, &prefs);
    for f in factors {
        if !pseudoprime(f) {
            eprintln!("composite factor: {} ({} bits)", f, f.bits());
        }
        println!("{}", f);
    }
}
