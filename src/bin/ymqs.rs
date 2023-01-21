// Copyright 2022 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::str::FromStr;

use yamaquasi::arith::U1024;
use yamaquasi::Uint;
use yamaquasi::{factor, pseudoprime, Algo, Preferences, Verbosity};

fn main() {
    let arg = arguments::parse(std::env::args()).unwrap();
    if arg.get::<bool>("help").is_some() || arg.orphans.len() != 1 {
        eprintln!("Usage: ymqs [OPTIONS] NUMBER");
        eprintln!("");
        eprintln!("Options:");
        eprintln!("  --help                    show this help");
        eprintln!("  --verbose silent|info|verbose|debug");
        eprintln!("  --mode ecm|qs|mpqs|siqs:  force algorithm selection");
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
    let v = arg.get::<String>("verbose").unwrap_or("info".into());
    let number = &arg.orphans[0];
    let n = U1024::from_str(number).expect("could not read decimal number");
    const MAXBITS: u32 = 512;
    if n.bits() > MAXBITS {
        panic!(
            "Number size ({} bits) exceeds {} bits limit",
            n.bits(),
            MAXBITS
        )
    }
    let n = Uint::from_str(number).unwrap();

    let mut prefs = Preferences::default();
    prefs.fb_size = fb_user;
    prefs.large_factor = large;
    prefs.use_double = double;
    prefs.threads = threads;
    prefs.verbosity = Verbosity::from_str(&v).unwrap();
    if prefs.verbose(Verbosity::Info) {
        eprintln!("Input number {}", n);
    }
    let alg = Algo::from_str(&mode).unwrap();
    let factors = factor(n, alg, &prefs);
    for f in factors {
        if !pseudoprime(f) && prefs.verbose(Verbosity::Info) {
            eprintln!("composite factor: {} ({} bits)", f, f.bits());
        }
        println!("{}", f);
    }
}
