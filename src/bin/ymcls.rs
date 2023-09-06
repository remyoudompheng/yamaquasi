// Copyright 2023 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::io::Write;
use std::path::PathBuf;
use std::str::FromStr;

use yamaquasi::classgroup::{self, estimate};
use yamaquasi::{Int, Preferences, Verbosity};

fn main() {
    let arg = arguments::parse(std::env::args()).unwrap();
    if arg.get::<bool>("help").is_some() || arg.orphans.len() != 2 {
        eprintln!("Usage: ymcls [OPTIONS] [-]DISCRIMINANT OUTPUTDIR");
        eprintln!("");
        eprintln!("Options:");
        eprintln!("  --help                    show this help");
        eprintln!("  --verbose silent|info|verbose|debug");
        eprintln!("  --threads N:              enable up to N computation threads");
        eprintln!("  --fb F:                   override automatic factor base size");
        eprintln!("  --large B1:               multiplier for large primes");
        eprintln!("  --use-double true|false:  use double large prime");
        return;
    }
    let threads = arg.get::<usize>("threads");
    let fb_user = arg.get::<u32>("fb");
    let large = arg.get::<u64>("large");
    let double = arg.get::<bool>("use-double");
    let v = arg.get::<String>("verbose").unwrap_or("info".into());
    let number = &arg.orphans[0];
    let outdir = &arg.orphans[1];
    let mut d = Int::from_str(number).expect("could not read input number");
    if d.is_positive() {
        d = -d;
    }
    const MAXBITS: u32 = 512;
    if d.unsigned_abs().bits() > MAXBITS {
        panic!(
            "Number size ({} bits) exceeds {MAXBITS} bits limit",
            d.unsigned_abs().bits(),
        )
    }
    if d.bit(1) {
        panic!("Discriminant must be 0 or 1 mod 4");
    }

    let mut prefs = Preferences::default();
    prefs.fb_size = fb_user;
    prefs.large_factor = large;
    prefs.use_double = double;
    prefs.threads = threads;
    prefs.verbosity = Verbosity::from_str(&v).unwrap();
    if prefs.verbose(Verbosity::Info) {
        eprintln!("Computing class group of discriminant {d}");
    }
    let (hmin, hmax) = estimate(&d);
    if prefs.verbose(Verbosity::Info) {
        eprintln!("Estimate by class number formula {hmin:.5e}-{hmax:.5e}")
    }

    // Create output directory
    std::fs::create_dir_all(outdir).unwrap();
    prefs.outdir = Some(PathBuf::from(outdir));

    // Dump parameters
    {
        let mut json: Vec<u8> = vec![];
        writeln!(&mut json, "{{").unwrap();
        writeln!(&mut json, r#"  "d": "{d}","#).unwrap();
        writeln!(&mut json, r#"  "h_estimate_min": {hmin},"#).unwrap();
        writeln!(&mut json, r#"  "h_estimate_max": {hmax}"#).unwrap();
        writeln!(&mut json, "}}").unwrap();
        std::fs::write(PathBuf::from(outdir).join("args.json"), json).unwrap();
    }

    // Create thread pool
    let tpool: Option<rayon::ThreadPool> = match prefs.threads {
        None | Some(1) => None,
        Some(t) => {
            if prefs.verbose(Verbosity::Verbose) {
                eprintln!("Using a pool of {t} threads");
            }
            Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(t)
                    .build()
                    .expect("cannot create thread pool"),
            )
        }
    };
    let tpool = tpool.as_ref();
    classgroup::ideal_relations(&d, &prefs, tpool);
}
