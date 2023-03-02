// Copyright 2023 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! This program collects example numbers known to have failed
//! at some point during development.
//!
//! They may take time to process, so they are not part of unit tests.
//! Run it in release mode (cargo test -r --examples)

use std::str::FromStr;

use yamaquasi::{factor, Algo, Preferences, Uint, Verbosity};

fn main() {
    factor_examples();
}

const EXAMPLES: &[&str] = &[
    // Factor base 2, 3, 5, 17, 29, 41, 47, 61, 103, 131 (mult=1)
    "212433504133480536121",
    // Number with a poor factor base
    // 2, 5, 7, 11, 23, 59, 71, 79, 137, 139
    "23115841824250710013889",
    // Factor base 2, 5, 7, 11, 17, 29, 31, 37, 53, 101 (mult=1)
    "186177544101593788852542809",
    // A number with best multiplier 1 but a poor factor base:
    // 2, 3, 11, 13, 17, 79, 89, 101, 127, 131
    "145188395687209308974668110093663098558233257760297",
];

fn factor_examples() {
    // All examples should be factorable with ECM and the 3 quadratic sieves.
    let mut prefs = Preferences::default();
    prefs.verbosity = Verbosity::Silent;
    for e in EXAMPLES {
        eprintln!("Example {e}");
        let n = Uint::from_str(e).unwrap();
        for alg in [Algo::Ecm, Algo::Qs, Algo::Mpqs, Algo::Siqs] {
            let fs = factor(n, alg, &prefs);
            eprintln!("Algo {alg:?} OK: {fs:?}");
        }
    }
}

#[test]
fn test_examples() {
    factor_examples()
}
