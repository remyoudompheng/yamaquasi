// Copyright 2022 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::str::FromStr;

use yamaquasi::{siqs, Uint};

fn main() {
    let arg = arguments::parse(std::env::args()).unwrap();
    if arg.get::<bool>("help").is_some() || arg.orphans.len() != 1 {
        eprintln!("Usage: ymqs-calibrate [OPTIONS] NUMBER");
        eprintln!("");
        eprintln!("Options:");
        eprintln!("  --help                    show this help");
        return;
    }
    let number = &arg.orphans[0];
    let n = Uint::from_str(number).unwrap();
    eprintln!("Input number {}", n);
    siqs::siqs_calibrate(n);
}
