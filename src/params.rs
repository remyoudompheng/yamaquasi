// Copyright 2022, 2023 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::cmp::min;

use crate::Uint;

pub fn factor_base_size(n: &Uint) -> u32 {
    let sz = n.bits();
    if sz < 160 {
        // Factor bases from Silverman's article
        // 100 primes for 80 bits
        // 200 primes for 100 bits
        // 400 primes for 120 bits
        return ((20 + (sz % 20)) << (sz / 20)) / 3;
    } else {
        // 500 for 192 bits, grow as exp(C sqrt(sz))
        let rt = ((256 * sz) as f64).sqrt() as u32;
        let (a, b) = (rt / 12, rt % 12);
        return (12 + b) << (a - 10);
    }
}

/// Select a factor base size for the classical quadratic sieve.
/// The bitsize argument refers to the size of the input integer
/// without multiplier (because the multiplied integer should be more
/// effective that the original integer: it should not use larger parameters).
///
/// Compared to MPQS, classical quadratic sieve uses a single huge interval
/// so resulting numbers (2M sqrt(n)) can be larger by 10-20 bits.
pub fn qs_fb_size(bitsize: u32, use_double: bool) -> u32 {
    if bitsize <= 250 {
        select_fb_size(bitsize, use_double, QS_FBSIZES)
    } else {
        // use old formula with a 40 bit penalty.
        let rt = ((256 * (bitsize + 40)) as f64).sqrt() as u32;
        let (a, b) = (rt / 12, rt % 12);
        min(500_000, (12 + b) << (a - 10))
    }
}

pub fn mpqs_fb_size(bitsize: u32, use_double: bool) -> u32 {
    if bitsize < 390 {
        select_fb_size(bitsize, use_double, MPQS_FBSIZES)
    } else {
        // maximal factor base size
        500_000
    }
}

pub fn clsgrp_fb_size(bitsize: u32, use_double: bool) -> u32 {
    if bitsize < 380 {
        select_fb_size(bitsize, use_double, CLASSGROUP_FBSIZES)
    } else {
        // maximal factor base size
        60_000
    }
}

fn select_fb_size(bitsize: u32, use_double: bool, table: &'static [(u32, u32, u32)]) -> u32 {
    let idx = table.partition_point(|&(sz, _, _)| sz <= bitsize);
    if idx == 0 {
        16
    } else if idx == table.len() {
        // very large: return something less than 500000.
        min(500000, 1400 * (bitsize - 200))
    } else {
        // linearly interpolate
        let prev = table[idx - 1];
        let next = table[idx];
        let (fb_lo, fb_hi) = if use_double {
            (prev.2, next.2)
        } else {
            (prev.1, next.1)
        };
        ((next.0 - bitsize) * fb_lo + (bitsize - prev.0) * fb_hi) / (next.0 - prev.0)
    }
}

/// Preferred sizes for factor bases in classical quadratic sieve.
const QS_FBSIZES: &'static [(u32, u32, u32)] = &[
    // Bit size, Factor base (no double large prime), Factor base (with double large prime)
    (16, 16, 16),
    (70, 120, 60),
    (80, 120, 60),
    (90, 200, 90),
    (100, 300, 120),
    (110, 500, 200),
    (120, 800, 300),
    (130, 1200, 450),
    (140, 2000, 650),
    (150, 3200, 1000),
    (160, 5500, 1500),
    (170, 8000, 2500),
    (180, 13000, 4000),
    (190, 20000, 7000),
    (200, 26000, 10000),
    (210, 30000, 14000),
    (220, 35000, 20000),
    (230, 50000, 35000),
    (240, 80000, 50000),
    (250, 120000, 65000),
    (260, 160000, 80000),
];

/// Preferred sizes for factor bases in MPQS.
/// The polynomial roots computation is costly so smaller factor bases (compared to SIQS).
const MPQS_FBSIZES: &'static [(u32, u32, u32)] = &[
    // Bit size, Factor base (no double large prime), Factor base (with double large prime)
    (16, 16, 16),
    (70, 100, 60),
    (80, 130, 70),
    (90, 160, 90),
    (100, 200, 120),
    (110, 300, 150),
    (120, 400, 200),
    (130, 550, 250),
    (140, 750, 300),
    (150, 1000, 380),
    (160, 1400, 500),
    (170, 2000, 800),
    (180, 3000, 1300),
    (190, 5000, 2000),
    (200, 8000, 3000),
    (210, 11000, 4500),
    (220, 15000, 7000),
    (230, 20000, 10000),
    (240, 26000, 14000),
    (250, 35000, 18000),
    (260, 40000, 23000),
    (270, 48000, 30000),
    (280, 60000, 40000),
    (290, 75000, 55000),
    (300, 90000, 75000),
    (310, 110000, 90000),
    (320, 140000, 110000),
    (330, 200000, 130000),
    (340, 260000, 170000),
    (350, 330000, 220000),
    (360, 400000, 280000),
    // Maximal factor base size
    (390, 500000, 500000),
];

/// Factor base size for class group computations.
///
/// We follow the growth of SIQS parameters but the factor
/// bases are considerably smaller to keep linear algebra
/// cost below sieving.
const CLASSGROUP_FBSIZES: &'static [(u32, u32, u32)] = &[
    // For small inputs, linear algebra is fast, we can use
    // parameters similar to factoring.
    (16, 16, 16),
    (32, 24, 24),
    (60, 64, 48),
    (80, 100, 80),
    (120, 250, 150),
    (150, 700, 400),
    // For larger inputs, we need to keep factor base very small.
    (200, 1500, 800),
    (220, 3000, 1500),
    (250, 7000, 3500),
    (280, 12000, 7000),
    (320, 30000, 15000),
    (350, 60000, 30000),
    (380, 100000, 60000),
];

/// ECM/P-1 suitable parameters according to values of B2.
/// d1: size of "giant steps", such that φ(d1) is small and close to 2^k
/// d2: number of giant steps, a power of 2 close to φ(d1)/2,
///     multiplied by a number of blocks (2, 3, 4)
///
/// For small values where multipoint evaluation is not used,
/// the power of 2 constraint can be relaxed and d2 should be very close
/// to φ(d1)/2 (polynomial degree) for optimal cost.
const STAGE2_PARAMS: &[(f64, u64, u64)] = &[
    // B2, d1, d2
    // Using quadratic method, d2=φ(d1)/2, cost d2^2
    (660., 66, 10),
    (1080., 90, 12),
    (1920., 120, 16),
    (3e3, 150, 20),
    (5.04e3, 210, 24),
    (7.7e3, 240, 32),
    (13.2e3, 330, 40),
    (20e3, 420, 48),
    (33e3, 510, 64),
    (53e3, 660, 80),
    (81e3, 840, 96),
    (126e3, 1050, 120),
    (181e3, 1260, 144),
    (323e3, 1680, 192),
    (554e3, 2310, 240),
    (786e3, 2730, 288),
    (1.37e6, 3570, 384),
    // Use polyeval starting from here
    (2.3e6, 4620, 512),       // φ/2=480
    (4.7e6, 4620, 1024),      // φ/2=480
    (7.1e6, 4620, 1536),      // φ/2=480
    (9.5e6, 4620, 2048),      // φ/2=480
    (19e6, 9240, 2048),       // φ/2=960
    (28e6, 9240, 3072),       // φ/2=960
    (38e6, 9240, 4096),       // φ/2=960
    (78e6, 19110, 4096),      // φ/2=2016
    (117e6, 19110, 6144),     // φ/2=2016
    (156e6, 19110, 8192),     // φ/2=2016
    (322e6, 39270, 8192),     // φ/2=3840
    (643e6, 39270, 16384),    // φ/2=3840
    (1.3e9, 79170, 16384),    // φ/2=8064
    (2.6e9, 79170, 32768),    // φ/2=8064
    (5.2e9, 159390, 32768),   // φ/2=15840
    (10.5e9, 159390, 65536),  // φ/2=15840
    (21.6e9, 330330, 65536),  // φ/2=31680
    (32.5e9, 330330, 98304),  // φ/2=31680
    (43e9, 330330, 131072),   // φ/2=31680
    (136e9, 690690, 196608),  // φ/2=63360
    (362e9, 1381380, 262144), // φ/2=126720
    (543e9, 1381380, 393216), // φ/2=126720
    (724e9, 1381380, 524288),
    (1.5e12, 2852850, 524288), // φ/2=259200
    (2.99e12, 2852850, 1048576),
    (5.98e12, 5705700, 1048576), // φ/2=518400
    (1.2e13, 5705700, 2097152),
    (2.46e13, 11741730, 2097152), // φ/2=1013760
    (4.92e13, 11741730, 4194304),
];

pub fn stage2_params(b2: f64) -> (f64, u64, u64) {
    *STAGE2_PARAMS
        .iter()
        .min_by(|x, y| (x.0 - b2).abs().total_cmp(&(y.0 - b2).abs()))
        .unwrap()
}

#[test]
fn test_factor_base_size() {
    let b80 = factor_base_size(&(Uint::from(1u64) << 80));
    let b100 = factor_base_size(&(Uint::from(1u64) << 100));
    let b120 = factor_base_size(&(Uint::from(1u64) << 120));
    eprintln!("Bound 80-bit: {}", b80);
    eprintln!("Bound 100-bit: {}", b100);
    eprintln!("Bound 120-bit: {}", b120);
    assert!(80 <= b80 && b80 <= 120);
    assert!(160 <= b100 && b100 <= 240);
    assert!(320 <= b120 && b120 <= 480);

    let b160 = factor_base_size(&(Uint::from(1u64) << 160));
    let b192 = factor_base_size(&(Uint::from(1u64) << 192));
    let b224 = factor_base_size(&(Uint::from(1u64) << 224));
    let b256 = factor_base_size(&(Uint::from(1u64) << 256));
    eprintln!("Bound 160-bit: {}", b160);
    eprintln!("Bound 192-bit: {}", b192);
    eprintln!("Bound 224-bit: {}", b224);
    eprintln!("Bound 256-bit: {}", b256);

    assert!(1200 <= b160 && b160 <= 2000);
    assert!(2500 <= b192 && b192 <= 5000);
    assert!(7000 <= b224 && b224 <= 15000);
    assert!(30_000 <= b256 && b256 <= 50_000);
}
