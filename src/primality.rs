// SPDX-License-Identifier: BSD-3-Clause
// SPDX-FileCopyrightText: 2026 Rémy Oudompheng

//! Primality testing

use crate::{fbase, Uint};

use crate::arith::Num;
use crate::arith_montgomery::{MInt, Mod64, Zmod, ZmodN};

/// Primality test of u64 using a deterministic Miller-Rabin test for small bases.
///
/// When it returns false, the number is guaranteed composite.
/// When it returns true, the number is prime conditionally to a separate exhaustive study.
pub fn isprime64(p: u64) -> bool {
    if p < 4 {
        return p >= 2;
    }
    if p & 1 == 0 {
        return false;
    }
    if p >> 28 == 0 {
        return crate::trialdiv::isprime32(p as u32);
    }

    // Compute auxiliary numbers for modular arithmetic.
    let zn = Mod64::new64(p);
    let r1 = 0_u64.wrapping_sub(p) % p; // 2^64 % p == (2^64-p) % p
    let r2 = ((r1 as u128 * r1 as u128) % (p as u128)) as u64;

    let tz = (p - 1).trailing_zeros();
    let podd = p >> tz;

    let one = r1;
    let pm1 = p - r1;
    // Performs the Miller test for base b.
    let miller = move |b: u64| {
        let b = zn.mul(zn.cast(b), r2);
        // Compute b^podd
        let mut pow = zn.pow1(b, podd);
        let mut ok = pow == one || pow == pm1;
        for _ in 0..tz {
            pow = zn.mul(pow, pow);
            if pow == pm1 {
                ok = true;
                break;
            } else if pow == one {
                break;
            }
        }
        ok
    };

    let h = p.wrapping_mul(14695981039346656037) >> 56;

    // Bases for 48-bit integers.
    if p >> 48 == 0 {
        // Witness: 2 and an optimal base for each bucket
        if !miller(2) {
            return false;
        }
        if !miller(MILLER_RABIN48[h as usize] as u64) {
            return false;
        }
        for a in [2, 3, 4, 5, 6, 7, 8, 9, 12] {
            // Is there a solution to (x+1)(ax+1) = p ?
            if perfect_sqrt_u64(4 * a * p + (a - 1) * (a - 1)).is_some() {
                return false;
            }
        }
        // Is there a solution to (2x+1)(3x+1) = p
        if perfect_sqrt_u64(24 * p + 1).is_some() {
            return false;
        }
        return true;
    }

    // Bases for 64-bit integers
    // Witnesses are: 2 and 2 optimal bases for each hash bucket.
    if !miller(2) {
        return false;
    }
    let (b1, b2) = MILLER_RABIN64[h as usize];
    if !miller(b1 as u64) {
        return false;
    }
    if !miller(b2 as u64) {
        return false;
    }

    for a in 2..17 {
        // Is there a solution to (x+1)(ax+1) = p ?
        let ap = 4 * a as u128 * p as u128 + ((a - 1) * (a - 1)) as u128;
        if perfect_sqrt_u80(ap).is_some() {
            return false;
        }
    }
    // Is there a solution to (2x+1)(3x+1) = p
    if perfect_sqrt_u80(24 * p as u128 + 1).is_some() {
        return false;
    }
    // Is there a solution to (3x+1)(4x+1) = p
    if perfect_sqrt_u80(48 * p as u128 + 1).is_some() {
        return false;
    }
    // Is there a solution to (2x+1)(5x+1) = p
    if perfect_sqrt_u80(40 * p as u128 + 9).is_some() {
        return false;
    }
    true
}

const MILLER_RABIN48: [u16; 256] = [
    263, 855, 389, 3097, 11, 101, 1607, 553, 433, 1229, 4639, 17725, 1507, 203, 503, 623, 6015,
    205, 47, 741, 199, 1479, 4185, 255, 895, 505, 713, 1545, 205, 45, 3565, 293, 403, 3141, 11,
    2261, 345, 437, 1817, 2121, 1527, 493, 709, 5153, 1669, 237, 6311, 191, 1675, 317, 2441, 159,
    3449, 2935, 255, 2179, 2335, 1287, 339, 381, 837, 593, 3553, 1487, 2723, 413, 215, 429, 217,
    1217, 367, 3567, 71, 209, 221, 213, 1033, 2975, 7675, 1221, 497, 1631, 1941, 2273, 677, 255,
    645, 1733, 209, 969, 1641, 85, 1569, 921, 347, 459, 29, 7699, 581, 179, 2847, 1339, 1407, 2119,
    2205, 939, 683, 591, 227, 6229, 509, 165, 1759, 6479, 3265, 1943, 315, 373, 1977, 2709, 1149,
    1869, 1445, 1815, 485, 7077, 3503, 2113, 3917, 381, 375, 161, 305, 303, 2737, 3005, 2811, 999,
    69, 599, 2043, 933, 297, 411, 1507, 101, 1869, 191, 829, 1781, 3145, 2391, 1019, 537, 369, 157,
    2475, 893, 3777, 65, 3811, 1659, 3335, 855, 345, 1317, 29, 39, 3277, 1319, 587, 1123, 1075,
    769, 2299, 17, 2563, 959, 1395, 3535, 29, 1159, 1421, 3377, 1213, 2127, 1237, 13049, 101, 711,
    293, 2497, 1259, 485, 4785, 181, 141, 211, 5407, 1637, 1201, 331, 525, 2703, 391, 77, 647, 389,
    263, 427, 1561, 2075, 3211, 293, 671, 1141, 4503, 3197, 185, 213, 83, 1601, 159, 1527, 693,
    17473, 593, 493, 1327, 393, 437, 59, 451, 2179, 15209, 991, 725, 1517, 6139, 1777, 119, 1239,
    459, 3893, 325, 19, 901, 911, 525, 1281, 1243, 1283, 677, 303, 1109, 5959,
];

#[rustfmt::skip]
const MILLER_RABIN64: [(u16, u16); 256] = [
    (291, 559), (39, 87), (77, 141), (41, 387), (37, 339), (297, 513), (95, 109), (513, 925),
    (323, 355), (19, 23), (13, 341), (199, 207), (171, 229), (79, 285), (17, 145), (67, 187),
    (489, 773), (123, 281), (83, 163), (113, 143), (159, 259), (159, 1051), (11, 123), (695, 765),
    (139, 209), (107, 827), (99, 457), (143, 305), (201, 491), (73, 311), (335, 463), (191, 353),
    (305, 743), (31, 429), (69, 279), (73, 115), (153, 171), (59, 69), (65, 199), (79, 787),
    (33, 371), (83, 85), (29, 269), (33, 333), (305, 375), (11, 161), (63, 365), (439, 449),
    (63, 135), (109, 585), (307, 449), (209, 295), (59, 153), (57, 425), (263, 337), (383, 387),
    (33, 79), (51, 67), (341, 583), (143, 515), (181, 301), (43, 449), (183, 655), (7, 425),
    (227, 437), (113, 349), (21, 321), (107, 233), (381, 709), (195, 435), (213, 399), (87, 227),
    (17, 335), (253, 879), (149, 333), (93, 165), (239, 367), (95, 145), (211, 449), (89, 295),
    (223, 401), (293, 381), (17, 189), (209, 319), (153, 253), (11, 57), (229, 233), (71, 419),
    (167, 265), (321, 323), (55, 513), (621, 639), (105, 757), (175, 377), (7, 351), (235, 449),
    (255, 539), (299, 321), (17, 401), (313, 415), (303, 555), (461, 509), (241, 257), (205, 589),
    (301, 565), (331, 595), (187, 369), (67, 403), (269, 323), (461, 603), (47, 429), (323, 485),
    (89, 151), (183, 247), (479, 513), (379, 401), (21, 133), (145, 251), (39, 187), (71, 127),
    (227, 239), (283, 419), (23, 409), (163, 403), (31, 181), (141, 327), (67, 373), (113, 487),
    (35, 115), (41, 403), (107, 249), (13, 99), (89, 455), (79, 251), (129, 709), (17, 31),
    (403, 585), (101, 179), (111, 447), (287, 677), (7, 59), (209, 385), (197, 357), (457, 927),
    (139, 337), (175, 309), (555, 599), (93, 277), (57, 417), (365, 413), (231, 415), (73, 323),
    (223, 615), (559, 569), (21, 205), (223, 251), (283, 471), (437, 457), (185, 321), (205, 207),
    (33, 197), (57, 179), (203, 407), (221, 487), (189, 347), (93, 209), (313, 489), (79, 151),
    (509, 639), (385, 447), (131, 227), (165, 177), (103, 271), (671, 847), (253, 261), (173, 689),
    (395, 473), (319, 549), (61, 443), (161, 295), (157, 255), (65, 101), (471, 565), (241, 827),
    (77, 149), (249, 633), (229, 429), (251, 341), (447, 465), (139, 329), (77, 819), (353, 377),
    (757, 837), (219, 241), (391, 521), (131, 215), (159, 235), (101, 355), (179, 751), (63, 319),
    (197, 419), (83, 89), (113, 215), (95, 97), (41, 59), (385, 537), (35, 223), (319, 327),
    (357, 483), (89, 189), (241, 377), (127, 567), (425, 515), (179, 183), (257, 323), (739, 743),
    (261, 519), (211, 337), (291, 465), (397, 465), (337, 391), (73, 333), (217, 601), (103, 127),
    (111, 269), (283, 393), (113, 211), (531, 1317), (449, 1113), (7, 163), (115, 495), (23, 99),
    (115, 551), (395, 399), (279, 307), (293, 305), (47, 87), (145, 199), (53, 213), (723, 769),
    (19, 119), (251, 435), (21, 95), (109, 251), (63, 111), (85, 449), (157, 181), (605, 669),
    (211, 511), (359, 439), (91, 185), (115, 139), (145, 553), (337, 443), (189, 465), (417, 849),
];

/// Probabilistic primality test using a Miller test for small bases.
pub fn pseudoprime(p: Uint) -> bool {
    // Montgomery arithmetic is only for odd numbers.
    if !p.bit(0) {
        return p.try_into() == Ok(2_u64);
    }
    if p.bits() <= 64 {
        return isprime64(p.low_u64());
    }
    miller_rabin(p)
}

fn pow_mod(zp: &ZmodN, x: MInt, exp: &Uint) -> MInt {
    let mut res = zp.one();
    let mut x = x;
    for b in 0..exp.bits() {
        if exp.bit(b) {
            res = zp.mul(&res, &x);
        }
        x = zp.mul(&x, &x);
    }
    res
}

fn miller_rabin(p: Uint) -> bool {
    let zp = ZmodN::new(p);
    let s = (p.low_u64() - 1).trailing_zeros();
    let p_odd = p >> s;
    for &b in &fbase::SMALL_PRIMES {
        let mut pow = pow_mod(&zp, zp.from_int(b.into()), &p_odd);
        let pm1 = zp.sub(&zp.zero(), &zp.one());
        let mut ok = pow == zp.one() || pow == pm1;
        for _ in 0..s {
            pow = zp.mul(&pow, &pow);
            if pow == pm1 {
                ok = true;
                break;
            } else if pow == zp.one() {
                break;
            }
        }
        if !ok {
            return false;
        }
    }
    true
}

#[allow(unused)]
fn perfect_sqrt_u32(n: u32) -> Option<u32> {
    let tz = n.trailing_zeros();
    if tz & 1 == 1 {
        return None;
    }
    let nred = n >> tz;
    if nred & 7 != 1 {
        return None;
    }
    // 1/sqrt(x) modulo 32
    const INVSQRT: [u8; 8] = [0, 5, 6, 3, 4, 1, 2, 7];
    let mut r = 1 + 4 * INVSQRT[(nred / 8) as usize & 7] as u32;
    // r *= (3 - n*r*r)/2 is the Newton iteration for 1/sqrt(x)
    for _ in 0..2 {
        r = r.wrapping_mul(3_u32.wrapping_sub(nred.wrapping_mul(r).wrapping_mul(r)) >> 1);
    }
    r = r.wrapping_mul(nred);
    r &= 0xffff;
    let rneg = 0x10000 - r;
    if r * r == nred {
        Some(r << (tz / 2))
    } else if rneg * rneg == nred {
        Some(rneg << (tz / 2))
    } else {
        None
    }
}

fn perfect_sqrt_u64(n: u64) -> Option<u64> {
    let tz = n.trailing_zeros();
    if tz & 1 == 1 {
        return None;
    }
    let nred = n >> tz;
    if nred & 7 != 1 {
        return None;
    }
    // 1/sqrt(x) modulo 64
    const INVSQRT: [u8; 16] = [0, 5, 6, 3, 12, 1, 2, 15, 8, 13, 14, 11, 4, 9, 10, 7];
    let mut r = 1 + 4 * INVSQRT[(nred / 8) as usize & 15] as u64;
    // r *= (3 - n*r*r)/2 is the Newton iteration for 1/sqrt(x)
    for _ in 0..3 {
        r = r.wrapping_mul(3_u64.wrapping_sub(nred.wrapping_mul(r).wrapping_mul(r)) >> 1);
    }
    r = r.wrapping_mul(nred);
    r &= 0xffffffff;
    let rneg = 0x100000000 - r;
    if r * r == nred {
        Some(r << (tz / 2))
    } else if rneg * rneg == nred {
        Some(rneg << (tz / 2))
    } else {
        None
    }
}

fn perfect_sqrt_u80(n: u128) -> Option<u64> {
    let tz = n.trailing_zeros();
    if tz & 1 == 1 {
        return None;
    }
    let nred = (n >> tz) as u64;
    if nred & 7 != 1 {
        return None;
    }
    // 1/sqrt(x) modulo 64
    const INVSQRT: [u8; 16] = [0, 5, 6, 3, 12, 1, 2, 15, 8, 13, 14, 11, 4, 9, 10, 7];
    let mut r = 1 + 4 * INVSQRT[(nred / 8) as usize & 15] as u64;
    // r *= (3 - n*r*r)/2 is the Newton iteration for 1/sqrt(x)
    for _ in 0..3 {
        r = r.wrapping_mul(3_u64.wrapping_sub(nred.wrapping_mul(r).wrapping_mul(r)) >> 1);
    }
    r = r.wrapping_mul(nred);
    r &= (1 << 40) - 1;
    let mut rneg: u64 = (1 << 40) - r;
    r <<= tz / 2;
    rneg <<= tz / 2;
    if r as u128 * r as u128 == n {
        Some(r)
    } else if rneg as u128 * rneg as u128 == n {
        Some(rneg)
    } else {
        None
    }
}

#[test]
fn test_pseudoprime() {
    use std::str::FromStr;

    assert!(!pseudoprime(1_u64.into()));
    assert!(pseudoprime(2_u64.into()));
    assert!(!pseudoprime(4_u64.into()));
    assert!(pseudoprime(17_u64.into()));
    // Large prime
    assert!(pseudoprime(Uint::from_str("1515019151549030796823931666316891543876480618160148234227332522965297454091879022905608234715852754566536639937").unwrap()));
    // Some composite number.
    assert!(!pseudoprime(
        Uint::from_str("893439027234689082677874957196339924735254319540").unwrap()
    ));
    // Carmichael number.
    assert!(!pseudoprime(9746347772161_u64.into()));

    // Sorenson-Webster, Strong pseudoprimes to twelve bases, arXiv:1509.00864
    assert!(!pseudoprime(Uint::from(3186_65857_83403_11511_67461_u128)));
    // Strong pseudoprime for 2..37
    assert!(!pseudoprime(Uint::from(33170_44064_67988_73859_61981_u128)));
    assert!(!pseudoprime(Uint::from(2995741773170734841812261_u128)));
    assert!(!pseudoprime(Uint::from(3404730287403079539471001_u128)));

    // Z. Zhang, Finding C_3-strong pseudoprimes (doi:10.1090/s0025-5718-04-01693-x)
    // Strong pseudoprime for 2..67
    assert!(!pseudoprime(
        Uint::from_str("664285341720894140846825851168090899459337851067").unwrap()
    ));
}

#[test]
fn test_isprime_many() {
    // Takes a few minutes (1us x 1M primes)
    let mut sieve = fbase::PrimeSieve::new();
    let mut count = 0;
    loop {
        let s = sieve.next();
        if s.len() == 0 {
            break;
        }
        if s[0] >> 28 != 0 {
            break;
        }
        for &p in &s[..16] {
            count += 1;
            assert!(isprime64(p as u64), "{p}");
            // Also test p(2p-1) and p(3p-2) which are often pseudoprimes
            let pp = (p as u64) * (2 * p - 1) as u64;
            assert!(!isprime64(pp), "{pp}");
            let pp = (p as u64) * (3 * p - 2) as u64;
            assert!(!isprime64(pp), "{pp}");
        }
    }
    eprintln!("test_isprime_many: tested {count} primes");
}

#[test]
fn test_isprime64() {
    assert!(!isprime64(1));
    assert!(isprime64(2));
    assert!(isprime64(3));
    assert!(!isprime64(4));
    assert!(!isprime64(9));
    assert!(!isprime64(15));
    assert!(isprime64(17));
    for p in fbase::primes(4096) {
        assert!(isprime64(p as u64), "{p}");
    }
    assert!(!isprime64(9746347772161));
    // Some prime number
    assert!(isprime64(9938261980284378737));
    // Some composite number
    assert!(!isprime64(11775166524998067797));

    // Strong pseudoprime to the first eight prime bases
    // Jiang, Deng, Math Comp 83 (2014)
    assert!(!isprime64(3825_12305_65464_13051));

    // Various strong pseudoprimes
    // Jaeschke, On strong pseudoprimes to several bases, Math Comp 61 (1993)
    assert!(!isprime64(25326001));
    assert!(!isprime64(161304001));
    assert!(!isprime64(960946321));
    assert!(!isprime64(1157839381));
    assert!(!isprime64(3215031751));
    assert!(!isprime64(908201935681));
    assert!(!isprime64(966299321527));
    assert!(!isprime64(997031384161));
}

#[test]
fn test_perfect_sqrt() {
    for i in 0..0x8000 {
        let x: u32 = (2 * i as u32 + 1).pow(2);
        assert_eq!(
            perfect_sqrt_u32(x),
            Some(2 * i + 1),
            "x={x} r={:?}",
            perfect_sqrt_u32(x)
        );
    }

    for i in 0..(u32::MAX / 39) as u64 {
        let x = 19 * (2 * i + 1);
        assert_eq!(
            perfect_sqrt_u64(x * x),
            Some(x),
            "x={x} r={:?}",
            perfect_sqrt_u64(x * x)
        );
    }

    for i in 0..(1 << 39) / 123457 as u64 {
        let x = 123457 * (2 * i + 1);
        let r = perfect_sqrt_u80(x as u128 * x as u128);
        assert_eq!(r, Some(x), "x={x} r={:?}", r);
    }
}
