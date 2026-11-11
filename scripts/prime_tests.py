# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileCopyrightText: 2026 Rémy Oudompheng

"""
Analysis of primality tests.

Uses input files from: https://www.cecm.sfu.ca/Pseudoprimes
(Jan Feitsma, William Galway, Tables of pseudoprimes and related data)

See also: https://miller-rabin.appspot.com/ (file 2-SPRP-2-to-64.zip)

Many pseudoprimes are semiprimes with shape (ak+1)(bk+1) where a,b are small integers.

The code splits pseudoprimes into 256 hash buckets using a simple hash function
and searchs for small Miller-Rabin bases covering the entire bucket (after excluding
several families of (ak+1)(bk+1) numbers).
"""

import math
import os
from concurrent.futures import ProcessPoolExecutor

import gmpy2


def strong_pseudoprimes(ns, b):
    return [n for n in ns if strong_pseudoprime(n, b)]


def strong_pseudoprime(n, b):
    """
    n is a strong pseudoprime in base b if for n-1 = r 2^s
    - b^r = 1 mod n
    - b^(r << k) = -1 for some k in (0..s)
    """
    n = gmpy2.mpz(n)
    e = n - 1
    tz = (e ^ (e - 1)).bit_length() - 1
    x = pow(b, e >> tz, n)
    if x == 1:
        return int(n)
    for _ in range(tz):
        if x == n - 1:
            return int(n)
        x = pow(x, 2, n)
    return None


def special(x):
    for a in range(2, 6):
        disc = 4 * a * x + (a - 1) ** 2
        if disc == math.isqrt(disc) ** 2:
            return True


def strong_pseudoprimes2(bits):
    with os.popen("xzcat /tmp/2-SPRP-2-to-64.xz") as f:
        for line in f:
            x = int(line)
            if x.bit_length() > bits:
                break
            yield x


def hbucket(n):
    return ((n * 14695981039346656037) >> 56) & 0xFF


def prime32():
    print("== 32 bit integers ==")
    strong_ps2 = list(strong_pseudoprimes2(32))
    assert not any(gmpy2.is_prime(x) for x in strong_ps2)

    special = set()
    stats = {}
    for x in strong_ps2:
        # Many pseudoprimes have shape (k + 1)(ak + 1) where a is small integer
        # n = (k + 1)(ak + 1) iff ak² + (a+1)k + 1 == n
        # (a+1)^2 + 4(n-1)a = 4an + (a-1)^2 is a square
        # In particular n is not prime.
        for a in (2, 3, 4, 5, 6, 7, 8, 9, 12, 16):
            disc = 4 * a * x + (a - 1) ** 2
            if gmpy2.is_square(disc):
                stats[a] = stats.get(a, 0) + 1
                special.add(x)
                break
        # (2k+1)(3k+1)=n <=> 24n+1 is a square
        if gmpy2.is_square(24 * x + 1):
            stats[1.5] = stats.get(1.5, 0) + 1
            special.add(x)
        # (3k+1)(4k+1)=n <=> 48n+1 is a square
        elif gmpy2.is_square(48 * x + 1):
            stats[1.33] = stats.get(1.33, 0) + 1
            special.add(x)

    print(len(strong_ps2), "strong 2-pseudoprimes")
    print(len(special), "special", stats)

    # Size 2 Miller-Rabin bases for 32-bit integers
    bs = list(range(3, 500, 2))
    best = len(strong_ps2)
    ps2 = sorted(set(strong_ps2) - special)
    with ProcessPoolExecutor() as pool:
        for b, spb in zip(bs, pool.map(strong_pseudoprimes, (ps2 for _ in bs), bs)):
            count = len(spb)
            if count <= best:
                best = count
                print(f"Basis 2,{b} => {count}")
                if count < 10:
                    print(
                        [
                            x
                            for x in strong_ps2
                            if strong_pseudoprime(x, b) and x not in special
                        ]
                    )


def spsp_stats():
    sizes = 65 * [0]
    strong_ps2 = list(strong_pseudoprimes2(64))
    for x in strong_ps2:
        sizes[x.bit_length()] += 1
    print("Size stats:", sizes)

    # assert not any(gmpy2.is_prime(x) for x in strong_ps2)

    print(len(strong_ps2), "strong 2-pseudoprimes")

    ps2 = strong_pseudoprimes(strong_ps2, 3)
    print("2,3=>", len(ps2), "strong pseudoprimes")
    ps3 = strong_pseudoprimes(ps2, 5)
    print("2,3,5=>", len(ps3), "strong pseudoprimes")
    ps4 = strong_pseudoprimes(ps3, 7)
    print("2,3,5,7=>", len(ps4), "strong pseudoprimes")
    ps5 = strong_pseudoprimes(ps4, 11)
    print("2,3,5,7,11=>", len(ps5), "strong pseudoprimes")

    ps2 = strong_pseudoprimes(strong_ps2, 1009)
    print("2,1009=>", len(ps2), "strong pseudoprimes")
    ps3 = strong_pseudoprimes(ps2, 1013)
    print("2,1009,1013=>", len(ps3), "strong pseudoprimes")
    ps4 = strong_pseudoprimes(ps3, 1019)
    print("2,1009,1013,1019=>", len(ps4), "strong pseudoprimes")
    ps5 = strong_pseudoprimes(ps4, 1021)
    print("2,1009,1013,1019,1021=>", len(ps5), "strong pseudoprimes")


def prime48hash(h):
    print(f"== 48 bit integers hash={h} ==")
    strong_ps2 = [x for x in strong_pseudoprimes2(48) if hbucket(x) == h]

    special = set()
    stats = {}
    for x in strong_ps2:
        # Many pseudoprimes have shape (k + 1)(ak + 1) where a is small integer
        # n = (k + 1)(ak + 1) iff ak² + (a+1)k + 1 == n
        # (a+1)^2 + 4(n-1)a = 4an + (a-1)^2 is a square
        # In particular n is not prime.
        for a in (2, 3, 4, 5, 6, 7, 8, 9, 12):
            disc = 4 * a * x + (a - 1) ** 2
            if gmpy2.is_square(disc):
                stats[a] = stats.get(a, 0) + 1
                special.add(x)
                break
        # a=3/2
        if gmpy2.is_square(24 * x + 1):
            stats[1.5] = stats.get(1.5, 0) + 1
            special.add(x)

    print(len(strong_ps2), "strong 2-pseudoprimes")
    print(len(special), "special", stats)

    # Find good Miller-Rabin bases for 2^48
    strong_ps2 = [x for x in strong_ps2 if x not in special]
    bs = list(range(3, 50000, 2))
    best = len(strong_ps2)
    with ProcessPoolExecutor() as pool:
        for b, spb in zip(
            bs, pool.map(strong_pseudoprimes, (strong_ps2 for _ in bs), bs)
        ):
            count = len(spb)
            if count <= best:
                best = count
                print(f"Basis 2,{b} => {count}")
            if count == 0:
                return b
    assert best == 0


def prime64hash(h):
    print(f"== 64 bit integers hash={h} ==")
    strong_ps2 = [x for x in strong_pseudoprimes2(64) if hbucket(x) == h]
    print(len(strong_ps2), "strong 2-pseudoprimes")

    special = set()
    stats = {}
    for x in strong_ps2:
        # Many pseudoprimes have shape (k + 1)(ak + 1) where a is small integer
        # In particular n is not prime.
        for a in range(2, 17):
            disc = 4 * a * x + (a - 1) ** 2
            if gmpy2.is_square(disc):
                stats[a] = stats.get(a, 0) + 1
                special.add(x)
                break
        # a=3/2
        if gmpy2.is_square(24 * x + 1):
            stats[1.5] = stats.get(1.5, 0) + 1
            special.add(x)
        # a=4/3
        elif gmpy2.is_square(48 * x + 1):
            stats[1.33] = stats.get(1.33, 0) + 1
            special.add(x)
        # a=5/2
        elif gmpy2.is_square(40 * x + 9):
            stats[2.5] = stats.get(2.5, 0) + 1

    print(len(special), "special", stats)

    # Find good Miller-Rabin bases for 2^64
    strong_ps2 = [x for x in strong_ps2 if x not in special]
    candidates = {}
    bs = list(range(3, 5000, 2))
    best = len(strong_ps2)
    with ProcessPoolExecutor(8) as pool:
        for b, spb in zip(
            bs, pool.map(strong_pseudoprimes, (strong_ps2 for _ in bs), bs)
        ):
            candidates[b] = set(spb)
            for b2 in range(3, b, 2):
                results = candidates[b] & candidates[b2]
                count = len(results)
                if count <= best:
                    best = count
                    print(f"Basis 2,{b2},{b} => {count}")
                    if count == 0:
                        return b2, b


def special_stats():
    stats = {}
    total = 0
    for x in strong_pseudoprimes2(48):
        total += 1
        # Many pseudoprimes have shape (k + 1)(ak + 1) where a is small integer
        # n = (k + 1)(ak + 1) iff ak² + (a+1)k + 1 == n
        # (a+1)^2 + 4(n-1)a = 4an + (a-1)^2 is a square
        # In particular n is not prime.
        for a in range(2, 20):
            disc = 4 * a * x + (a - 1) ** 2
            if gmpy2.is_square(disc):
                stats[a] = stats.get(a, 0) + 1
                break
        # (2k+1)(3k+1)=n <=> 24n+1 is a square
        if gmpy2.is_square(24 * x + 1):
            stats[1.5] = stats.get(1.5, 0) + 1
        # (3k+1)(4k+1)=n <=> 48n+1 is a square
        elif gmpy2.is_square(48 * x + 1):
            stats[1.33] = stats.get(1.33, 0) + 1
        # (2k+1)(5k+1)=n <=> 40n+9 is a square
        elif gmpy2.is_square(40 * x + 9):
            stats[2.5] = stats.get(2.5, 0) + 1
    print(total)
    print(stats)


# fmt:off
# Precomputed optimal Miller-Rabin base for each hash bucket
MILLER_RABIN48 = [
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
]
MILLER_RABIN64 = [
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
]
# fmt:on


def check_prime48():
    print("== Check 48 bit integers ==")
    total = 0
    stats = {}
    for x in strong_pseudoprimes2(48):
        h = hbucket(x)
        total += 1

        is_special = False
        # Many pseudoprimes have shape (k + 1)(ak + 1) where a is small integer
        # In particular n is not prime.
        for a in (2, 3, 4, 5, 6, 7, 8, 9, 12):
            disc = 4 * a * x + (a - 1) ** 2
            if gmpy2.is_square(disc):
                stats[a] = stats.get(a, 0) + 1
                is_special = True
                break
        # a=3/2
        if gmpy2.is_square(24 * x + 1):
            stats[1.5] = stats.get(1.5, 0) + 1
            is_special = True

        if is_special:
            continue

        b = MILLER_RABIN48[h]
        if all(strong_pseudoprime(x, base) for base in (2, b)):
            raise ValueError(f"strong pseudoprime {x}")

    print(total, "pseudoprimes checked")
    print(stats)


def check_prime64():
    print("== Check 64 bit integers ==")
    total = 0
    stats = {}
    for x in strong_pseudoprimes2(64):
        h = hbucket(x)
        total += 1

        is_special = False
        # Many pseudoprimes have shape (k + 1)(ak + 1) where a is small integer
        # In particular n is not prime.
        for a in range(2, 17):
            disc = 4 * a * x + (a - 1) ** 2
            if gmpy2.is_square(disc):
                stats[a] = stats.get(a, 0) + 1
                is_special = True
                break
        # a=3/2
        if gmpy2.is_square(24 * x + 1):
            stats[1.5] = stats.get(1.5, 0) + 1
            is_special = True
        # a=4/3
        elif gmpy2.is_square(48 * x + 1):
            stats[1.33] = stats.get(1.33, 0) + 1
            is_special = True
        # a=5/2
        elif gmpy2.is_square(40 * x + 9):
            stats[2.5] = stats.get(2.5, 0) + 1
            is_special = True
        if is_special:
            continue

        b1, b2 = MILLER_RABIN64[h]
        if all(strong_pseudoprime(x, base) for base in (2, b1, b2)):
            raise ValueError(f"strong pseudoprime {x}")

    print(total, "pseudoprimes checked")
    print(stats)


if __name__ == "__main__":
    # Strong pseudoprime from Zhang, Finding C3-strong pseudoprimes
    # doi:10.1090/s0025-5718-04-01693-x
    n = 9688312712744590973050578123260748216127001625571
    for b in range(2, 71):
        assert strong_pseudoprime(n, b)

    spsp_stats()
    special_stats()

    prime32()

    basis48 = {}
    for i in range(256):
        basis48[i] = prime48hash(i)
    print(basis48)

    basis64 = {}
    for i in range(256):
        basis64[i] = prime64hash(i)
    print(basis64)

    check_prime48()
    check_prime64()
