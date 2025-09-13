def ecm(
    n: int,
    curves: int,
    b1: int,
    b2: float,
    /,
    verbose: str = "silent",
    threads: int | None = None,
) -> list[int]: ...


def factor(
    n: int,
    /,
    algo: str,
    verbose: str,
    timeout: float | None = None,
    threads: int | None = None,
    qs_fb_size: int | None = None,
    qs_interval_size: int | None = None,
    qs_use_double: bool | None = None,
) -> list[int]: ...

def factor_smooth(n: int, factor_bits: int) -> list[int]: ...

def quadratic_classgroup(
    d: int, /, verbose: str = "silent", threads: int | None = None
) -> tuple[int, list[int], list[tuple[int, list[int]]]]: ...
