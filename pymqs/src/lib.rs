// Copyright 2022, 2023 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::ffi::{c_char, CString};
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use pyo3::exceptions::{PyKeyboardInterrupt, PyValueError};
use pyo3::ffi::PyLong_FromString;
use pyo3::prelude::*;
use pyo3::pyfunction;
use pyo3::types::{PyList, PyLong};

use yamaquasi::{self, Algo, Int, Preferences, Uint, Verbosity};

#[pymodule]
fn pymqs(_: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(factor, m)?)?;
    m.add_function(wrap_pyfunction!(ecm, m)?)?;
    m.add_function(wrap_pyfunction!(quadratic_classgroup, m)?)?;
    Ok(())
}

#[pyfunction]
#[pyo3(
    signature = (n, /, algo = "auto",
                qs_fb_size = None, qs_interval_size = None, qs_use_double = None,
                verbose = "silent", timeout = None, threads = None),
    text_signature = "(n: int, /, algo: str, verbose: str, timeout=None, threads=None, qs_fb_size=None, qs_interval_size=None, qs_use_double=None) -> List[int]",
)]
/// Factors an integer into prime factors. The result is a list
/// whose product is the input argument.
///
/// An optional timeout (in seconds) can be specified to obtain
/// possibly partial factorizations. The function may exceed the specified
/// timeout depending on the computation status. If the timeout is reached,
/// returned factors may be composite.
///
/// Possible values for algo are:
/// auto (default), pm1, ecm, qs, mpqs, siqs.
///
/// Additional parameters are available for the quadratic sieves:
/// qs_fb_size: number of primes in the factor base
/// qs_interval_size: size of sieve intervals (a multiple of 32768)
fn factor(
    py: Python<'_>,
    n: &PyLong,
    algo: &str,
    qs_fb_size: Option<u32>,
    qs_interval_size: Option<u32>,
    qs_use_double: Option<bool>,
    verbose: &str,
    timeout: Option<f64>,
    threads: Option<usize>,
) -> PyResult<Py<PyList>> {
    let verbosity =
        Verbosity::from_str(verbose).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut prefs = Preferences::default();
    prefs.fb_size = qs_fb_size;
    prefs.interval_size = qs_interval_size;
    prefs.use_double = qs_use_double;
    prefs.threads = threads;
    prefs.verbosity = verbosity;
    // Handle interrupts.
    let start = std::time::Instant::now();
    let interrupted = Arc::new(AtomicBool::new(false));
    prefs.should_abort = Some(Box::new({
        let interrupted = interrupted.clone();
        move || {
            if let Some(t) = timeout {
                if start.elapsed().as_secs_f64() > t {
                    return true;
                }
            }
            if interrupted.load(Ordering::Relaxed) {
                return true;
            }
            let sig = Python::with_gil(|py| py.check_signals().is_err());
            if sig {
                interrupted.store(true, Ordering::Relaxed);
                if verbosity >= Verbosity::Info {
                    eprintln!("Interupted");
                }
            }
            sig
        }
    }));
    let alg = Algo::from_str(algo).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let n = Uint::from_str(&n.to_string()).map_err(|_| {
        PyValueError::new_err(format!(
            "Yamaquasi only accepts positive integers with at most 150 decimal digits"
        ))
    })?;
    if n.bits() > 512 {
        return Err(PyValueError::new_err(format!(
            "Yamaquasi only accepts positive integers with at most 150 decimal digits"
        )));
    }
    let result = py.allow_threads(|| yamaquasi::factor(n, alg, &prefs));
    if timeout.is_some()
        && Some(start.elapsed().as_secs_f64()) >= timeout
        && verbosity >= Verbosity::Info
    {
        eprintln!("Timeout reached");
    }
    // The expected semantics of KeyboardInterrupt are to end
    // the program even if the function has interesting data to return.
    if interrupted.load(Ordering::Relaxed) {
        return Err(PyKeyboardInterrupt::new_err(()));
    }
    let Ok(factors) = result else {
        return Err(PyValueError::new_err(format!("failed to factor {n}")));
    };
    let l = PyList::empty(py);
    for f in factors {
        // Use string for conversion.
        // FIXME: this is ugly.
        let s = CString::new(f.to_string()).unwrap();
        let sptr = s.as_ptr();
        let nullptr: *mut *mut c_char = std::ptr::null::<*mut c_char>() as *mut _;
        let obj = unsafe {
            let obj_ptr = PyLong_FromString(sptr, nullptr, 0);
            PyObject::from_owned_ptr(py, obj_ptr)
        };
        l.append(obj)?;
    }
    Ok(l.into())
}

#[pyfunction]
#[pyo3(
    signature = (n, curves, b1, b2, /, verbose = "silent", threads = None),
    text_signature = "(n: int, curves: int, b1: int, b2: float, /, verbose=\"silent\", threads=None) -> List[int]"
)]
fn ecm(
    py: Python<'_>,
    n: &PyLong,
    curves: u64,
    b1: u64,
    b2: f64,
    verbose: &str,
    threads: Option<usize>,
) -> PyResult<Py<PyList>> {
    let verbosity =
        Verbosity::from_str(verbose).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut prefs = Preferences::default();
    prefs.threads = threads;
    prefs.verbosity = verbosity;
    let n = Uint::from_str(&n.to_string()).map_err(|_| {
        PyValueError::new_err(format!(
            "Yamaquasi only accepts positive integers with at most 150 decimal digits"
        ))
    })?;
    if n.bits() > 512 {
        return Err(PyValueError::new_err(format!(
            "Yamaquasi only accepts positive integers with at most 150 decimal digits"
        )));
    }
    let tpool: Option<rayon::ThreadPool> = prefs.threads.map(|t| {
        if prefs.verbose(Verbosity::Verbose) {
            eprintln!("Using a pool of {t} threads");
        }
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build()
            .expect("cannot create thread pool")
    });
    let tpool = tpool.as_ref();
    let result = py
        .allow_threads(|| yamaquasi::ecm::ecm(n, curves as usize, b1 as usize, b2, &prefs, tpool));
    let (p, q) = match result {
        Some(t) => t,
        None => {
            return Err(PyValueError::new_err(format!(
                "No factors found by ECM B1={b1} B2={b2} with {curves} curves"
            )))
        }
    };

    let l = PyList::empty(py);
    for f in [p, q] {
        // Use string for conversion.
        // FIXME: this is ugly.
        let s = CString::new(f.to_string()).unwrap();
        let sptr = s.as_ptr();
        let nullptr: *mut *mut c_char = std::ptr::null::<*mut c_char>() as *mut _;
        let obj = unsafe {
            let obj_ptr = PyLong_FromString(sptr, nullptr, 0);
            PyObject::from_owned_ptr(py, obj_ptr)
        };
        l.append(obj)?;
    }
    Ok(l.into())
}

#[pyfunction]
#[pyo3(
    signature = (d, /, verbose = "silent", threads = None),
    text_signature = "(d: int, /, verbose=\"silent\", threads=None) -> Tuple[int, List[int], List[Tuple[int,List[int]]]]"
)]
/// Compute the class group of Q(sqrt(d)) where d is a negative integer
/// such that d % 4 == 0 or 1.
///
/// The result is a tuple containing:
/// - the class number h
/// - the invariants of the class group [di] such that product(di) == h
/// - a list of generators (p, [vi]) where [vi] are the coordinates of p
fn quadratic_classgroup(
    py: Python<'_>,
    d: &PyLong,
    verbose: &str,
    threads: Option<usize>,
) -> PyResult<Py<PyAny>> {
    let verbosity =
        Verbosity::from_str(verbose).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut prefs = Preferences::default();
    prefs.threads = threads;
    prefs.verbosity = verbosity;
    let d = Int::from_str(&d.to_string()).map_err(|_| {
        PyValueError::new_err(format!(
            "Yamaquasi only accepts negative integers with at most 75 decimal digits"
        ))
    })?;
    if !d.is_negative() || d.unsigned_abs().bits() > 250 {
        return Err(PyValueError::new_err(format!(
            "Yamaquasi only accepts negative integers with at most 75 decimal digits"
        )));
    }
    if d.bit(1) {
        return Err(PyValueError::new_err(format!(
            "Discriminant must be 0 or 1 mod 4"
        )));
    }
    let tpool: Option<rayon::ThreadPool> = prefs.threads.map(|t| {
        if prefs.verbose(Verbosity::Verbose) {
            eprintln!("Using a pool of {t} threads");
        }
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build()
            .expect("cannot create thread pool")
    });
    let tpool = tpool.as_ref();
    let result = py.allow_threads(|| yamaquasi::classgroup::classgroup(&d, &prefs, tpool));
    let grp = match result {
        Some(grp) => grp,
        None => {
            return Err(PyValueError::new_err(format!(
                "Classgroup computation failure"
            )))
        }
    };
    Ok((grp.h, grp.invariants, grp.gens).into_py(py))
}
