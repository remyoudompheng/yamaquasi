// Copyright 2022, 2023 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::ffi::CString;
use std::str::FromStr;

use pyo3::exceptions::PyValueError;
use pyo3::ffi::PyLong_FromString;
use pyo3::prelude::*;
use pyo3::pyfunction;
use pyo3::types::{PyList, PyLong};

use yamaquasi::{self, Algo, Preferences, Uint, Verbosity};

#[pymodule]
fn pymqs(_: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(factor, m)?)?;
    m.add_function(wrap_pyfunction!(ecm, m)?)?;
    Ok(())
}

#[pyfunction(algo = "\"auto\"", verbose = "\"silent\"", threads = "None")]
#[pyo3(text_signature = "(n: int, algo: str, threads: str) -> List[int]")]
/// Factors an integer into prime factors. The result is a list
/// whose product is the input argument.
///
/// Possible values for algo are:
/// auto (default), pm1, ecm, qs, mpqs, siqs.
fn factor(
    py: Python<'_>,
    npy: &PyLong,
    algo: &str,
    verbose: &str,
    threads: Option<usize>,
) -> PyResult<Py<PyList>> {
    let verbosity =
        Verbosity::from_str(verbose).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut prefs = Preferences::default();
    prefs.threads = threads;
    prefs.verbosity = verbosity;
    let alg = Algo::from_str(algo).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let n = Uint::from_str(&npy.to_string()).map_err(|_| {
        PyValueError::new_err(format!(
            "Yamaquasi only accepts positive integers with at most 150 decimal digits"
        ))
    })?;
    if n.bits() > 512 {
        return Err(PyValueError::new_err(format!(
            "Yamaquasi only accepts positive integers with at most 150 decimal digits"
        )));
    }
    let factors = py.allow_threads(|| yamaquasi::factor(n, alg, &prefs));
    let l = PyList::empty(py);
    for f in factors {
        // Use string for conversion.
        // FIXME: this is ugly.
        let s = CString::new(f.to_string()).unwrap();
        let sptr = s.as_ptr();
        let nullptr: *mut *mut i8 = std::ptr::null::<*mut i8>() as *mut _;
        let obj = unsafe {
            let obj_ptr = PyLong_FromString(sptr, nullptr, 0);
            PyObject::from_owned_ptr(py, obj_ptr)
        };
        l.append(obj)?;
    }
    Ok(l.into())
}

#[pyfunction(verbose = "\"silent\"", threads = "None")]
fn ecm(
    py: Python<'_>,
    npy: &PyLong,
    curves: u64,
    b1: u64,
    b2: f64,
    verbose: &str,
    threads: Option<usize>,
) -> PyResult<Py<PyList>> {
    let verbosity =
        Verbosity::from_str(verbose).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let prefs = Preferences {
        threads,
        verbosity,
        ..Preferences::default()
    };
    let n = Uint::from_str(&npy.to_string()).map_err(|_| {
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
        let nullptr: *mut *mut i8 = std::ptr::null::<*mut i8>() as *mut _;
        let obj = unsafe {
            let obj_ptr = PyLong_FromString(sptr, nullptr, 0);
            PyObject::from_owned_ptr(py, obj_ptr)
        };
        l.append(obj)?;
    }
    Ok(l.into())
}
