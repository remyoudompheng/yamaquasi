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

use yamaquasi::{self, Algo, Preferences, Uint};

#[pymodule]
fn pymqs(_: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(factor, m)?)?;
    Ok(())
}

#[pyfunction(algo = "\"auto\"", threads = "None")]
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
    threads: Option<usize>,
) -> PyResult<Py<PyList>> {
    let prefs = Preferences {
        threads,
        ..Preferences::default()
    };
    let alg = Algo::from_str(algo).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let n = Uint::from_str(&npy.to_string()).map_err(|e| {
        PyValueError::new_err(format!(
            "Yamaquasi only accepts positive integers with at most 150 decimal digits"
        ))
    })?;
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
