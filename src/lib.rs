// Copyright 2022 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

pub mod arith;
pub mod fbase;
pub mod matrix;
pub mod params;
pub mod relations;

// Implementations
pub mod mpqs;
pub mod qsieve;

// We need to perform modular multiplication modulo the input number.
pub type Int = arith::I1024;
pub type Uint = arith::U1024;

const DEBUG: bool = false;
