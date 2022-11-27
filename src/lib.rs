pub mod arith;
pub mod poly;
pub mod params;

// We need to perform modular multiplication modulo the input number.
pub type Int = arith::I1024;
pub type Uint = arith::U1024;
