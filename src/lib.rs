#[macro_use]
extern crate ndarray;
extern crate num_traits;
extern crate rustfft;

use ndarray::ScalarOperand;
use num_traits::Float;

mod spectrum;
pub mod windows;

pub trait StftNum: Float + rustfft::FFTnum + ScalarOperand {}

impl StftNum for f32 {}
impl StftNum for f64 {}

pub use crate::spectrum::{amplitude_to_db, normalize, power_to_db, PadMode, Stft, StftBuilder};
pub use crate::windows::{get_window, hamming, hann, Window};
