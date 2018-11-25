#[macro_use]
extern crate ndarray;
extern crate num_traits;
extern crate rustfft;

mod spectrum;
pub mod windows;

pub use spectrum::{amplitude_to_db, normalize, power_to_db, PadMode, Stft, StftBuilder};
pub use windows::{get_window, hamming, hann, Window};
