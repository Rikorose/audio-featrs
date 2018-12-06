extern crate audio_featrs;
extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;

use audio_featrs::{get_window, PadMode, StftBuilder, Window};
use ndarray::prelude::*;
use ndarray_rand::{RandomExt, F32};
use rand::distributions::Normal;

type Result<T> = ::std::result::Result<T, Box<::std::error::Error>>;

fn main() -> Result<()> {
    let size = 10;
    let n_fft = 6;
    let x = Array1::random(size, F32(Normal::new(-1., 1.)));
    println!("Signal: {:?}", x);
    let stft = StftBuilder::new()
        .n_fft(n_fft)
        .normalize(true)
        .pad_mode(PadMode::Truncate)
        .hop_length(n_fft / 2)
        .window(get_window(Window::Hann, n_fft, true))
        .build()?;

    let spec = stft.process(x.as_slice().unwrap().to_vec());
    println!("Spectrogram: {:?}", spec);
    Ok(())
}
