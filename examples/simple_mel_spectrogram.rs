extern crate audio_featrs;
extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;

use audio_featrs::{filters, get_window, PadMode, StftBuilder, Window};
use ndarray::prelude::*;
use ndarray_rand::{RandomExt, F32};
use rand::distributions::Normal;

type Result<T> = ::std::result::Result<T, Box<::std::error::Error>>;

fn main() -> Result<()> {
    let sr = 44100;
    let size = 50000;
    let n_fft = 1024;
    let hop = 441;
    let x = Array1::random(size, F32(Normal::new(-1., 1.)));
    let stft = StftBuilder::new()
        .n_fft(n_fft)
        .normalize(true)
        .pad_mode(PadMode::End)
        .hop_length(hop)
        .window(get_window(Window::Hann, n_fft, true))
        .build()?;

    let spec = stft.process(x.as_slice().unwrap().to_vec())?;
    let spec = spec.t(); // Transpose [F, _] -> [_, F]

    let n_mel = 10;
    println!("{:?}", spec.shape());
    let mel_fb = filters::mel::<f32>(sr, n_fft, n_mel, None, None)?;
    println!("{:?}", mel_fb.shape());
    let mel_spec = spec.dot(&mel_fb);
    println!("{:?}", mel_spec.shape());
    Ok(())
}
