use std::cmp::min;
use std::iter;
use std::option::Option;
use std::sync::Arc;

use ndarray::prelude::*;
use ndarray::Zip;
use num_traits::Float;

use rustfft::num_complex::Complex;
use rustfft::{FFTplanner, FFT};

use windows;
use StftNum;

type Result<T> = ::std::result::Result<T, Box<::std::error::Error>>;

pub enum PadMode {
    Truncate,
    End,
    Center,
}

impl Default for PadMode {
    fn default() -> PadMode {
        PadMode::Truncate
    }
}

pub struct Stft<T> {
    pub n_fft: usize,
    pub hop_length: usize,
    pub pad_mode: PadMode,
    pub window: Array1<T>,
    pub normalization: T,
    fft: Arc<FFT<T>>,
}

pub struct StftBuilder<T> {
    n_fft: Option<usize>,
    hop_length: Option<usize>,
    win_length: Option<usize>,
    pad_mode: Option<PadMode>,
    window: Option<Array1<T>>,
    window_named: Option<windows::Window>,
    normalize: Option<bool>,
}

impl<T: StftNum> StftBuilder<T> {
    pub fn new() -> StftBuilder<T> {
        StftBuilder {
            n_fft: None,
            hop_length: None,
            win_length: None,
            pad_mode: None,
            window: None,
            window_named: None,
            normalize: None,
        }
    }
    pub fn n_fft(mut self, n_fft: usize) -> StftBuilder<T> {
        self.n_fft = Some(n_fft);
        self
    }
    pub fn hop_length(mut self, hop_length: usize) -> StftBuilder<T> {
        self.hop_length = Some(hop_length);
        self
    }
    pub fn win_length(mut self, win_length: usize) -> StftBuilder<T> {
        self.win_length = Some(win_length);
        self
    }
    pub fn pad_mode(mut self, pad_mode: PadMode) -> StftBuilder<T> {
        self.pad_mode = Some(pad_mode);
        self
    }
    pub fn window(mut self, window: Array1<T>) -> StftBuilder<T> {
        self.window = Some(window);
        self
    }
    pub fn window_from_vec(mut self, window: Vec<T>) -> StftBuilder<T> {
        self.window = Some(Array1::<T>::from_vec(window));
        self
    }
    pub fn window_named(mut self, window: windows::Window) -> StftBuilder<T> {
        self.window_named = Some(window);
        self
    }
    pub fn normalize(mut self, normalize: bool) -> StftBuilder<T> {
        self.normalize = Some(normalize);
        self
    }
    pub fn build(self) -> Result<Stft<T>> {
        let n_fft = self.n_fft.unwrap_or(2048);
        let win_length = self.win_length.unwrap_or(n_fft);
        let hop_length = self.hop_length.unwrap_or(win_length / 4);
        if win_length > n_fft {
            return Err(From::from("STFT win_length must be <= n_fft"));
        }
        let pad_mode = self.pad_mode.unwrap_or_default();
        let mut window = Array1::<T>::zeros(n_fft);
        let w_start = (n_fft - win_length) / 2;
        window
            .slice_mut(s![w_start..w_start + win_length])
            .assign(&match self.window {
                None => {
                    windows::get_window(self.window_named.unwrap_or_default(), win_length, true)
                }
                Some(w) => {
                    if w.len() != win_length {
                        return Err(From::from("Window length must be equal to n_fft."));
                    }
                    w
                }
            });
        let normalization = match self.normalize.unwrap_or(true) {
            true => window.map(|v| v.powi(2)).scalar_sum().sqrt(),
            false => T::one(),
        };
        let mut planner = FFTplanner::new(false);
        let fft = planner.plan_fft(n_fft);
        Ok(Stft {
            n_fft: n_fft,
            hop_length: hop_length,
            pad_mode: pad_mode,
            window: window,
            fft: fft,
            normalization: normalization,
        })
    }
}

impl<T: StftNum + std::fmt::Debug + std::fmt::Display> Stft<T> {
    fn pad(&self, mut signal: Vec<T>) -> Vec<T> {
        match self.pad_mode {
            PadMode::Truncate => signal,
            PadMode::End => {
                let n_pad = self.hop_length - (signal.len() - self.n_fft) % self.hop_length;
                let new_len = signal.len() + n_pad;
                signal.reserve(new_len);
                for elem in iter::repeat(T::zero()).take(n_pad) {
                    signal.push(elem);
                }
                signal
            }
            PadMode::Center => {
                let n_pad = self.hop_length - (signal.len() - self.n_fft) % self.hop_length;
                let n_pad_front = n_pad / 2;
                let n_pad_end = n_pad - n_pad_front;
                let new_len = signal.len() + n_pad_end;
                signal.reserve(new_len);
                signal.splice(0..0, iter::repeat(T::zero()).take(n_pad_front));
                for elem in iter::repeat(T::zero()).take(n_pad_end) {
                    signal.push(elem);
                }
                signal
            }
        }
    }

    pub fn process(&self, signal: Vec<T>) -> Result<Array2<T>> {
        let signal = self.pad(signal);
        let n_frames = 1 + (signal.len() - self.n_fft) / self.hop_length;

        let mut fft_input = Array1::<Complex<T>>::zeros(self.n_fft);
        let mut fft_output = Array1::<Complex<T>>::zeros(self.n_fft);

        let n_freqs = self.n_fft / 2 + 1;
        let mut output = Array2::<T>::zeros((n_freqs, n_frames).f());

        for frame in 0..n_frames {
            // Get slice of input audio multiply it with the window
            // and copy it to the input buffer
            let start = frame * self.hop_length;
            let end = min(signal.len(), start + self.n_fft);
            Zip::from(&mut fft_input)
                .and(&self.window)
                .and(&signal[start..end])
                .apply(|i, &w, &a| {
                    *i = Complex::<T>::new(w * a, T::zero());
                });

            // Perform FFT
            self.fft.process(
                &mut fft_input.as_slice_mut().ok_or("Stft input is None")?,
                &mut fft_output.as_slice_mut().ok_or("Stft output is None")?,
            );

            // And copy onesided to the output buffer
            output.slice_mut(s![.., frame]).assign(
                &fft_output
                    .slice_mut(s![0..n_freqs])
                    .map(|v| (v / self.normalization).norm() as T),
            );
        }
        Ok(output)
    }
}

pub fn normalize<T>(spec: &mut Array2<T>, min_level_db: Option<i16>, ref_level_db: Option<i16>)
where
    T: Clone + Float,
    i16: Into<T>,
{
    // Ideally this would be done using a clamp() function
    let min_level_db = min_level_db.unwrap_or(-100).into();
    let ref_level_db = ref_level_db.unwrap_or_default().into();
    spec.mapv_inplace(|v| {
        ((v - ref_level_db - min_level_db) / -min_level_db)
            .min(T::one())
            .max(T::zero())
    });
}

pub fn amplitude_to_db<T: Float>(spec: &mut Array2<T>) {
    let multiplier = T::from(20).unwrap();
    spec.mapv_inplace(|v| multiplier * v.log10());
}

pub fn power_to_db<T: Float>(spec: &mut Array2<T>) {
    let multiplier = T::from(10).unwrap();
    spec.mapv_inplace(|v| multiplier * v.log10());
}
