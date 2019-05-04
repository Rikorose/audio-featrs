use ndarray::prelude::*;
use num_traits::FromPrimitive;

use crate::StftNum;

type Result<T> = ::std::result::Result<T, Box<::std::error::Error>>;

#[inline(always)]
fn hz2mel<T: StftNum + FromPrimitive>(f: T) -> T {
    T::from(2595.).unwrap() * (T::one() + f / T::from(700.).unwrap()).log10()
}

#[inline(always)]
fn _mel2hz<T: StftNum + FromPrimitive>(mel: T) -> T {
    T::from(700.).unwrap() * (T::from(10.).unwrap().powf(mel / T::from(2595.).unwrap()) - T::one())
}

pub fn mel<T: StftNum + FromPrimitive + ::std::fmt::Debug>(
    sr: usize,
    n_fft: usize,
    n_mels: usize,
    f_min: Option<T>,
    f_max: Option<T>,
) -> Result<Array2<T>> {
    let m_min: T = match f_min {
        None => T::zero(),
        Some(f) => hz2mel(f),
    };
    let m_max: T = hz2mel(f_max.unwrap_or_else(|| T::from(sr / 2).unwrap()));

    // Mel points
    let mut f_pts = Array::linspace(m_min, m_max, n_mels + 2);
    // Frequency points
    f_pts.mapv_inplace(_mel2hz);
    // Convert to frequency bins
    let bins = f_pts.mapv(|v| (T::from(n_fft).unwrap() * v / T::from(sr).unwrap()).floor());

    let mut fb = Array2::<T>::zeros((n_fft / 2 + 1, n_mels));
    for m in 1..=n_mels {
        let f_m_minus = bins[m - 1];
        let f_m = bins[m];
        let f_m_plus = bins[m + 1];
        if f_m_minus != f_m {
            fb.slice_mut(s![
                f_m_minus.to_usize().unwrap()..f_m.to_usize().unwrap(),
                m - 1,
            ])
            .assign(&((Array::range(f_m_minus, f_m, T::one()) - f_m_minus) / (f_m - f_m_minus)))
        }
        if f_m_plus != f_m {
            fb.slice_mut(s![
                f_m.to_usize().unwrap()..f_m_plus.to_usize().unwrap(),
                m - 1,
            ])
            .assign(&((Array::range(-f_m, -f_m_plus, -T::one()) + f_m_plus) / (f_m_plus - f_m)))
        }
    }

    Ok(fb)
}
