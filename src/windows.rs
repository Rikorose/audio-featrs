use std::f64::consts::PI;

use ndarray::prelude::*;
use ndarray::ScalarOperand;
use num_traits::{Float, FromPrimitive, Zero};

pub enum Window {
    Hann,
    Hamming,
}

impl Default for Window {
    fn default() -> Window {
        Window::Hann
    }
}

#[inline(always)]
pub fn get_window<T>(window: Window, size: usize, fftbins: bool) -> Array1<T>
where
    T: Float + FromPrimitive + ScalarOperand,
{
    let sym = !fftbins;
    match window {
        Window::Hann => hann::<T>(size, sym),
        Window::Hamming => hamming::<T>(size, sym),
    }
}

#[inline(always)]
fn _extend(size: usize, sym: bool) -> (usize, bool) {
    match sym {
        false => (size + 1, true),
        true => (size, false),
    }
}

#[inline(always)]
fn _maybe_truncate<T>(w: &mut Array1<T>, needed: bool) {
    if needed {
        w.slice_inplace(s![..-1]);
    }
}

fn general_cosine<T>(size: usize, a: &[T], sym: bool) -> Array1<T>
where
    T: Float + FromPrimitive + Zero + Clone + ScalarOperand,
{
    let (size, needs_trunk) = _extend(size, sym);
    let pi = T::from(PI).unwrap();
    let fac = Array1::<T>::linspace(-pi, pi, size);
    let mut w = Array1::<T>::zeros(size);
    for k in 0..a.len() {
        let tmp: Array1<T> = fac.map(|v| T::cos(*v * T::from(k).unwrap())) * T::from(a[k]).unwrap();
        w = w + tmp;
        // TODO:
        //w += &tmp;
    }
    _maybe_truncate(&mut w, needs_trunk);
    w
}

fn general_hamming<T>(size: usize, alpha: T, sym: bool) -> Array1<T>
where
    T: Float + FromPrimitive + ScalarOperand,
{
    general_cosine::<T>(size, &[alpha, T::from(1.0).unwrap() - alpha], sym)
}

pub fn hamming<T>(size: usize, sym: bool) -> Array1<T>
where
    T: Float + FromPrimitive + ScalarOperand,
{
    general_hamming::<T>(size, T::from(0.54).unwrap(), sym)
}

pub fn hann<T>(size: usize, sym: bool) -> Array1<T>
where
    T: Float + FromPrimitive + ScalarOperand,
{
    general_hamming::<T>(size, T::from(0.5).unwrap(), sym)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Debug;

    fn assert_close<F>(a: &[F], b: &[F], delta: F)
    where
        F: Float + Debug,
    {
        assert_eq!(a.len(), b.len());
        for (&x, &y) in a.iter().zip(b) {
            if x.is_finite() && y.is_finite() {
                assert!((x - y).abs() <= delta, "{:?} !~ {:?}", x, y);
            } else {
                assert!(x == y, "{:?} !~ {:?}", x, y);
            }
        }
    }

    #[test]
    fn test_general_cosine() {
        assert_close(
            general_cosine(5, &[0.5, 0.3, 0.2], true)
                .as_slice()
                .unwrap(),
            &[0.4, 0.3, 1.0, 0.3, 0.4],
            1e-10,
        );
        assert_close(
            general_cosine(4, &[0.5, 0.3, 0.2], false)
                .as_slice()
                .unwrap(),
            &[0.4, 0.3, 1.0, 0.3],
            1e-10,
        );
    }

    #[test]
    fn test_general_hamming() {
        assert_close(
            general_hamming(5, 0.7, true).as_slice().unwrap(),
            &[0.4, 0.7, 1.0, 0.7, 0.4],
            1e-10,
        );
        assert_close(
            general_hamming(5, 0.75, false).as_slice().unwrap(),
            &[0.5, 0.6727457514, 0.9522542486, 0.9522542486, 0.6727457514],
            1e-10,
        );
        assert_close(
            general_hamming(6, 0.75, true).as_slice().unwrap(),
            &[
                0.5,
                0.6727457514,
                0.9522542486,
                0.9522542486,
                0.6727457514,
                0.5,
            ],
            1e-10,
        );
    }
}
