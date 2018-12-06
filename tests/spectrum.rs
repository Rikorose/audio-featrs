extern crate ndarray;
extern crate ndarray_rand;
extern crate num_traits;
extern crate numpy;
extern crate pyo3;
extern crate rand;

extern crate audio_featrs;

use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use num_traits::Float;
use numpy::{IntoPyArray, PyArrayDyn, TypeNum};
use pyo3::{prelude::*, types::PyDict, PyResult};
use rand::distributions::Normal;
use std::fmt::Debug;

use audio_featrs::{PadMode, StftBuilder};

type Result<T> = ::std::result::Result<T, Box<::std::error::Error>>;

fn assert_close<F>(a: &[F], b: &[F], delta: F)
where
    F: Float + Debug,
{
    assert_eq!(
        a.len(),
        b.len(),
        "assert_close() failed: Array lengths are not the same"
    );
    let mut is_close = true;
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        if x.is_finite() && y.is_finite() {
            if (x - y).abs() > delta {
                is_close = false;
                eprintln!("{:?} !~ {:?} at pos {}", x, y, i);
            }
        } else {
            if x != y {
                is_close = false;
                eprintln!("{:?} !~ {:?} at pos {}", x, y, i);
            }
        }
    }
    assert!(is_close, "assert_close() failed!");
}

fn _librosa_stft<T: Float + TypeNum>(
    py: Python,
    x: Array1<T>,
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
    window: Option<Array1<T>>,
) -> PyResult<Vec<T>> {
    let globals = PyDict::new(py);
    globals.set_item("librosa", py.import("librosa")?)?;
    globals.set_item("np", py.import("numpy")?)?;

    let locals = PyDict::new(py);
    locals.set_item("x", x.into_pyarray(py))?;
    locals.set_item("n_fft", n_fft)?;
    locals.set_item("hop_length", hop_length)?;
    locals.set_item("win_length", win_length)?;
    match window {
        Some(w) => locals.set_item("window", w.into_pyarray(py))?,
        None => locals.set_item("window", "hann")?,
    };
    let spec: &PyArrayDyn<T> = py
        .eval(
            "np.abs(librosa.stft(x, n_fft, hop_length, win_length, window, center=False),
                    dtype=x.dtype)",
            Some(&globals),
            Some(&locals),
        )?
        .extract()?;

    Ok(spec.as_slice().to_vec())
}

fn _test_stft(
    size: usize,
    n_fft: usize,
    hop_length: Option<usize>,
    win_length: Option<usize>,
    rand_win_size: Option<usize>,
    pad: PadMode,
) -> Result<()> {
    let x = Array1::random(size, Normal::new(-1., 1.));
    let builder = StftBuilder::<f64>::new()
        .n_fft(n_fft)
        .normalize(false)
        .pad_mode(pad);
    let builder = match hop_length {
        Some(h) => builder.hop_length(h),
        None => builder,
    };
    if !win_length.is_none() & !rand_win_size.is_none() {
        return Err(From::from(
            "Either win_length or rand_win_size must be None.",
        ));
    }
    let builder = match win_length {
        Some(wl) => builder.win_length(wl),
        None => builder,
    };
    let window = match rand_win_size {
        Some(ws) => Some(Array1::random(Dim(ws), Normal::new(0., 1.))),
        None => Some(Array1::from_elem(Dim(win_length.unwrap_or(n_fft)), 1.)),
    };
    let builder = match window.clone() {
        Some(w) => builder.window(w),
        None => builder,
    };

    let stft = builder.build()?;
    let spec = stft.process(x.clone().as_slice().unwrap().to_vec())?;
    println!("spec shape: {:?}", spec.shape());

    let win_length = win_length.unwrap_or(n_fft);
    let gil = Python::acquire_gil();
    let py = gil.python();
    let spec_gt = _librosa_stft(py, x, n_fft, stft.hop_length, win_length, window)
        .map_err(|e| {
            eprintln!("Error calling _librosa_stft(): {:?}", e);
            eprintln!(
                "Consider using `RUST_TEST_THREADS=1 cargo test` for a correct printing order."
            );
            e.print_and_set_sys_last_vars(gil.python()); // Print error manually
        })
        .unwrap();

    // TODO: Investigate low precision of the 0th frequency bin in comparison with librosa
    assert_close(spec.as_slice_memory_order().unwrap(), &spec_gt, 7e-5.into());
    Ok(())
}

#[test]
fn test_stft_pad_truncate_basic() {
    _test_stft(10, 10, None, None, None, PadMode::Truncate).unwrap();
}
#[test]
fn test_stft_pad_truncate_basic_2() {
    _test_stft(10, 7, Some(2), None, None, PadMode::Truncate).unwrap();
}
#[test]
fn test_stft_pad_truncate_basic_3() {
    _test_stft(4000, 1024, Some(512), None, None, PadMode::Truncate).unwrap();
}
#[test]
fn test_stft_pad_truncate_hop() {
    _test_stft(4000, 1024, Some(441), None, None, PadMode::Truncate).unwrap();
}
#[test]
fn test_stft_pad_truncate_win_length_1() {
    _test_stft(10, 7, Some(2), Some(5), None, PadMode::Truncate).unwrap();
}
#[test]
fn test_stft_pad_truncate_win_length_2() {
    _test_stft(4000, 1024, Some(512), Some(100), None, PadMode::Truncate).unwrap();
}
#[test]
fn test_stft_pad_truncate_rand_win_1() {
    _test_stft(10, 7, Some(2), None, Some(7), PadMode::Truncate).unwrap();
}
#[test]
fn test_stft_pad_truncate_rand_win_2() {
    _test_stft(4000, 1024, Some(512), None, Some(1024), PadMode::Truncate).unwrap();
}

#[test]
#[should_panic]
fn test_stft_build_assertion_1() {
    _test_stft(10, 10, None, Some(11), None, PadMode::Truncate).unwrap();
}
#[test]
#[should_panic]
fn test_stft_build_assertion_2() {
    _test_stft(10, 10, None, None, Some(11), PadMode::Truncate).unwrap();
}
#[test]
#[should_panic]
fn test_stft_build_assertion_3() {
    _test_stft(10, 10, None, None, Some(9), PadMode::Truncate).unwrap();
}
#[test]
#[should_panic]
fn test_stft_build_assertion_4() {
    _test_stft(10, 11, None, None, None, PadMode::Truncate).unwrap();
}

// TODO: test_stft_pad_center()
// Center is not supported atm because we need a reflect pad mode as provided by numpy
