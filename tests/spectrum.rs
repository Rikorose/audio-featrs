extern crate audio_featrs;
extern crate ndarray;
extern crate ndarray_rand;
extern crate num_traits;
extern crate numpy;
extern crate pyo3;
extern crate rand;

use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use num_traits::Float;
use numpy::{PyArrayDyn, IntoPyArray};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::distributions::Range;
use std::fmt::Debug;
use std::io::Error;

use audio_featrs::*;

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

fn _librosa_stft(
    x: Array1<f32>,
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
    window: Option<Array1<f32>>,
) -> Result<Vec<f32>, Error> {
    let gil = Python::acquire_gil();
    let py = gil.python();
    let locals = PyDict::new(py);
    let librosa = py.import("librosa")?;
    let np = py.import("numpy")?;

    locals.set_item("librosa", librosa)?;
    locals.set_item("np", np)?;
    locals.set_item("x", x.into_pyarray(py))?;
    locals.set_item("n_fft", n_fft)?;
    locals.set_item("hop_length", hop_length)?;
    locals.set_item("win_length", win_length)?;
    match window {
        Some(w) => locals.set_item("window", w.into_pyarray(py))?,
        None => locals.set_item("window", "hann")?,
    };
    let spec = match py.eval(
        "np.abs(librosa.stft(x, n_fft, hop_length, win_length, window, center=False))",
        None,
        Some(locals),
    ) {
        Ok(s) => s.cast_as::<PyArrayDyn<f32>>().unwrap(),
        Err(err) => panic!("Error calling librosa.stft: {:?}", err),
    };

    Ok(spec.as_slice().to_vec())
}

fn _test_stft(
    size: usize,
    n_fft: usize,
    hop_length: Option<usize>,
    win_length: Option<usize>,
    rand_win_size: Option<usize>,
    pad: PadMode,
) -> Result<(), ()> {
    let x = Array1::<f32>::random(Dim(size), Range::new(-1., 1.));
    let builder = StftBuilder::new()
        .n_fft(n_fft)
        .normalize(false)
        .pad_mode(pad);
    let builder = match hop_length {
        Some(h) => builder.hop_length(h),
        None => builder,
    };
    assert!(
        win_length.is_none() || rand_win_size.is_none(),
        "Either win_length or rand_win_size must be None."
    );
    let builder = match win_length {
        Some(wl) => builder.win_length(wl),
        None => builder,
    };
    let window = match rand_win_size {
        Some(ws) => Some(Array1::<f32>::random(Dim(ws), Range::new(0., 1.))),
        None => None,
    };
    let builder = match window.clone() {
        Some(w) => builder.window(w),
        None => builder,
    };
    let stft = builder.build();
    let spec = stft.process(x.as_slice().unwrap().to_vec());

    let spec_gt = _librosa_stft(
        x,
        n_fft,
        hop_length.unwrap_or(stft.hop_length),
        win_length.unwrap_or(n_fft),
        window,
    )
    .unwrap();

    assert_close(spec.as_slice_memory_order().unwrap(), &spec_gt, 1e-5.into());
    Ok(())
}

#[test]
fn test_stft_pad_truncate() {
    _test_stft(10, 10, None, None, None, PadMode::Truncate).unwrap();

    _test_stft(10, 7, Some(2), None, None, PadMode::Truncate).unwrap();
    _test_stft(4000, 1024, None, None, None, PadMode::Truncate).unwrap();

    _test_stft(4000, 1024, Some(441), None, None, PadMode::Truncate).unwrap();

    _test_stft(10, 7, Some(2), Some(5), None, PadMode::Truncate).unwrap();
    _test_stft(4000, 1024, None, Some(100), None, PadMode::Truncate).unwrap();

    _test_stft(10, 7, Some(2), None, Some(7), PadMode::Truncate).unwrap();
    _test_stft(4000, 1024, None, None, Some(1024), PadMode::Truncate).unwrap();
}
// TODO: test_stft_pad_center()
// Center is not supported atm because we need a reflect pad mode as provided by numpy
