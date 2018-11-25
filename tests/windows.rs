extern crate audio_featrs;
extern crate num_traits;

use audio_featrs::windows::*;
use num_traits::Float;
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
fn test_hamming() {
    assert_close(
        hamming(6, false).as_slice().unwrap(),
        &[0.08, 0.31, 0.77, 1.0, 0.77, 0.31],
        1e-10,
    );
    assert_close(
        hamming(7, false).as_slice().unwrap(),
        &[
            0.08,
            0.2531946911449826,
            0.6423596296199047,
            0.9544456792351128,
            0.9544456792351128,
            0.6423596296199047,
            0.2531946911449826,
        ],
        1e-10,
    );
    assert_close(
        hamming(6, true).as_slice().unwrap(),
        &[
            0.08,
            0.3978521825875242,
            0.9121478174124757,
            0.9121478174124757,
            0.3978521825875242,
            0.08,
        ],
        1e-10,
    );
    assert_close(
        hamming(7, true).as_slice().unwrap(),
        &[0.08, 0.31, 0.77, 1.0, 0.77, 0.31, 0.08],
        1e-10,
    );
}

#[test]
fn test_hann() {
    assert_close(
        hann(6, false).as_slice().unwrap(),
        &[0.0, 0.25, 0.75, 1.0, 0.75, 0.25],
        1e-10,
    );
    assert_close(
        hann(7, false).as_slice().unwrap(),
        &[
            0.0,
            0.1882550990706332,
            0.6112604669781572,
            0.9504844339512095,
            0.9504844339512095,
            0.6112604669781572,
            0.1882550990706332,
        ],
        1e-10,
    );
    assert_close(
        hann(6, true).as_slice().unwrap(),
        &[
            0.0,
            0.3454915028125263,
            0.9045084971874737,
            0.9045084971874737,
            0.3454915028125263,
            0.0,
        ],
        1e-10,
    );
    assert_close(
        hann(7, true).as_slice().unwrap(),
        &[0.0, 0.25, 0.75, 1.0, 0.75, 0.25, 0.0],
        1e-10,
    );
}
