extern crate hound;
extern crate itertools;
extern crate ndarray;
extern crate png;

extern crate audio_featrs;

use std::env;
use std::fs::File;
use std::io;
use std::path::Path;

use itertools::Itertools;
use ndarray::prelude::*;

// Stft implementation
use audio_featrs::*;

// To use encoder.set()
use png::HasParameters;

type Result<T> = ::std::result::Result<T, Box<::std::error::Error>>;

fn read_wav<'a, R: io::Read>(
    reader: &'a mut hound::WavReader<R>,
    mono: bool,
) -> Box<Iterator<Item = f32> + 'a> {
    let n_channels = reader.spec().channels;
    let max_val = 2_i32.pow(reader.spec().bits_per_sample as u32 - 1) as f32;
    println!("{}", max_val);
    let it: Box<Iterator<Item = f32> + 'a> = match reader.spec().sample_format {
        hound::SampleFormat::Float => Box::new(reader.samples::<f32>().map(|s| s.unwrap() as f32)),
        hound::SampleFormat::Int => Box::new(
            reader
                .samples::<i32>()
                .map(move |s| s.unwrap() as f32 / max_val),
        ),
    };
    if mono {
        Box::new(it.batching(move |it_v| match it_v.next() {
            None => None,
            Some(v) => {
                let v_r: f32 = it_v.take(n_channels as usize - 1).sum();
                Some((v + v_r) / (n_channels as f32))
            }
        }))
    } else {
        it
    }
}

fn main() -> Result<()> {
    // Check args
    let args = env::args();
    if args.len() <= 1 {
        return Err(From::from("Usage: read_wav audio_file_1.wav audio_file_2.wav ..."));
    }
    // Compute the spectrogram for all files given on the command line and save them as png
    for fname in args.skip(1) {
        println!("Processing file {}", fname);
        let p = Path::new(&fname);
        if !p.exists() {
            return Err(From::from("Input audio file not found"));
        }
        let mut reader = hound::WavReader::open(&p.as_os_str())?;
        let sr = reader.spec().sample_rate as usize;
        println!(
            "Reading file with {} channels, sample rate {} and length {}",
            reader.spec().channels,
            reader.spec().sample_rate,
            reader.len(),
        );
        let samples = read_wav(&mut reader, true).collect::<Vec<f32>>();

        // Spectrogram params
        let n_fft = 2048;
        let n_hop = 441;
        let min_level_db = Some(-80);
        let ref_level_db = Some(20);
        let n_mel = 150;
        let f_min = Some(512.);
        let f_max = Some(16000.);
        let stft = StftBuilder::new()
            .n_fft(n_fft)
            .hop_length(n_hop)
            .pad_mode(PadMode::Center)
            .build()?;
        let spec = stft.process(samples)?;
        let mut mel_spec = spec.t().dot(&filters::mel(sr, n_fft, n_mel, f_min, f_max)?);
        amplitude_to_db(&mut mel_spec);
        normalize(&mut mel_spec, min_level_db, ref_level_db);

        // Convert to image buffer
        mel_spec.invert_axis(Axis(1));
        let img = mel_spec.t().iter().map(|v| ((1. - v) * 255.0) as u8).collect::<Vec<u8>>();

        // Setup png writer
        let file = File::create(p.with_extension("png"))?;
        let ref mut w = io::BufWriter::new(file);
        let mut encoder = png::Encoder::new(w, mel_spec.shape()[0] as u32, mel_spec.shape()[1] as u32);
        encoder.set(png::ColorType::Grayscale);
        let mut writer = encoder.write_header()?;
        writer.write_image_data(&img)?;
    }
    Ok(())
}
