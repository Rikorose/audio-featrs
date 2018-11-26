# audio-featrs
Audio feature calculation written in Rust.

## Example

```rust
extern crate audio_featrs;
extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;

use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use rand::distributions::Range;
use audio_featrs::{StftBuilder, PadMode, get_window, Window};

fn main() {
    let size = 10;
    let x = Array1::<f32>::random(Dim(size), Range::new(-1., 1.));
    println!("Signal: {:?}", x);
    let stft = StftBuilder::new()
        .n_fft(size)
        .normalize(true)
        .pad_mode(PadMode::Truncate)
        .hop_length(441)
        .window(get_window(Window::Hann, size, true))
        .build();

    let spec = stft.process(x.as_slice().unwrap().to_vec());
    println!("Spectrogram: {:?}", spec);
}
```

## Credits

  * Librosa
  * Scipy

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
