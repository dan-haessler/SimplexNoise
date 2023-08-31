use image::{ImageBuffer, ImageResult};
use noise::SimplexNoise;

fn main() -> ImageResult<()> {
    let mut generator = SimplexNoise::new("noise");

    ImageBuffer::from_fn(1024, 1024, |x, y| {
        let noise = generator.noise_2d(x as f32, y as f32);
        let brightness = (((noise + 1.0) / 2.0) * 255.) as u8;
        image::Rgb([brightness, brightness, brightness])
    })
    .save("./generated/noise.png")
}
