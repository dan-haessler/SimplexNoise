use image::{ImageBuffer, ImageResult};
use noise::SimplexNoise;

fn main() -> ImageResult<()> {
    let mut generator = SimplexNoise::new("fractal");

    ImageBuffer::from_fn(1024, 1024, |x, y| {
        let noise =
            generator.fractal_2d(x as f32, y as f32, 16, 0.005, 2.0, 2.19, 0.5, 0., 255.) as u8;
        image::Rgb([noise, noise, noise])
    })
    .save("./generated/fractal.png")
}
