use image::{ImageBuffer, ImageResult};
use noise::SimplexNoise;

fn main() -> ImageResult<()> {
    let mut generator = SimplexNoise::new("fractal");

    ImageBuffer::from_fn(1024, 1024, |x, y| {
        let noise = generator.fractal_2d(x as f32, y as f32, 16, 0.005, 2.0, 2.19, 0.5, 0., 255.);

        if noise < 100. {
            image::Rgb([0, 0, noise as u8])
        } else if noise >= 100. && noise < 130. {
            image::Rgb([(noise) as u8, noise as u8, 0])
        } else if noise >= 130. && noise < 160. {
            image::Rgb([(noise / 1.5) as u8, noise as u8, 0])
        } else {
            image::Rgb([(noise / 3.) as u8, noise as u8, 0])
        }
    })
    .save("./generated/terrain.png")
}
