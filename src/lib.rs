use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

const GRADIENT_1D: [f32; 3] = [0., 1., -1.];

const GRADIENT_2D: [(f32, f32); 8] = [
    (0., 1.),
    (0., -1.),
    (1., 0.),
    (1., 1.),
    (1., -1.),
    (-1., 0.),
    (-1., 1.),
    (-1., -1.),
];

const GRADIENT_3D: [(f32, f32, f32); 12] = [
    (1., 0., 1.),
    (1., 0., -1.),
    (1., 1., 0.),
    (1., -1., 0.),
    (-1., 0., 1.),
    (-1., 0., -1.),
    (-1., 1., 0.),
    (-1., -1., 0.),
    (0., 1., 1.),
    (0., 1., -1.),
    (0., -1., 1.),
    (0., -1., -1.),
];

// Linear congruential generator
#[derive(Clone, Debug)]
pub struct RNG {
    seed: u64,
    current: u64,
}

impl RNG {
    pub fn new(seed: impl Hash) -> Self {
        let seed = Self::hash(seed);
        Self {
            current: seed,
            seed: seed,
        }
    }

    fn hash(seed: impl Hash) -> u64 {
        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        hasher.finish()
    }

    pub fn reseed(&mut self, seed: impl Hash) {
        self.seed = Self::hash(seed);
        self.current = self.seed;
    }

    pub fn generate(&mut self) -> u64 {
        self.current =
            (134775813u64.wrapping_mul(self.current.wrapping_add(1))) & (std::u32::MAX as u64);
        self.current
    }

    pub fn permutation_table(&mut self) -> [u8; 256] {
        let mut numbers: Vec<u8> = (0..=255).collect::<Vec<u8>>();
        let mut table: [u8; 256] = [0; 256];
        for i in 0..numbers.len() {
            table[i] = numbers.swap_remove(self.generate() as usize % numbers.len());
        }
        table
    }
}

#[derive(Clone, Debug)]
pub struct SimplexNoise {
    rng: RNG,
    permutation: [u8; 256],
}

impl SimplexNoise {
    pub fn new(seed: impl Hash) -> Self {
        let mut rng = RNG::new(seed);
        Self {
            rng: rng.clone(),
            permutation: rng.permutation_table(),
        }
    }

    /// Reseeding the RNG and create a new permutation table.
    ///
    /// ```
    /// use noise::SimplexNoise;
    ///
    /// // Use "a" as initial seed.
    /// let mut generator = SimplexNoise::new("a");
    /// let noise_a: f32 = generator.noise_1d(0.0);
    ///
    /// // Use "b" as seed.
    /// generator.reseed("b");
    /// let noise_b: f32 = generator.noise_1d(0.0);
    ///
    /// // Use "a" again as seed.
    /// generator.reseed("a");
    /// let noise_c: f32 = generator.noise_1d(0.0);
    ///
    /// // Validate
    /// assert!(noise_a != noise_b);
    /// assert!(noise_a == noise_c);
    /// ```
    pub fn reseed(&mut self, seed: impl Hash) {
        self.rng.reseed(seed);
        self.permutation = self.rng.permutation_table();
    }

    fn hash<T: From<u8>>(&self, index: usize) -> T {
        return T::from(self.permutation[index % 256]);
    }

    fn corner_1d(random: usize, x: f32) -> f32 {
        let mut t = 1.0 * x * x;
        t *= t;
        t * t * (x * GRADIENT_1D[random % GRADIENT_1D.len()])
    }

    fn corner_2d(random: usize, x: f32, y: f32) -> f32 {
        let mut t: f32 = 0.5 - x * x - y * y;
        if t < 0.0 {
            return 0.0;
        }
        t *= t;
        let (g_x, g_y) = GRADIENT_2D[random % GRADIENT_2D.len()];
        t * t * (x * g_x + y * g_y)
    }

    fn corner_3d(random: usize, x: f32, y: f32, z: f32) -> f32 {
        let mut t: f32 = 0.6 - x * x - y * y - z * z;
        if t < 0.0 {
            return 0.0;
        }
        let (g_x, g_y, g_z) = GRADIENT_3D[random % GRADIENT_3D.len()];
        t *= t;
        t * t * (x * g_x + y * g_y + z * g_z)
    }

    /// 1D Simplex Noise.
    ///
    /// ```
    /// use noise::SimplexNoise;
    ///
    /// let mut generator = SimplexNoise::new("test");
    /// let noise: f32 = generator.noise_1d(0.0);
    ///
    /// // Validate value is within [-1.0; 1.0].
    /// assert!(noise >= -1.0);
    /// assert!(noise <= 1.0);
    /// ```
    pub fn noise_1d(&mut self, x: f32) -> f32 {
        // No skewing needed, determine simplex.
        let i0 = x.floor();
        let i1 = i0 + 1.;

        // Distances to corner
        let x0 = x - i0;
        let x1 = x0 - 1.0;

        // Get the hashed gradient corner contributions.
        let n0 = Self::corner_1d(self.hash(i0 as usize), x0);
        let n1 = Self::corner_1d(self.hash(i1 as usize), x1);

        // Scaled to fit [-1, 1]
        n0 + n1 / 8.0
    }

    /// 2D Simplex Noise.
    ///
    /// ```
    /// use noise::SimplexNoise;
    ///
    /// let mut generator = SimplexNoise::new("test");
    /// let noise: f32 = generator.noise_2d(0.0, 0.0);
    ///
    /// // Validate value is within [-1.0; 1.0].
    /// assert!(noise >= -1.0);
    /// assert!(noise <= 1.0);
    /// ```
    pub fn noise_2d(&mut self, x: f32, y: f32) -> f32 {
        // Skewing/Unskewing factors for 2d simplex.
        const F2: f32 = 0.3660254037844385965883020617184229195117950439453125; // (f64::sqrt(3.0) - 1.0) * 0.5;
        const G2: f32 = 0.2113248654051871344705659794271923601627349853515625; // (3.0 - f64::sqrt(3.0)) / 6.0;

        // Skew input to get the indices of the simplex.
        let s: f32 = (x + y) * F2;
        let i: f32 = (x + s).floor();
        let j: f32 = (y + s).floor();

        // Unskew back to origin space.
        let t: f32 = (i + j) * G2;
        let x0: f32 = x - (i - t);
        let y0: f32 = y - (j - t);

        // Determine simplex
        let i1: usize;
        let j1: usize;

        if x0 > y0 {
            // Lower triangle (0,0)->(1,0)->(1,1)
            i1 = 1;
            j1 = 0;
        } else {
            // Upper triangle (0,0)->(0,1)->(1,1)
            i1 = 0;
            j1 = 1;
        }

        // Offsets for middle corner in unskewed coords.
        let x1: f32 = x0 - i1 as f32 + G2;
        let y1: f32 = y0 - j1 as f32 + G2;

        // Offsets for last corner in unskewed coords
        let x2: f32 = x0 - 1.0 + 2.0 * G2;
        let y2: f32 = y0 - 1.0 + 2.0 * G2;

        let ii = i as usize;
        let jj = j as usize;

        // Get the hashed gradient corner contributions.
        let gi0 = self.hash(ii + self.hash::<usize>(jj));
        let gi1 = self.hash(ii + i1 + self.hash::<usize>(jj + j1));
        let gi2 = self.hash(ii + self.hash::<usize>(jj + 1) + 1);

        let n0: f32 = Self::corner_2d(gi0, x0, y0);
        let n1: f32 = Self::corner_2d(gi1, x1, y1);
        let n2: f32 = Self::corner_2d(gi2, x2, y2);

        // Scaled to fit [-1, 1]
        46.76454 * (n0 + n1 + n2)
    }

    /// 3D Simplex Noise.
    ///
    /// ```
    /// use noise::SimplexNoise;
    ///
    /// let mut generator = SimplexNoise::new("test");
    /// let noise: f32 = generator.noise_3d(0.0, 0.0, 0.0);
    ///
    /// // Validate value is within [-1.0; 1.0].
    /// assert!(noise >= -1.0);
    /// assert!(noise <= 1.0);
    /// ```
    pub fn noise_3d(&mut self, x: f32, y: f32, z: f32) -> f32 {
        // Skewing/Unskewing factors for 3d simplex.
        const F3: f32 = 1.0 / 3.0;
        const G3: f32 = 1.0 / 6.0;

        // Skew input to get the indices of the simplex.
        let s = (x + y + z) / 3.0;
        let i = (x + s).floor();
        let j = (y + s).floor();
        let k = (z + s).floor();

        // Unskew back to origin space.
        let t = (i + j + k) / 6.0;
        let x0 = x - (i - t);
        let y0 = y - (j - t);
        let z0 = z - (k - t);

        // Determine simplex
        let (mut i1, mut j1, mut k1, mut i2, mut j2, mut k2) = (
            0 as usize, 0 as usize, 0 as usize, 0 as usize, 0 as usize, 0 as usize,
        );
        if x0 >= y0 {
            if y0 >= z0 {
                (i1, i2, j2) = (1, 1, 1);
            } else if x0 >= z0 {
                (i1, i2, k2) = (1, 1, 1);
            } else {
                (j1, i2, k2) = (1, 1, 1);
            }
        } else {
            if y0 < z0 {
                (k1, j2, k2) = (1, 1, 1);
            } else if x0 < z0 {
                (j1, j2, k2) = (1, 1, 1);
            } else {
                (j1, i2, j2) = (1, 1, 1);
            }
        }

        let x1: f32 = x0 - (i1 as f32) + G3;
        let y1: f32 = y0 - (j1 as f32) + G3;
        let z1: f32 = z0 - (k1 as f32) + G3;
        let x2: f32 = x0 - (i2 as f32) + 2.0 * G3;
        let y2: f32 = y0 - (j2 as f32) + 2.0 * G3;
        let z2: f32 = z0 - (k2 as f32) + 2.0 * G3;
        let x3: f32 = x0 - 1.0 + 3.0 * G3;
        let y3: f32 = y0 - 1.0 + 3.0 * G3;
        let z3: f32 = z0 - 1.0 + 3.0 * G3;
        
        let ii = i as usize;
        let jj = j as usize;
        let kk = k as usize;

        // Get the hashed gradient corner contributions.
        let gi0 = self.hash(ii + self.hash::<usize>(jj + self.hash::<usize>(kk)));
        let gi1 = self.hash(ii + i1 + self.hash::<usize>(jj + j1 + self.hash::<usize>(kk + k1)));
        let gi2 = self.hash(ii + i2 + self.hash::<usize>(jj + j2 + self.hash::<usize>(kk + k2)));
        let gi3 = self.hash(ii + 1 + self.hash::<usize>(jj + 1 + self.hash::<usize>(kk + 1)));

        let n0 = Self::corner_3d(gi0, x0, y0, z0);
        let n1 = Self::corner_3d(gi1, x1, y1, z1);
        let n2 = Self::corner_3d(gi2, x2, y2, z2);
        let n3 = Self::corner_3d(gi3, x3, y3, z3);

        // Scaled to fit [-1, 1]
        36.86678 * (n0 + n1 + n2 + n3)
    }

    // Fractal Brownian motion for any noise function.
    fn fbm(
        mut noise_fn: impl FnMut(f32) -> f32,
        octaves: usize,
        mut frequency: f32,
        mut amplitude: f32,
        lacunarity: f32,
        gain: f32,
        low: f32,
        high: f32,
    ) -> f32 {
        let mut noise: f32 = 0.0;
        let mut denom: f32 = 0.0;

        for _ in 0..octaves {
            noise += amplitude * noise_fn(frequency);
            denom += amplitude;

            frequency *= lacunarity;
            amplitude *= gain;
        }
        noise /= denom;
        noise = noise * (high - low) / 2. + (high + low) / 2.;
        noise
    }

    /// Fractal Brownian motion with 1D Simplex Noise.
    ///
    /// ```
    /// use noise::SimplexNoise;
    ///
    /// let mut generator = SimplexNoise::new("test");
    /// let fbm_noise: f32 = generator.fractal_1d(0.0f32, 8, 0.005, 1.0, 2.0, 0.5, 0.0, 255.0);
    ///
    /// // Validate value is within [low; high].
    /// assert!(fbm_noise >= 0.0f32);
    /// assert!(fbm_noise <= 255.0f32);
    /// ```
    pub fn fractal_1d(
        &mut self,
        x: f32,
        octaves: usize,
        frequency: f32,
        amplitude: f32,
        lacunarity: f32,
        gain: f32,
        low: f32,
        high: f32,
    ) -> f32 {
        Self::fbm(
            |f: f32| self.noise_1d(x * f),
            octaves,
            frequency,
            amplitude,
            lacunarity,
            gain,
            low,
            high,
        )
    }

    /// Fractal Brownian motion with 2D Simplex Noise.
    ///
    /// ```
    /// use noise::SimplexNoise;
    ///
    /// let mut generator = SimplexNoise::new("test");
    /// let fbm_noise: f32 = generator.fractal_2d(0.0, 0.0, 8, 0.005, 1.0, 2.0, 0.5, 0.0, 255.0);
    ///
    /// // Validate value is within [low; high].
    /// assert!(fbm_noise >= 0.0);
    /// assert!(fbm_noise <= 255.0);
    /// ```
    pub fn fractal_2d(
        &mut self,
        x: f32,
        y: f32,
        octaves: usize,
        frequency: f32,
        amplitude: f32,
        lacunarity: f32,
        gain: f32,
        low: f32,
        high: f32,
    ) -> f32 {
        Self::fbm(
            |f: f32| self.noise_2d(x * f, y * f),
            octaves,
            frequency,
            amplitude,
            lacunarity,
            gain,
            low,
            high,
        )
    }

    /// Fractal Brownian motion with 3D Simplex Noise.
    ///
    /// ```
    /// use noise::SimplexNoise;
    ///
    /// let mut generator = SimplexNoise::new("test");
    /// let fbm_noise: f32 = generator.fractal_3d(0.0, 0.0, 0.0, 8, 0.005, 1.0, 2.0, 0.5, 0.0, 255.0);
    ///
    /// // Validate value is within [low; high].
    /// assert!(fbm_noise >= 0.0);
    /// assert!(fbm_noise <= 255.0);
    /// ```
    pub fn fractal_3d(
        &mut self,
        x: f32,
        y: f32,
        z: f32,
        octaves: usize,
        frequency: f32,
        amplitude: f32,
        lacunarity: f32,
        gain: f32,
        low: f32,
        high: f32,
    ) -> f32 {
        Self::fbm(
            |f: f32| self.noise_3d(x * f, y * f, z * f),
            octaves,
            frequency,
            amplitude,
            lacunarity,
            gain,
            low,
            high,
        )
    }
}
