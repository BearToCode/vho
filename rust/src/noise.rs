//! Ornstein-Uhlenbeck noise process for continuous action-space exploration.

/// Simple xorshift64* PRNG so we don't need an external `rand` dependency.
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        // xorshift64* requires a non-zero seed.
        Self {
            state: if seed == 0 { 0x9E3779B97F4A7C15 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Uniform float in [0, 1).
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Standard normal sample via Box-Muller.
    fn next_gaussian(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-7);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

/// Ornstein-Uhlenbeck process, producing temporally-correlated noise per
/// action dimension. Correlated noise gives smoother, more useful exploration
/// for physical control tasks than i.i.d. Gaussian noise sampled every step.
pub struct OuNoise {
    /// Current noise value per action dimension.
    state: Vec<f32>,
    /// Rate of mean reversion (how fast noise pulls back toward 0).
    theta: f32,
    /// Volatility of the random component.
    sigma: f32,
    /// Multiplier applied on top of sigma, decayed externally over training.
    scale: f32,
    rng: Xorshift64,
}

impl OuNoise {
    /// Create a new OU noise process for `dim` action dimensions.
    ///
    /// `theta` ~0.15 and `sigma` ~0.2 are common defaults for DDPG.
    pub fn new(dim: usize, theta: f32, sigma: f32, seed: u64) -> Self {
        Self {
            state: vec![0.0; dim],
            theta,
            sigma,
            scale: 1.0,
            rng: Xorshift64::new(seed),
        }
    }

    /// Get the current scale of the noise.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Advance the process by one step and return the noise sample.
    pub fn sample(&mut self) -> Vec<f32> {
        for x in self.state.iter_mut() {
            let dx = self.theta * (0.0 - *x) + self.sigma * self.rng.next_gaussian();
            *x += dx;
        }
        self.state.iter().map(|x| x * self.scale).collect()
    }

    /// Set the decay scale (e.g. 1.0 at start of training, decaying to 0.0).
    pub fn set_scale(&mut self, scale: f32) {
        self.scale = scale.clamp(0.0, 1.0);
    }

    /// Reset the internal state to zero, typically called at episode start.
    pub fn reset(&mut self) {
        for x in self.state.iter_mut() {
            *x = 0.0;
        }
    }
}
