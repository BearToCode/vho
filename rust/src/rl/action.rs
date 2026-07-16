use burn::Tensor;
use godot::prelude::*;
use probability::{prelude::*, source::Source};

use crate::{
    helicopter::Helicopter,
    rl::{Backend, DEVICE},
};

/// Dimension of the action output of the model.
pub const ACTION_DIM: usize = 4;

/// Perform a certain action in the simulation.
pub fn perform_action(u: Tensor<Backend, 2>, mut helicopter: Gd<Helicopter>) {
    let control_normalized = u
        .into_data()
        .to_vec::<f32>()
        .expect("Failed to read outputs from actor network");

    if control_normalized.len() != ACTION_DIM {
        panic!(
            "Wrong data size for control output: expected {0}, got {1}",
            ACTION_DIM,
            control_normalized.len()
        );
    }

    let mut helicopter_bind = helicopter.bind_mut();
    // Map the collective input from [-1, 1] of the model to [0, 1] for the helicopter
    let collective = (control_normalized[0] + 1.0) / 2.0;

    helicopter_bind.collective = collective;
    helicopter_bind.lateral_cyclic = control_normalized[1];
    helicopter_bind.longitudinal_cyclic = control_normalized[2];
    helicopter_bind.tail_rotor_cyclic = control_normalized[3];
}

/// Ornstein-Uhlenbeck exploration noise.
pub struct OuNoise {
    /// Running per-action-dimension noise state.
    state: [f32; ACTION_DIM],
    /// Mean-reversion rate: how strongly the state is pulled back toward zero.
    theta: f32,
    /// Volatility of the driving Gaussian process.
    sigma: f32,
}

impl OuNoise {
    pub fn new(theta: f32, sigma: f32) -> Self {
        Self {
            state: [0.0; ACTION_DIM],
            theta,
            sigma,
        }
    }

    /// Reset the internal state to zero (call at the start of each episode).
    pub fn reset(&mut self) {
        self.state = [0.0; ACTION_DIM];
    }

    /// Advance the process by `dt` and return the current noise as a `[1, ACTION_DIM]`
    /// tensor. `scale` multiplies the output so callers can decay exploration over time.
    pub fn sample<S: Source>(&mut self, dt: f32, scale: f32, source: &mut S) -> Tensor<Backend, 2> {
        let distribution = Gaussian::new(0.0, 1.0);
        let mut out = [0.0f32; ACTION_DIM];
        for i in 0..ACTION_DIM {
            let dw = distribution.sample(source) as f32 * dt.sqrt();
            self.state[i] += -self.theta * self.state[i] * dt + self.sigma * dw;
            out[i] = self.state[i] * scale;
        }

        Tensor::<Backend, 1>::from_floats(out.as_slice(), &DEVICE).reshape([1, ACTION_DIM])
    }
}
