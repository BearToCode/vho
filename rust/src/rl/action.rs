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

pub fn get_noise<S: Source>(noise_std: f32, source: &mut S) -> Tensor<Backend, 2> {
    // Tensor of gaussian noise with mean 0 and standard deviation `noise_std`
    let distribution = Gaussian::new(0.0, noise_std as f64);
    let noise: Vec<f32> = (0..ACTION_DIM)
        .map(|_| distribution.sample(source) as f32)
        .collect();

    Tensor::<Backend, 1>::from_floats(noise.as_slice(), &DEVICE).reshape([1, ACTION_DIM])
}
