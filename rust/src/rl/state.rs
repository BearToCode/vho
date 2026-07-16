use std::ops::{Index, IndexMut};

use burn::Tensor;
use godot::prelude::*;
use nalgebra::SVector;

use crate::{
    game::Game,
    rl::{Backend, DEVICE},
};

/// Dimension of the state input of the model.
pub const STATE_DIM: usize = 13;

pub type AgentStateVector = SVector<f32, STATE_DIM>;

/// Un-normalized agent state components.
/// All properties are in a body-fixed reference frame.
pub enum AgentStateComponent {
    /// [m/s]   Helicopter local x velocity.
    LinearVelocityX = 0,
    /// [m/s]   Helicopter local y velocity.
    LinearVelocityY,
    /// [m/s]   Helicopter local z velocity.
    LinearVelocityZ,
    /// [rad/s] Helicopter local x angular velocity.
    AngularVelocityX,
    /// [rad/s] Helicopter local y angular velocity.
    AngularVelocityY,
    /// [rad/s] Helicopter local z angular velocity.
    AngularVelocityZ,
    /// [rad/s] Helicopter roll angle.
    RotationAngleX,
    /// [rad/s] Helicopter pitch angle.
    RotationAngleZ,
    /// [m] Position error (current - initial) along global x.
    PositionX,
    /// [m] Position error (current - initial) along global y.
    PositionY,
    /// [m] Position error (current - initial) along global z.
    PositionZ,
    /// [rad] Longitudinal flap angle.
    LongitudinalFlapAngle,
    /// [rad] Lateral flap angle.
    LateralFlapAngle,
}

impl Index<AgentStateComponent> for AgentStateVector {
    type Output = f32;
    fn index(&self, index: AgentStateComponent) -> &Self::Output {
        &self[index as usize]
    }
}

impl IndexMut<AgentStateComponent> for AgentStateVector {
    fn index_mut(&mut self, index: AgentStateComponent) -> &mut Self::Output {
        return &mut self[index as usize];
    }
}

/// Get the (un-normalized) agent state vector.
pub fn get_agent_state(game: Gd<Game>) -> AgentStateVector {
    // Get all necessary references
    let game_bind = game.bind();

    let helicopter = game_bind.helicopter.clone().unwrap();
    let helicopter_bind = helicopter.bind();

    // Helicopter transform data
    let global_to_local = helicopter.get_transform().basis.inverse();

    let helicopter_rotation = helicopter.get_rotation();

    // Position error from the target hover point (the pose at scene start).
    let position_error = helicopter.get_global_position() - game_bind.helicopter_initial_position();

    let helicopter_linear_velocity = helicopter.get_linear_velocity();
    let helicopter_angular_velocity = helicopter.get_angular_velocity();

    let helicopter_local_linear_velocity = global_to_local * helicopter_linear_velocity;
    let helicopter_local_angular_velocity = global_to_local * helicopter_angular_velocity;

    // Copy data onto agent state vector
    type Agent = AgentStateComponent;
    let mut agent_state = AgentStateVector::zeros();
    agent_state[Agent::LinearVelocityX] = helicopter_local_linear_velocity.x;
    agent_state[Agent::LinearVelocityY] = helicopter_local_linear_velocity.y;
    agent_state[Agent::LinearVelocityZ] = helicopter_local_linear_velocity.z;
    agent_state[Agent::AngularVelocityX] = helicopter_local_angular_velocity.x;
    agent_state[Agent::AngularVelocityY] = helicopter_local_angular_velocity.y;
    agent_state[Agent::AngularVelocityZ] = helicopter_local_angular_velocity.z;
    agent_state[Agent::RotationAngleX] = helicopter_rotation.x;
    agent_state[Agent::RotationAngleZ] = helicopter_rotation.z;
    agent_state[Agent::PositionX] = position_error.x;
    agent_state[Agent::PositionY] = position_error.y;
    agent_state[Agent::PositionZ] = position_error.z;
    agent_state[Agent::LongitudinalFlapAngle] = helicopter_bind.lon_flapping;
    agent_state[Agent::LateralFlapAngle] = helicopter_bind.lat_flapping;

    return agent_state;
}

/// Running statistics for state normalization.
#[derive(Debug, Copy, Clone)]
struct OnlineStatistic {
    count: f64,
    mean: f64,
    m2: f64, // sum of squared deviations
}

impl OnlineStatistic {
    fn update(&mut self, x: f64) {
        self.count += 1.0;
        let delta = x - self.mean;
        self.mean += delta / self.count;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    fn std(&self) -> f64 {
        (self.m2 / self.count.max(1.0)).sqrt()
    }

    fn normalize(&self, x: f64) -> f64 {
        (x - self.mean) / (self.std() + 1e-8)
    }
}

pub struct OnlineStateNormalization {
    stats: [OnlineStatistic; STATE_DIM],
}

impl OnlineStateNormalization {
    pub fn new() -> Self {
        Self {
            stats: [OnlineStatistic {
                count: 0.0,
                mean: 0.0,
                m2: 0.0,
            }; STATE_DIM],
        }
    }

    /// Update the running statistics with a new state vector.
    pub fn update(&mut self, state: &AgentStateVector) {
        for i in 0..STATE_DIM {
            self.stats[i].update(state[i] as f64);
        }
    }

    /// Normalize the state vector using the running statistics.
    pub fn normalize(&self, state: &AgentStateVector) -> Tensor<Backend, 2> {
        let mut normalized = AgentStateVector::zeros();
        for i in 0..STATE_DIM {
            normalized[i] = self.stats[i].normalize(state[i] as f64) as f32;
        }

        Tensor::<Backend, 1>::from_data(normalized.as_slice(), &DEVICE).reshape([1, STATE_DIM])
    }

    pub fn save(&self, output_path: &str) {
        let mut serialized = Vec::with_capacity(STATE_DIM * 3);
        for stat in &self.stats {
            serialized.push(stat.mean as f32);
            serialized.push(stat.std() as f32);
            serialized.push(stat.count as f32);
        }

        let bytes: Vec<u8> = serialized
            .iter()
            .flat_map(|f| f.to_le_bytes().to_vec())
            .collect();

        std::fs::write(output_path, bytes).expect("Failed to write normalization model file");
    }

    pub fn load(&mut self, file_path: &str) {
        let data = std::fs::read(file_path)
            .expect("Failed to read normalization model file")
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<f32>>();

        assert_eq!(data.len(), STATE_DIM * 3);

        for i in 0..STATE_DIM {
            self.stats[i].mean = data[i * 3] as f64;
            self.stats[i].m2 = (data[i * 3 + 1] as f64).powi(2) * (data[i * 3 + 2] as f64);
            self.stats[i].count = data[i * 3 + 2] as f64;
        }
    }
}
