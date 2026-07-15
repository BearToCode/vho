use std::ops::{Index, IndexMut};

use burn::Tensor;
use godot::{global::tanh, prelude::*};
use nalgebra::SVector;

use crate::{
    game::Game,
    rl::{Backend, DEVICE},
};

/// Dimension of the state input of the model.
pub const STATE_DIM: usize = 12;

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
    /// [-]     Current ring direction x component.
    RingDirectionX,
    /// [-]     Current ring direction y component.
    RingDirectionY,
    /// [-]     Current ring direction z component.
    RingDirectionZ,
    /// [m]     Current ring distance along the specified direction.
    RingDistance,
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

/// Settings for normalization of the state.
pub struct StateNormalizationConfig {
    pub angular_velocity_scale: f32,
    pub linear_velocity_scale: f32,
    pub angle_scale: f32,
    pub position_scale: f32,
}

/// Get the (un-normalized) agent state vector.
pub fn get_agent_state(game: Gd<Game>) -> AgentStateVector {
    // Get all necessary references
    let game_bind = game.bind();

    let helicopter = game_bind.helicopter.clone().unwrap();

    let track = game_bind.track.clone().unwrap();
    let current_ring = track.bind().current_ring().unwrap();

    // Helicopter transform data
    let global_to_local = helicopter.get_transform().basis.inverse();

    let helicopter_position = helicopter.get_global_position();
    let helicopter_rotation = helicopter.get_rotation();

    let helicopter_linear_velocity = helicopter.get_linear_velocity();
    let helicopter_angular_velocity = helicopter.get_angular_velocity();

    let helicopter_local_linear_velocity = global_to_local * helicopter_linear_velocity;
    let helicopter_local_angular_velocity = global_to_local * helicopter_angular_velocity;

    // Ring transform data
    let ring_position = current_ring.get_global_position();
    let ring_local_relative_position = global_to_local * (ring_position - helicopter_position);
    let ring_direction = ring_local_relative_position.normalized();

    let ring_distance = ring_local_relative_position.length();

    // Copy data onto agent state vector
    type Agent = AgentStateComponent;
    let mut agent_state: nalgebra::Matrix<
        f32,
        nalgebra::Const<12>,
        nalgebra::Const<1>,
        nalgebra::ArrayStorage<f32, 12, 1>,
    > = AgentStateVector::zeros();
    agent_state[Agent::LinearVelocityX] = helicopter_local_linear_velocity.x;
    agent_state[Agent::LinearVelocityY] = helicopter_local_linear_velocity.y;
    agent_state[Agent::LinearVelocityZ] = helicopter_local_linear_velocity.z;
    agent_state[Agent::AngularVelocityX] = helicopter_local_angular_velocity.x;
    agent_state[Agent::AngularVelocityY] = helicopter_local_angular_velocity.y;
    agent_state[Agent::AngularVelocityZ] = helicopter_local_angular_velocity.z;
    agent_state[Agent::RotationAngleX] = helicopter_rotation.x;
    agent_state[Agent::RotationAngleZ] = helicopter_rotation.z;
    agent_state[Agent::RingDirectionX] = ring_direction.x;
    agent_state[Agent::RingDirectionY] = ring_direction.y;
    agent_state[Agent::RingDirectionZ] = ring_direction.z;
    agent_state[Agent::RingDistance] = ring_distance;

    return agent_state;
}

/// Get the normalized state for RL.
pub fn normalize_state(
    agent_state: &AgentStateVector,
    config: &StateNormalizationConfig,
) -> Tensor<Backend, 2> {
    type Agent = AgentStateComponent;

    const Y_MULTIPLIER: f32 = 10.0;

    let ring_distance_normalized =
        tanh((agent_state[Agent::RingDistance] * config.position_scale) as f64) as f32;

    let normalized = Tensor::<Backend, 1>::from_data(
        [
            agent_state[Agent::LinearVelocityX] * config.linear_velocity_scale,
            agent_state[Agent::LinearVelocityY] * config.linear_velocity_scale * Y_MULTIPLIER,
            agent_state[Agent::LinearVelocityZ] * config.linear_velocity_scale,
            agent_state[Agent::AngularVelocityX] * config.angular_velocity_scale,
            agent_state[Agent::AngularVelocityY] * config.angular_velocity_scale,
            agent_state[Agent::AngularVelocityZ] * config.angular_velocity_scale,
            agent_state[Agent::RotationAngleX] * config.angle_scale,
            agent_state[Agent::RotationAngleZ] * config.angle_scale,
            agent_state[Agent::RingDirectionX],
            agent_state[Agent::RingDirectionY] * Y_MULTIPLIER,
            agent_state[Agent::RingDirectionZ],
            ring_distance_normalized,
        ],
        &DEVICE,
    )
    .reshape([1, STATE_DIM]);

    // godot_print!("State: {}", agent_state);
    // godot_print!("Normalized: {}", normalized);

    return normalized;
}
