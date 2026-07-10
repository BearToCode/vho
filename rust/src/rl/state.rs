use std::ops::{Index, IndexMut};

use burn::Tensor;
use godot::prelude::*;
use nalgebra::SVector;

use crate::{
    game::Game,
    helicopter::{HelicopterStateComponent, ft_to_m},
    rl::{Backend, DEVICE},
};

/// Dimension of the state input of the model.
pub const STATE_DIM: usize = 14;

pub type AgentStateVector = SVector<f32, STATE_DIM>;

/// Un-normalized agent state components.
/// All properties are in a body-fixed reference frame.
pub enum AgentStateComponent {
    /// [m/s]   Helicopter forward velocity.
    ForwardVelocity = 0,
    /// [m/s]   Helicopter lateral velocity.
    LateralVelocity,
    /// [m/s]   Helicopter vertical velocity.
    VerticalVelocity,
    /// [rad/s] Helicopter pitch rate.
    PitchRate,
    /// [rad/s] Helicopter roll rate.
    RollRate,
    /// [rad/s] Helicopter yaw rate.
    YawRate,
    /// [rad/s] Helicopter pitch angle.
    PitchAngle,
    /// [rad/s] Helicopter roll angle.
    RollAngle,
    /// [m]     Current ring position x component.
    CurrentRingPositionX,
    /// [m]     Current ring position y component.
    CurrentRingPositionY,
    /// [m]     Current ring position z component.
    CurrentRingPositionZ,
    /// [m]     Next ring position x component.
    NextRingPositionX,
    /// [m]     Next ring position y component.
    NextRingPositionY,
    /// [m]     Next ring position z component.
    NextRingPositionZ,
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
    let helicopter_bind = helicopter.bind();

    let track = game_bind.track.clone().unwrap();
    let current_ring = track.bind().current_ring().unwrap();
    let next_ring = track.bind().next_ring();

    let mut agent_state = AgentStateVector::zeros();
    let helicopter_state = helicopter_bind.get_state_vector();

    // Helicopter necessary transform data
    let helicopter_position = helicopter.get_global_position();
    let global_to_local = helicopter.get_transform().basis.inverse();

    // Second 3 components: Helicopter position relative to fist ring, in local reference frame
    let current_ring_position = current_ring.get_global_position();
    let current_ring_relative_position =
        global_to_local * (current_ring_position - helicopter_position);

    // Third 3 components: Helicopter position relative to second ring, in local reference frame
    let next_ring_relative_position = if let Some(next_ring) = next_ring {
        let next_ring_position = next_ring.get_global_position();
        global_to_local * (next_ring_position - helicopter_position)
    } else {
        // Use same location as current ring if it's the last one
        current_ring_relative_position
    };

    // Copy data onto agent state vector
    type Agent = AgentStateComponent;
    type Helicopter = HelicopterStateComponent;
    agent_state[Agent::ForwardVelocity] = ft_to_m(helicopter_state[Helicopter::U]);
    agent_state[Agent::LateralVelocity] = ft_to_m(helicopter_state[Helicopter::V]);
    agent_state[Agent::VerticalVelocity] = ft_to_m(helicopter_state[Helicopter::W]);
    agent_state[Agent::RollRate] = helicopter_state[Helicopter::P];
    agent_state[Agent::PitchRate] = helicopter_state[Helicopter::Q];
    agent_state[Agent::YawRate] = helicopter_state[Helicopter::R];
    agent_state[Agent::RollAngle] = helicopter_state[Helicopter::Phi];
    agent_state[Agent::PitchAngle] = helicopter_state[Helicopter::Theta];
    agent_state[Agent::CurrentRingPositionX] = current_ring_relative_position.x;
    agent_state[Agent::CurrentRingPositionY] = current_ring_relative_position.y;
    agent_state[Agent::CurrentRingPositionZ] = current_ring_relative_position.z;
    agent_state[Agent::NextRingPositionX] = next_ring_relative_position.x;
    agent_state[Agent::NextRingPositionY] = next_ring_relative_position.y;
    agent_state[Agent::NextRingPositionZ] = next_ring_relative_position.z;

    return agent_state;
}

/// Get the normalized state for RL.
pub fn normalize_state(
    agent_state: &AgentStateVector,
    config: &StateNormalizationConfig,
) -> Tensor<Backend, 2> {
    type Agent = AgentStateComponent;

    return Tensor::<Backend, 1>::from_data(
        [
            agent_state[Agent::ForwardVelocity] * config.linear_velocity_scale,
            agent_state[Agent::LateralVelocity] * config.linear_velocity_scale,
            agent_state[Agent::VerticalVelocity] * config.linear_velocity_scale,
            agent_state[Agent::RollRate] * config.angular_velocity_scale,
            agent_state[Agent::PitchRate] * config.angular_velocity_scale,
            agent_state[Agent::YawRate] * config.angular_velocity_scale,
            agent_state[Agent::RollAngle] * config.angle_scale,
            agent_state[Agent::PitchAngle] * config.angle_scale,
            agent_state[Agent::CurrentRingPositionX] * config.position_scale,
            agent_state[Agent::CurrentRingPositionY] * config.position_scale,
            agent_state[Agent::CurrentRingPositionZ] * config.position_scale,
            agent_state[Agent::NextRingPositionX] * config.position_scale,
            agent_state[Agent::NextRingPositionY] * config.position_scale,
            agent_state[Agent::NextRingPositionZ] * config.position_scale,
        ],
        &DEVICE,
    )
    .reshape([1, STATE_DIM]);
}
