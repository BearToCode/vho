use nalgebra::Vector3;

use crate::rl::state::{AgentStateComponent, AgentStateVector};

#[allow(dead_code)]
pub fn stability_reward_function(state: &AgentStateVector) -> f32 {
    let roll = state[AgentStateComponent::RotationAngleX];
    let pitch = state[AgentStateComponent::RotationAngleZ];

    let w_roll = 1.0;
    let w_pitch = 1.0;

    -(w_roll * roll.powi(2) + w_pitch * pitch.powi(2))
}

#[allow(dead_code)]
pub fn track_progress_reward_function(state: &AgentStateVector) -> f32 {
    let velocity = Vector3::new(
        state[AgentStateComponent::LinearVelocityX],
        state[AgentStateComponent::LinearVelocityY],
        state[AgentStateComponent::LinearVelocityZ],
    );
    let ring_direction = Vector3::new(
        state[AgentStateComponent::RingDirectionX],
        state[AgentStateComponent::RingDirectionY],
        state[AgentStateComponent::RingDirectionZ],
    );

    let reward = velocity.dot(&ring_direction);

    const MAX_TILT_RAD: f32 = 30.0 * std::f32::consts::PI / 180.0;
    const TILT_PENALTY_WEIGHT: f32 = 1.0;

    let roll = state[AgentStateComponent::RotationAngleX];
    let pitch = state[AgentStateComponent::RotationAngleZ];

    let roll_excess = (roll.abs() - MAX_TILT_RAD).max(0.0);
    let pitch_excess = (pitch.abs() - MAX_TILT_RAD).max(0.0);

    let tilt_penalty = TILT_PENALTY_WEIGHT * (roll_excess + pitch_excess);

    return reward - tilt_penalty;
}
