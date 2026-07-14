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

    return reward;
}
