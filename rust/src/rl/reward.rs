use crate::rl::state::{AgentStateComponent, AgentStateVector};

#[allow(dead_code)]
pub fn stability_reward_function(state: &AgentStateVector) -> f32 {
    let v_x = state[AgentStateComponent::LinearVelocityX];
    let v_y = state[AgentStateComponent::LinearVelocityY];
    let v_z = state[AgentStateComponent::LinearVelocityZ];

    let w_x = state[AgentStateComponent::AngularVelocityX];
    let w_y = state[AgentStateComponent::AngularVelocityY];
    let w_z = state[AgentStateComponent::AngularVelocityZ];

    let weight_linear_velocity = 0.001;
    let weight_angular_velocity = 1.0;

    let linear_velocity_penalty =
        weight_linear_velocity * (v_x.powi(2) + v_y.powi(2) + v_z.powi(2)).sqrt();
    let angular_velocity_penalty =
        weight_angular_velocity * (w_x.powi(2) + w_y.powi(2) + w_z.powi(2)).sqrt();

    let total_penalty = linear_velocity_penalty + angular_velocity_penalty;

    let reward = -total_penalty;

    reward
}
