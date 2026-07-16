use crate::rl::state::{AgentStateComponent, AgentStateVector};

#[allow(dead_code)]
pub fn stability_reward_function(state: &AgentStateVector) -> f32 {
    type Agent = AgentStateComponent;

    let w_x = state[Agent::AngularVelocityX];
    let w_y = state[Agent::AngularVelocityY];
    let w_z = state[Agent::AngularVelocityZ];

    let p_x = state[Agent::PositionX];
    let p_y = state[Agent::PositionY];
    let p_z = state[Agent::PositionZ];

    let w_weight = 1.0; // Weight for angular velocity penalty
    let p_weight = 0.1; // Weight for position penalty

    let angular_velocity_penalty = (w_x.powi(2) + w_y.powi(2) + w_z.powi(2)).sqrt();
    let position_penalty = (p_x.powi(2) + p_y.powi(2) + p_z.powi(2)).sqrt();

    let total_penalty = angular_velocity_penalty * w_weight + position_penalty * p_weight;

    let reward = -total_penalty;

    reward
}
