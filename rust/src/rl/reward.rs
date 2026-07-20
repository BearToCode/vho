use crate::rl::state::{AgentStateComponent, AgentStateVector};

pub struct RewardConfig {
    pub altitude_weight: f32,
    pub angular_velocity_weight: f32,
    pub linear_velocity_weight: f32,
}

pub fn stability_reward_function(state: &AgentStateVector, config: &RewardConfig) -> f32 {
    type Agent = AgentStateComponent;

    let w_x = state[Agent::AngularVelocityX];
    let w_y = state[Agent::AngularVelocityY];
    let w_z = state[Agent::AngularVelocityZ];
    let w = (w_x.powi(2) + w_y.powi(2) + w_z.powi(2)).sqrt();

    let v_x = state[Agent::LinearVelocityX];
    let v_y = state[Agent::LinearVelocityY];
    let v_z = state[Agent::LinearVelocityZ];
    let v = (v_x.powi(2) + v_y.powi(2) + v_z.powi(2)).sqrt();

    let altitude = state[Agent::PositionErrorY];

    let compute_reward = |value: f32, range: f32| -> f32 {
        (1.0 / (std::f32::consts::E * value / range).cosh()).powi(2)
    };

    let angular_velocity_reward = compute_reward(w, 0.5) * config.angular_velocity_weight;
    let linear_velocity_reward = compute_reward(v, 2.0) * config.linear_velocity_weight;
    let altitude_reward = compute_reward(altitude, 1.0) * config.altitude_weight;

    angular_velocity_reward * linear_velocity_reward * altitude_reward
}
