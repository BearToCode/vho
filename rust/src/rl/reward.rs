use crate::rl::state::{AgentStateComponent, AgentStateVector};

// At the range value, reward is around 1% of the maximum
pub struct RewardConfig {
    pub roll_range: f32,
    pub pitch_range: f32,
    pub angular_velocity_range: f32,
    pub linear_velocity_range: f32,
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

    let roll = state[Agent::RotationAngleX];
    let pitch = state[Agent::RotationAngleZ];

    // For now, only consider velocity and orientation
    let compute_reward = |value: f32, range: f32| -> f32 {
        (1.0 / (std::f32::consts::E * value / range).cosh()).powi(2)
    };

    let roll_reward = compute_reward(roll, config.roll_range);
    let pitch_reward = compute_reward(pitch, config.pitch_range);
    let angular_velocity_reward = compute_reward(w, config.angular_velocity_range);
    let linear_velocity_reward = compute_reward(v, config.linear_velocity_range);

    // godot_print!(
    //     "roll_reward: {}, pitch_reward: {}, angular_velocity_reward: {}, linear_velocity_reward: {}",
    //     roll_reward,
    //     pitch_reward,
    //     angular_velocity_reward,
    //     linear_velocity_reward
    // );

    let sum_reward = (roll_reward + pitch_reward + angular_velocity_reward + linear_velocity_reward) * 0.25;
    
    sum_reward
}
