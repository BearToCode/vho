use crate::rl::state::{AgentStateComponent, AgentStateVector};

pub fn stability_reward_function(state: &AgentStateVector) -> f32 {
    type Agent = AgentStateComponent;

    let w_x = state[Agent::AngularVelocityX];
    let w_y = state[Agent::AngularVelocityY];
    let w_z = state[Agent::AngularVelocityZ];
    let w = (w_x.powi(2) + w_y.powi(2) + w_z.powi(2)).sqrt();

    let roll = state[Agent::RotationAngleX];
    let pitch = state[Agent::RotationAngleZ];

    let v_x = state[Agent::LinearVelocityX];
    let v_y = state[Agent::LinearVelocityY];
    let v_z = state[Agent::LinearVelocityZ];
    let v = (v_x.powi(2) + v_y.powi(2) + v_z.powi(2)).sqrt();

    // let p_x = state[Agent::PositionX];
    // let p_y = state[Agent::PositionY];
    // let p_z = state[Agent::PositionZ];

    // At the range value, reward is around 1% of the maximum
    const ROLL_RANGE: f32 = std::f32::consts::PI / 8.0; // rad
    const PITCH_RANGE: f32 = std::f32::consts::PI / 8.0; // rad
    const ANGULAR_VELOCITY_RANGE: f32 = 0.3 * std::f32::consts::PI; // rad/s
    const LINEAR_VELOCITY_RANGE: f32 = 3.0; // m/s

    // For now, only consider velocity and orientation
    let compute_reward = |value: f32, range: f32| -> f32 {
        (1.0 / (std::f32::consts::E * value / range).cosh()).powi(2)
    };

    let roll_reward = compute_reward(roll, ROLL_RANGE);
    let pitch_reward = compute_reward(pitch, PITCH_RANGE);
    let angular_velocity_reward = 3.0 * compute_reward(w, ANGULAR_VELOCITY_RANGE);
    let linear_velocity_reward = compute_reward(v, LINEAR_VELOCITY_RANGE);

    // godot_print!(
    //     "roll_reward: {}, pitch_reward: {}, angular_velocity_reward: {}, linear_velocity_reward: {}",
    //     roll_reward,
    //     pitch_reward,
    //     angular_velocity_reward,
    //     linear_velocity_reward
    // );

    let reward = roll_reward + pitch_reward + angular_velocity_reward + linear_velocity_reward;

    reward
}
