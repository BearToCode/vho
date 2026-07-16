use crate::rl::state::{AgentStateComponent, AgentStateVector};

pub fn stability_reward_function(state: &AgentStateVector) -> f32 {
    type Agent = AgentStateComponent;

    let w_x = state[Agent::AngularVelocityX];
    let w_y = state[Agent::AngularVelocityY];
    let w_z = state[Agent::AngularVelocityZ];

    let roll = state[Agent::RotationAngleX];
    let pitch = state[Agent::RotationAngleZ];

    let v_x = state[Agent::LinearVelocityX];
    let v_y = state[Agent::LinearVelocityY];
    let v_z = state[Agent::LinearVelocityZ];

    let p_x = state[Agent::PositionX];
    let p_y = state[Agent::PositionY];
    let p_z = state[Agent::PositionZ];

    let w_weight = 0.5; // angular velocity (rate) penalty
    let att_weight = 1.0; // roll/pitch attitude penalty
    let v_weight = 0.15; // linear velocity penalty
    let p_weight = 0.1; // position error penalty

    let angular_velocity_penalty = (w_x.powi(2) + w_y.powi(2) + w_z.powi(2)).sqrt();
    let attitude_penalty = (roll.powi(2) + pitch.powi(2)).sqrt();
    let velocity_penalty = (v_x.powi(2) + v_y.powi(2) + v_z.powi(2)).sqrt();
    let position_penalty = (p_x.powi(2) + p_y.powi(2) + p_z.powi(2)).sqrt();

    let total_penalty = w_weight * angular_velocity_penalty
        + att_weight * attitude_penalty
        + v_weight * velocity_penalty
        + p_weight * position_penalty;

    -total_penalty
}
