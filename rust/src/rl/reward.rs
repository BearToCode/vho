use crate::rl::state::{AgentStateComponent, AgentStateVector};

pub struct RewardConfig {
    /// Granted every step the helicopter has not failed.
    pub alive_bonus: f32,
    /// Weight on the angular velocity penalty.
    pub angular_velocity_weight: f32,
    /// Weight on the position error penalty.
    pub position_weight: f32,
    /// Position error past which the episode has failed, in meters. 0 disables.
    pub max_position_error: f32,
    /// Roll or pitch past which the episode has failed, in radians. 0 disables.
    pub max_tilt: f32,
}

/// Distance from the target hover point.
fn position_error(state: &AgentStateVector) -> f32 {
    type Agent = AgentStateComponent;

    (state[Agent::PositionX].powi(2)
        + state[Agent::PositionY].powi(2)
        + state[Agent::PositionZ].powi(2))
    .sqrt()
}

/// Magnitude of the angular velocity.
fn angular_speed(state: &AgentStateVector) -> f32 {
    type Agent = AgentStateComponent;

    (state[Agent::AngularVelocityX].powi(2)
        + state[Agent::AngularVelocityY].powi(2)
        + state[Agent::AngularVelocityZ].powi(2))
    .sqrt()
}

/// Whether the helicopter has failed, ending the episode.
///
/// This is a true terminal state, unlike hitting the episode time limit: the transition
/// into it must not bootstrap, because there is no future to have a value.
pub fn is_failure(state: &AgentStateVector, config: &RewardConfig) -> bool {
    type Agent = AgentStateComponent;

    if config.max_position_error > 0.0 && position_error(state) > config.max_position_error {
        return true;
    }

    if config.max_tilt > 0.0 {
        let roll = state[Agent::RotationAngleX].abs();
        let pitch = state[Agent::RotationAngleZ].abs();

        if roll > config.max_tilt || pitch > config.max_tilt {
            return true;
        }
    }

    false
}

/// Reward for one step.
///
/// The alive bonus is what makes the failure terminal meaningful, and the two only work
/// together. With an all-negative reward, an agent that can end the episode is better
/// off failing immediately to stop accumulating penalty, so it would learn to crash on
/// purpose. With the bonus, surviving beats stopping, and within surviving, holding
/// position still beats drifting. Conversely the bonus alone does nothing without the
/// terminal: every episode would run the same number of steps regardless, so it would
/// add a constant to every state's value and leave the optimal policy untouched.
///
/// It follows that `alive_bonus` must exceed the penalty of a policy worth keeping,
/// otherwise failing is still the better option.
pub fn stability_reward_function(state: &AgentStateVector, config: &RewardConfig) -> f32 {
    let penalty = angular_speed(state) * config.angular_velocity_weight
        + position_error(state) * config.position_weight;

    config.alive_bonus - penalty
}
