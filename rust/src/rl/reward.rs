use crate::rl::state::{AgentStateComponent, AgentStateVector};

pub type RewardFunction = Box<dyn Fn(&AgentStateVector, &AgentStateVector) -> f32>;
pub type FieldRewardFunction = fn(x: &AgentStateVector) -> f32;

pub fn reward_function_from_field(field_fn: FieldRewardFunction) -> RewardFunction {
    Box::new(move |x: &AgentStateVector, x_next: &AgentStateVector| field_fn(x_next) - field_fn(x))
}

#[allow(dead_code)]
pub fn fwd_stability_reward_field(x: &AgentStateVector) -> f32 {
    -x[AgentStateComponent::ForwardVelocity].powf(2.0)
}

#[allow(dead_code)]
pub fn stability_reward_field(x: &AgentStateVector) -> f32 {
    (-x[AgentStateComponent::ForwardVelocity].powf(2.0)
        - x[AgentStateComponent::LateralVelocity].powf(2.0)
        - x[AgentStateComponent::VerticalVelocity].powf(2.0))
        / 10.0
}
