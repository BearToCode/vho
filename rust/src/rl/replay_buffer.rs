use burn::Tensor;
use rand::seq::index::sample;

use crate::rl::{
    Backend, DEVICE,
    action::ACTION_DIM,
    adhdp::ADHDPTrainData,
    state::{AgentStateVector, OnlineStateNormalization, STATE_DIM},
};

/// One (state, action, reward, next_state) transition collected from the environment.
/// States are stored un-normalized; see `ReplayBuffer::sample`.
#[derive(Clone, Copy)]
pub struct Transition {
    pub state: AgentStateVector,
    pub action: [f32; ACTION_DIM],
    pub reward: f32,
    pub next_state: AgentStateVector,
    /// Whether `next_state` is a failure. Reaching the episode time limit is not a
    /// failure: the world did not end, we just stopped watching, so it still bootstraps.
    pub terminal: bool,
}

/// Fixed-capacity ring buffer of transitions, sampled from uniformly at random to
/// break the temporal correlation between consecutive environment steps.
pub struct ReplayBuffer {
    capacity: usize,
    data: Vec<Transition>,
    /// Index the next `push` will write to (wraps around once full).
    next_idx: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            data: Vec::with_capacity(capacity),
            next_idx: 0,
        }
    }

    pub fn push(&mut self, transition: Transition) {
        if self.data.len() < self.capacity {
            self.data.push(transition);
        } else {
            self.data[self.next_idx] = transition;
        }
        self.next_idx = (self.next_idx + 1) % self.capacity;
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Draw a random batch (without replacement) as batched tensors. Returns `None` if
    /// there aren't enough transitions yet.
    ///
    /// States are normalized here rather than at collection time, so that every state
    /// in the returned batch shares the same (current) normalization statistics.
    pub fn sample(
        &self,
        batch_size: usize,
        normalization: &OnlineStateNormalization,
    ) -> Option<ADHDPTrainData> {
        if batch_size == 0 || self.data.len() < batch_size {
            return None;
        }

        let mut rng = rand::rng();
        let indices = sample(&mut rng, self.data.len(), batch_size);

        let mut states = Vec::with_capacity(batch_size * STATE_DIM);
        let mut actions = Vec::with_capacity(batch_size * ACTION_DIM);
        let mut rewards = Vec::with_capacity(batch_size);
        let mut next_states = Vec::with_capacity(batch_size * STATE_DIM);
        let mut dones: Vec<f32> = Vec::with_capacity(batch_size);

        for i in indices.iter() {
            let t = &self.data[i];
            normalization.normalize_into(&t.state, &mut states);
            actions.extend_from_slice(&t.action);
            rewards.push(t.reward);
            normalization.normalize_into(&t.next_state, &mut next_states);
            dones.push(if t.terminal { 1.0 } else { 0.0 });
        }

        let x = Tensor::<Backend, 1>::from_data(states.as_slice(), &DEVICE)
            .reshape([batch_size, STATE_DIM]);
        let u = Tensor::<Backend, 1>::from_data(actions.as_slice(), &DEVICE)
            .reshape([batch_size, ACTION_DIM]);
        let reward = Tensor::<Backend, 1>::from_data(rewards.as_slice(), &DEVICE)
            .reshape([batch_size, 1]);
        let x_next = Tensor::<Backend, 1>::from_data(next_states.as_slice(), &DEVICE)
            .reshape([batch_size, STATE_DIM]);
        let done =
            Tensor::<Backend, 1>::from_data(dones.as_slice(), &DEVICE).reshape([batch_size, 1]);

        Some(ADHDPTrainData {
            x,
            u,
            reward,
            x_next,
            done,
        })
    }
}
