use burn::Tensor;

use crate::rl::{Backend, DEVICE, action::ACTION_DIM, adhdp::ADHDPBatch, state::STATE_DIM};

/// A single stored transition, kept CPU-side so sampling is cheap and does not
/// pin GPU memory.
struct Transition {
    x: Vec<f32>,
    u: Vec<f32>,
    reward: f32,
    x_next: Vec<f32>,
    /// 1.0 if `x_next` is terminal (no bootstrap), 0.0 otherwise.
    done: f32,
}

/// Fixed-capacity experience replay buffer with uniform random sampling.
///
/// A DDPG-style critic bootstraps off its own predictions; training on the single,
/// highly-correlated online transition produced each frame makes that bootstrap
/// diverge (the "deadly triad"). Storing transitions and sampling random
/// minibatches decorrelates the updates, which is what actually stabilizes
/// learning.
pub struct ReplayBuffer {
    capacity: usize,
    data: Vec<Transition>,
    /// Next write index, used as a ring buffer once at capacity.
    position: usize,
    /// xorshift64 RNG state for index sampling (self-contained, deterministic).
    rng: u64,
}

impl ReplayBuffer {
    pub fn new(capacity: usize, seed: u64) -> Self {
        let capacity = capacity.max(1);
        Self {
            capacity,
            data: Vec::with_capacity(capacity.min(1 << 16)),
            position: 0,
            rng: seed | 1, // xorshift state must be non-zero
        }
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Store one transition, overwriting the oldest once at capacity.
    pub fn push(&mut self, x: &[f32], u: &[f32], reward: f32, x_next: &[f32], done: f32) {
        let transition = Transition {
            x: x.to_vec(),
            u: u.to_vec(),
            reward,
            x_next: x_next.to_vec(),
            done,
        };

        if self.data.len() < self.capacity {
            self.data.push(transition);
        } else {
            self.data[self.position] = transition;
        }
        self.position = (self.position + 1) % self.capacity;
    }

    /// Next xorshift64 pseudo-random value.
    fn next_u64(&mut self) -> u64 {
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng = x;
        x
    }

    /// Uniformly sample a minibatch, returning batched tensors ready for training.
    /// Returns `None` until at least `batch_size` transitions are stored, which
    /// doubles as a short warmup period.
    pub fn sample(&mut self, batch_size: usize) -> Option<ADHDPBatch> {
        let n = self.data.len();
        if batch_size == 0 || n < batch_size {
            return None;
        }

        let mut xs = Vec::with_capacity(batch_size * STATE_DIM);
        let mut us = Vec::with_capacity(batch_size * ACTION_DIM);
        let mut rewards = Vec::with_capacity(batch_size);
        let mut x_nexts = Vec::with_capacity(batch_size * STATE_DIM);
        let mut dones = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            let idx = (self.next_u64() % n as u64) as usize;
            let t = &self.data[idx];
            xs.extend_from_slice(&t.x);
            us.extend_from_slice(&t.u);
            rewards.push(t.reward);
            x_nexts.extend_from_slice(&t.x_next);
            dones.push(t.done);
        }

        let x = Tensor::<Backend, 1>::from_floats(xs.as_slice(), &DEVICE)
            .reshape([batch_size, STATE_DIM]);
        let u = Tensor::<Backend, 1>::from_floats(us.as_slice(), &DEVICE)
            .reshape([batch_size, ACTION_DIM]);
        let reward = Tensor::<Backend, 1>::from_floats(rewards.as_slice(), &DEVICE)
            .reshape([batch_size, 1]);
        let x_next = Tensor::<Backend, 1>::from_floats(x_nexts.as_slice(), &DEVICE)
            .reshape([batch_size, STATE_DIM]);
        let done = Tensor::<Backend, 1>::from_floats(dones.as_slice(), &DEVICE)
            .reshape([batch_size, 1]);

        Some(ADHDPBatch {
            x,
            u,
            reward,
            x_next,
            done,
        })
    }
}
