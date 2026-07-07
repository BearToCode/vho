//! Fixed-capacity replay buffer storing transitions as plain Vec<f32> data
//! (not tensors) so storage is cheap and backend-agnostic. Tensors are only
//! constructed when sampling a training batch.

/// A single stored transition. `next_state` is `None` for terminal transitions.
struct Transition {
    state: Vec<f32>,
    action: Vec<f32>,
    reward: f32,
    next_state: Option<Vec<f32>>,
}

/// Simple xorshift64* PRNG, reused here to avoid an external `rand` dependency.
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9E3779B97F4A7C15 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Uniform index in [0, bound).
    fn next_index(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }
}

/// A fixed-capacity ring-buffer of transitions with uniform random sampling.
pub struct ReplayBuffer {
    transitions: Vec<Transition>,
    capacity: usize,
    /// Index where the next transition will be written (ring buffer cursor).
    write_pos: usize,
    rng: Xorshift64,
}

/// A sampled minibatch, still as flat Vec<f32> data (state_dim/action_dim
/// implicit from the stored transitions). Caller reshapes into tensors.
pub struct Batch {
    pub states: Vec<f32>,
    pub actions: Vec<f32>,
    pub rewards: Vec<f32>,
    /// Flattened next-states for non-terminal transitions, in order.
    pub next_states: Vec<f32>,
    /// For each sampled transition, whether it's terminal (true) or not.
    pub is_terminal: Vec<bool>,
    pub batch_size: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize, seed: u64) -> Self {
        Self {
            transitions: Vec::with_capacity(capacity),
            capacity,
            write_pos: 0,
            rng: Xorshift64::new(seed),
        }
    }

    pub fn len(&self) -> usize {
        self.transitions.len()
    }

    /// Store a transition, overwriting the oldest one once at capacity.
    pub fn push(
        &mut self,
        state: Vec<f32>,
        action: Vec<f32>,
        reward: f32,
        next_state: Option<Vec<f32>>,
    ) {
        let transition = Transition {
            state,
            action,
            reward,
            next_state,
        };

        if self.transitions.len() < self.capacity {
            self.transitions.push(transition);
        } else {
            self.transitions[self.write_pos] = transition;
        }
        self.write_pos = (self.write_pos + 1) % self.capacity;
    }

    /// Sample `batch_size` transitions uniformly at random (with replacement).
    /// Returns `None` if the buffer doesn't have enough transitions yet.
    ///
    /// Terminal transitions contribute zeroed next-state entries into
    /// `next_states` at their batch position, but callers must use
    /// `is_terminal` to mask them out rather than treating the zeros as real
    /// state, since an all-zero state is not necessarily meaningful.
    pub fn sample(
        &mut self,
        batch_size: usize,
        state_dim: usize,
        action_dim: usize,
    ) -> Option<Batch> {
        if self.transitions.len() < batch_size {
            return None;
        }

        let mut states = Vec::with_capacity(batch_size * state_dim);
        let mut actions = Vec::with_capacity(batch_size * action_dim);
        let mut rewards = Vec::with_capacity(batch_size);
        let mut next_states = Vec::with_capacity(batch_size * state_dim);
        let mut is_terminal = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            let idx = self.rng.next_index(self.transitions.len());
            let t = &self.transitions[idx];

            states.extend_from_slice(&t.state);
            actions.extend_from_slice(&t.action);
            rewards.push(t.reward);

            match &t.next_state {
                Some(ns) => {
                    next_states.extend_from_slice(ns);
                    is_terminal.push(false);
                }
                None => {
                    next_states.extend(std::iter::repeat(0.0).take(state_dim));
                    is_terminal.push(true);
                }
            }
        }

        Some(Batch {
            states,
            actions,
            rewards,
            next_states,
            is_terminal,
            batch_size,
        })
    }
}
