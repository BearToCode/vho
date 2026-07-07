use burn::{
    Tensor,
    backend::{
        Autodiff,
        flex::{Flex, FlexDevice},
    },
    module::Module,
    optim::{Adam, AdamConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
};
use godot::{global::rad_to_deg, prelude::*};

use crate::{
    game::Game,
    helicopter::{Helicopter, HelicopterStateVectorComponent},
    networks::{ActorModel, CriticModel},
    noise::OuNoise,
    replay_buffer::ReplayBuffer,
    ring::Ring,
};

/// The Burn backend to use. Flex is a lightweight Rust backend that runs on the CPU.
type Backend = Autodiff<Flex>;

/// Dimension of the state input of the model.
const STATE_DIM: usize = 14;
/// Dimension of the action output of the model.
const ACTION_DIM: usize = 4;

pub struct EpisodeState {
    /// Elapsed time.
    pub time: f32,
    /// Progression along the track.
    pub track_progress: f32,
    /// Number of rings passed.
    pub rings_passed: usize,
    /// Accumulated reward this episode.
    pub reward: f32,
    /// Sum of critic losses this episode (for averaging).
    pub critic_loss_sum: f32,
    /// Sum of actor losses this episode (for averaging).
    pub actor_loss_sum: f32,
    /// Number of train_step calls this episode (for averaging).
    pub train_steps: usize,
}

impl EpisodeState {
    pub fn new() -> Self {
        EpisodeState {
            time: 0.0,
            track_progress: 0.0,
            rings_passed: 0,
            reward: 0.0,
            critic_loss_sum: 0.0,
            actor_loss_sum: 0.0,
            train_steps: 0,
        }
    }
}

#[derive(GodotClass)]
#[class(base=Node3D)]
/// Godot class for the agent, which handles the training.
pub struct Agent {
    base: Base<Node3D>,

    // RL data
    /// The device the training runs on.
    device: FlexDevice,
    /// The online actor model.
    actor: ActorModel<Backend>,
    /// The online critic model.
    critic: CriticModel<Backend>,
    /// The actor target model.
    actor_target: ActorModel<Backend>,
    /// The critic target model.
    critic_target: CriticModel<Backend>,
    /// Adam optimizer for the actor model.
    actor_optimizer: OptimizerAdaptor<Adam, ActorModel<Backend>, Backend>,
    /// Adam optimizer for the critic model.
    critic_optimizer: OptimizerAdaptor<Adam, CriticModel<Backend>, Backend>,

    // RL states
    /// Total elapsed training time, used to decay exploration noise.
    total_training_time: f32,
    /// Number of episodes.
    episode_count: usize,
    /// Episode statistics.
    episode: EpisodeState,
    /// Previous state tensor.
    prev_x: Option<Tensor<Backend, 2>>,
    /// Previous action tensor.
    prev_u: Option<Tensor<Backend, 2>>,
    /// Exploration noise process, one component per action dimension.
    action_noise: OuNoise,
    /// Replay buffer of past transitions for decorrelated minibatch training.
    replay_buffer: ReplayBuffer,

    /// Path to the output directory.
    log_dir: String,

    // Godot params
    #[export]
    game: Option<Gd<Game>>,

    #[export_group(name = "Reinforcement Learning")]
    #[export]
    /// Actor model learning rate.
    actor_lr: f64,
    #[export]
    /// Critic model learning rate.
    critic_lr: f64,
    #[export]
    /// Gamma parameter of the Bellman function.
    gamma: f32,
    #[export]
    /// Tau parameter for Polyak updates.
    tau: f32,
    #[export]
    /// Maximum duration of each episode in seconds.
    max_episode_time: f32,
    #[export]
    /// Training time after which noise scale reaches 0 (fully exploiting).
    noise_decay_time: f32,
    #[export_subgroup(name = "Replay Batch")]
    #[export]
    /// Number of transitions to sample per training step.
    batch_size: i64,
    #[export]
    /// Minimum buffer size before training starts.
    min_buffer_size: i64,

    #[export_group(name = "Input Ranges")]
    #[export]
    /// Range of allowed collective action values (-range, +range)
    collective_range: f32,
    #[export]
    /// Range of allowed lateral cyclic action values (-range, +range)
    lateral_cyclic_range: f32,
    #[export]
    /// Range of allowed longitudinal cyclic action values (-range, +range)
    longitudinal_cyclic_range: f32,
    #[export]
    /// Range of allowed tail rotor cyclic action values (-range, +range)
    tail_rotor_cyclic_range: f32,
}

/// Type of critic update to perform.
pub enum CriticUpdate {
    /// This is a terminal update. The target value is equal to the reward.
    Terminal,
    /// Normal update. The target value is reward + j_next * gamma
    Normal(Tensor<Backend, 2>),
}

#[godot_api]
impl INode3D for Agent {
    fn init(base: Base<Node3D>) -> Self {
        use chrono::Local;
        use std::fs;

        let device = FlexDevice;

        let actor = ActorModel::new(STATE_DIM, ACTION_DIM, &device);
        let critic = CriticModel::new(STATE_DIM, ACTION_DIM, &device);

        let actor_target = actor.clone();
        let critic_target = critic.clone();

        let log_dir = Local::now()
            .format("../output/run_%Y_%m_%d__%H_%M_%S/")
            .to_string();
        if let Err(e) = fs::create_dir_all(&log_dir) {
            godot_error!("Failed to create run directory {log_dir}: {e}");
        }

        Self {
            base,

            // RL data
            device: device.clone(),
            actor,
            critic,
            actor_target,
            critic_target,
            actor_optimizer: AdamConfig::new().build().into(),
            critic_optimizer: AdamConfig::new().build().into(),

            // RL params
            actor_lr: 1e-4,
            critic_lr: 1e-3,
            gamma: 0.95,
            tau: 0.005,
            max_episode_time: 5.0,
            total_training_time: 0.0,
            noise_decay_time: 20_000.0,

            // RL states
            episode_count: 0,
            episode: EpisodeState::new(),
            action_noise: OuNoise::new(ACTION_DIM, 0.15, 0.2, 0xC0FFEE),
            prev_u: None,
            prev_x: None,
            replay_buffer: ReplayBuffer::new(100_000, 0xDEADBEEF),

            log_dir,

            // Godot params
            game: None,
            collective_range: 5.0,
            lateral_cyclic_range: 1.0,
            longitudinal_cyclic_range: 1.0,
            tail_rotor_cyclic_range: 0.3,
            batch_size: 128,
            min_buffer_size: 1_000,
        }
    }

    fn physics_process(&mut self, delta: f32) {
        self.episode.time += delta;
        self.total_training_time += delta;

        // Decay exploration noise linearly to 0 over noise_decay_time.
        let scale = 1.0 - (self.total_training_time / self.noise_decay_time).min(1.0);
        self.action_noise.set_scale(scale);

        let mut game: Gd<Game> = self.game.clone().unwrap();
        let helicopter: Gd<Helicopter> = game.bind().helicopter.clone().unwrap();

        if self.episode.time > self.max_episode_time {
            self.log_episode(false);
            game.bind_mut().reset();
            self.reset_episode();

            return;
        }

        // Let the agent act based on the current state
        let (x, u) = {
            let game_bind = game.bind();
            let track_bind = game_bind.track.as_ref().unwrap().bind();

            let current_ring = track_bind.current_ring().expect("Current ring is not set");
            let next_ring = track_bind.next_ring();
            let helicopter = game_bind.helicopter.as_ref().unwrap();

            let x = self.get_state(helicopter.clone(), current_ring, next_ring);
            let u = self.actor.forward(x.clone());

            let noisy_u = self.add_exploration_noise(u.clone());
            self.act(noisy_u, helicopter.clone());

            (x, u)
        };

        // Check if the helicopter is crashing
        if self.crashing(helicopter.clone()) {
            let crash_penalty = 30.0;
            let reward_value = -crash_penalty;

            if let (Some(prev_x), Some(prev_u)) = (&self.prev_x, &self.prev_u) {
                self.replay_buffer.push(
                    Self::tensor_to_vec(prev_x),
                    Self::tensor_to_vec(prev_u),
                    reward_value,
                    None, // terminal
                );
                self.episode.reward += reward_value;
            }

            let (critic_loss, actor_loss) = self.train_from_buffer();
            self.episode.critic_loss_sum += critic_loss;
            self.episode.actor_loss_sum += actor_loss;
            self.episode.train_steps += 1;

            self.log_episode(true);
            game.bind_mut().reset();
            self.reset_episode();
            return;
        }

        // Train all the networks
        if let Some(prev_x) = self.prev_x.as_ref()
            && let Some(prev_u) = self.prev_u.as_ref()
        {
            let new_progress = game.bind().track_progress();
            let new_rings_passed = game.bind().rings_passed();

            let progress_reward = (new_progress - self.episode.track_progress) * 10.0;
            let rings_reward = (new_rings_passed - self.episode.rings_passed) as f32 * 10.0;
            let living_penalty = 0.05 * delta;
            let reward_value = progress_reward + rings_reward - living_penalty;

            self.replay_buffer.push(
                Self::tensor_to_vec(prev_x),
                Self::tensor_to_vec(prev_u),
                reward_value,
                Some(Self::tensor_to_vec(&x)),
            );

            let (critic_loss, actor_loss) = self.train_from_buffer();

            self.episode.reward += reward_value;
            self.episode.critic_loss_sum += critic_loss;
            self.episode.actor_loss_sum += actor_loss;
            self.episode.train_steps += 1;
            self.episode.track_progress = new_progress;
            self.episode.rings_passed = new_rings_passed;
        }

        self.prev_x = Some(x);
        self.prev_u = Some(u);
    }
}

impl Agent {
    /// Get the state tensor from the simulation.
    fn get_state(
        &self,
        helicopter: Gd<Helicopter>,
        current_ring: Gd<Ring>,
        next_ring: Option<Gd<Ring>>,
    ) -> Tensor<Backend, 2> {
        // First 8 components: Helicopter dynamics state vector
        let mut helicopter_state = helicopter.bind().get_state_vector().clone();
        type SV = HelicopterStateVectorComponent;
        // Scale velocities  for better gradients
        helicopter_state[SV::U as usize] /= 50.0;
        helicopter_state[SV::W as usize] /= 50.0;
        helicopter_state[SV::V as usize] /= 50.0;
        helicopter_state[SV::Q] *= 2.0;
        helicopter_state[SV::P] *= 2.0;
        helicopter_state[SV::R] *= 2.0;

        // Helicopter necessary transform data
        let helicopter_position = helicopter.get_global_position();
        let global_to_local = helicopter.get_transform().basis.inverse();

        // Second 3 components: Helicopter position relative to fist ring, in local reference frame
        let current_ring_position = current_ring.get_global_position();
        let mut current_ring_relative_position =
            global_to_local * (current_ring_position - helicopter_position);

        // Third 3 components: Helicopter position relative to second ring, in local reference frame
        let mut next_ring_relative_position = if let Some(next_ring) = next_ring {
            let next_ring_position = next_ring.get_global_position();
            global_to_local * (next_ring_position - helicopter_position)
        } else {
            // Use same location as current ring if it's the last one
            current_ring_relative_position
        };

        // Rescale rings relative positions
        current_ring_relative_position /= 100.0;
        next_ring_relative_position /= 100.0;

        return Tensor::<Backend, 1>::from_data(
            [
                helicopter_state.data.as_slice(),
                &[
                    current_ring_relative_position.x,
                    current_ring_relative_position.y,
                    current_ring_relative_position.z,
                ],
                &[
                    next_ring_relative_position.x,
                    next_ring_relative_position.y,
                    next_ring_relative_position.z,
                ],
            ]
            .concat()
            .as_slice(),
            &self.device,
        )
        .reshape([1, STATE_DIM]);
    }

    /// Sample a minibatch from the replay buffer and perform one training
    /// step on it. Returns (0.0, 0.0) if there isn't enough data yet to
    /// sample a full batch (no-op, but still counted in episode stats as a
    /// zero-loss step — acceptable early on since it only affects a few
    /// episodes' worth of logging before the buffer fills).
    fn train_from_buffer(&mut self) -> (f32, f32) {
        if self.replay_buffer.len() < self.min_buffer_size as usize {
            return (0.0, 0.0);
        }

        let batch = match self
            .replay_buffer
            .sample(self.batch_size as usize, STATE_DIM, ACTION_DIM)
        {
            Some(b) => b,
            None => return (0.0, 0.0),
        };

        let bs = batch.batch_size;

        let states = Tensor::<Backend, 1>::from_floats(batch.states.as_slice(), &self.device)
            .reshape([bs, STATE_DIM]);
        let actions = Tensor::<Backend, 1>::from_floats(batch.actions.as_slice(), &self.device)
            .reshape([bs, ACTION_DIM]);
        let rewards = Tensor::<Backend, 1>::from_floats(batch.rewards.as_slice(), &self.device)
            .reshape([bs, 1]);
        let next_states =
            Tensor::<Backend, 1>::from_floats(batch.next_states.as_slice(), &self.device)
                .reshape([bs, STATE_DIM]);

        // Mask out target contributions for terminal transitions: for those,
        // the target should be just `reward`, not `reward + gamma * j_next`.
        let non_terminal_mask: Vec<f32> = batch
            .is_terminal
            .iter()
            .map(|&t| if t { 0.0 } else { 1.0 })
            .collect();
        let non_terminal_mask =
            Tensor::<Backend, 1>::from_floats(non_terminal_mask.as_slice(), &self.device)
                .reshape([bs, 1]);

        // 1. critic update
        let u_next = self.actor_target.forward(next_states.clone()).detach();
        let j_next = self.critic_target.forward(next_states, u_next).detach();
        let target = rewards + j_next.mul_scalar(self.gamma) * non_terminal_mask;

        let j_pred = self.critic.forward(states.clone(), actions.clone());
        let critic_loss = (target - j_pred).powf_scalar(2.0).mean();
        let critic_loss_value = critic_loss
            .clone()
            .into_data()
            .to_vec::<f32>()
            .expect("Failed to read critic loss")[0];

        let grads = critic_loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.critic);
        self.critic = self
            .critic_optimizer
            .step(self.critic_lr, self.critic.clone(), grads);

        // 2. actor update (maximize J)
        let u_pred = self.actor.forward(states.clone());
        let j_for_actor = self.critic.forward(states, u_pred).mean();
        let actor_loss = j_for_actor.neg();
        let actor_loss_value = actor_loss
            .clone()
            .into_data()
            .to_vec::<f32>()
            .expect("Failed to read actor loss")[0];

        let grads = actor_loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.actor);
        self.actor = self
            .actor_optimizer
            .step(self.actor_lr, self.actor.clone(), grads);

        // 3. Polyak averaging
        self.actor_target.polyak_update(&self.actor, self.tau);
        self.critic_target.polyak_update(&self.critic, self.tau);

        (critic_loss_value, actor_loss_value)
    }

    /// Perform a certain action in the simulation.
    fn act(&self, u: Tensor<Backend, 2>, mut helicopter: Gd<Helicopter>) {
        let control_normalized = u
            .into_data()
            .to_vec::<f32>()
            .expect("Failed to read outputs from actor network");

        if control_normalized.len() != ACTION_DIM {
            panic!(
                "Wrong data size for control output: expected {0}, got {1}",
                ACTION_DIM,
                control_normalized.len()
            );
        }

        let mut helicopter_bind = helicopter.bind_mut();
        helicopter_bind.collective = self.collective_range * control_normalized[0];
        helicopter_bind.lateral_cyclic = self.lateral_cyclic_range * control_normalized[1];
        helicopter_bind.longitudinal_cyclic =
            self.longitudinal_cyclic_range * control_normalized[2];
        helicopter_bind.tail_rotor_cyclic = self.tail_rotor_cyclic_range * control_normalized[3];
    }

    /// Check if the helicopter is flying in a invalid state.
    fn crashing(&self, helicopter: Gd<Helicopter>) -> bool {
        let helicopter_bind = helicopter.bind();
        let helicopter_state = helicopter_bind.get_state_vector();
        type SV = HelicopterStateVectorComponent;
        let pitch_angle = helicopter_state[SV::Theta];
        let roll_angle = helicopter_state[SV::Phi];

        if rad_to_deg(f32::abs(pitch_angle) as f64) > 40.0 {
            return true;
        }

        if rad_to_deg(f32::abs(roll_angle) as f64) > 40.0 {
            return true;
        }

        return false;
    }

    /// Add decayed OU exploration noise to an action tensor, clamping to [-1, 1]
    /// since actor outputs are assumed normalized before being scaled by the
    /// input ranges in `act`.
    fn add_exploration_noise(&mut self, u: Tensor<Backend, 2>) -> Tensor<Backend, 2> {
        let noise = self.action_noise.sample();
        let noise_tensor = Tensor::<Backend, 1>::from_floats(noise.as_slice(), &self.device)
            .reshape([1, ACTION_DIM]);

        (u + noise_tensor).clamp(-1.0, 1.0)
    }

    /// Reset the episode.
    fn reset_episode(&mut self) {
        self.episode_count += 1;
        if self.episode_count % 100 == 0 {
            self.save_checkpoint();
        }

        self.episode = EpisodeState::new();
        self.prev_u = None;
        self.prev_x = None;
        self.action_noise.reset();
    }

    /// Append this episode's stats as one row in the training log CSV.
    /// Writes a header row if the file doesn't exist yet.
    fn log_episode(&self, crashed: bool) {
        use std::io::Write;

        let log_path = format!("{}training_log.csv", self.log_dir);

        let file_exists = std::path::Path::new(&log_path).exists();

        let mut file = match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
        {
            Ok(f) => f,
            Err(e) => {
                godot_error!("Failed to open training log: {e}");
                return;
            }
        };

        if !file_exists {
            let header = "episode_time,track_progress,rings_passed,episode_reward,avg_critic_loss,avg_actor_loss,noise_scale,crashed\n";
            if let Err(e) = file.write_all(header.as_bytes()) {
                godot_error!("Failed to write CSV header: {e}");
                return;
            }
        }

        let avg_critic_loss = if self.episode.train_steps > 0 {
            self.episode.critic_loss_sum / self.episode.train_steps as f32
        } else {
            0.0
        };
        let avg_actor_loss = if self.episode.train_steps > 0 {
            self.episode.actor_loss_sum / self.episode.train_steps as f32
        } else {
            0.0
        };

        let row = format!(
            "{},{},{},{},{},{},{},{}\n",
            self.episode.time,
            self.episode.track_progress,
            self.episode.rings_passed,
            self.episode.reward,
            avg_critic_loss,
            avg_actor_loss,
            self.action_noise.scale(),
            crashed,
        );

        if let Err(e) = file.write_all(row.as_bytes()) {
            godot_error!("Failed to write CSV row: {e}");
        }

        godot_print!("Episode completed: {}", row);
    }

    /// Save the current actor and critic model weights to disk.
    /// Call this periodically (e.g. every N episodes) or on demand.
    pub fn save_checkpoint(&self) {
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

        if let Err(e) = self.actor.clone().save_file(
            format!("{}/actor_{}", self.log_dir, self.episode_count),
            &recorder,
        ) {
            godot_print!("Failed to save actor: {e}");
        }
        if let Err(e) = self.critic.clone().save_file(
            format!("{}/critic_{}", self.log_dir, self.episode_count),
            &recorder,
        ) {
            godot_print!("Failed to save critic: {e}");
        }

        godot_print!("Saved checkpoint");
    }

    /// Convert a [1, dim] tensor to a flat Vec<f32>.
    fn tensor_to_vec(t: &Tensor<Backend, 2>) -> Vec<f32> {
        t.clone()
            .into_data()
            .to_vec::<f32>()
            .expect("Failed to read tensor data")
    }
}
