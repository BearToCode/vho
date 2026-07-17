use burn::Tensor;
use godot::prelude::*;
use probability::source;

use crate::{
    game::Game,
    rl::{
        Backend,
        action::{ACTION_DIM, OuNoise, perform_action},
        adhdp::{ADHDP, ADHDPConfig, ADHDPStepTrainData, ADHDPTrainData},
        episode::Episode,
        replay_buffer::{ReplayBuffer, Transition},
        reward::stability_reward_function,
        state::{AgentStateVector, OnlineStateNormalization, get_agent_state},
    },
};

struct StepData {
    /// Un-normalized state, so the replay buffer can re-normalize with fresh statistics.
    state: AgentStateVector,
    u: [f32; ACTION_DIM],
}

/// Read a `[1, N]` tensor's data out into a plain `[f32; N]` array.
fn to_array<const N: usize>(t: &Tensor<Backend, 2>) -> [f32; N] {
    let data = t
        .clone()
        .into_data()
        .to_vec::<f32>()
        .expect("Failed to read tensor data");

    data.try_into()
        .unwrap_or_else(|data: Vec<f32>| panic!("Expected {N} elements, got {}", data.len()))
}

#[derive(GodotClass)]
#[class(base=Node3D)]
/// Godot class for the agent, which handles the training.
pub struct Agent {
    base: Base<Node3D>,

    /* Internal states */
    /// Total elapsed training time, used to decay exploration noise.
    total_training_time: f32,
    /// Number of episodes.
    episode_count: usize,
    /// ADHDP
    adhdp: Option<ADHDP>,
    /// Episode statistics.
    episode: Episode,
    /// Previous episode data.
    previous_episode: Option<Episode>,
    /// Previous step data.
    previous_step: Option<StepData>,
    /// Specific subdirectory for this run.
    run_directory: String,
    /// Noise source generator
    noise_source: Option<source::Default>,
    /// Ornstein-Uhlenbeck exploration process.
    ou_noise: Option<OuNoise>,
    /// Best deterministic evaluation reward seen so far (for saving the best policy).
    best_episode_reward: f32,
    /// Whether the current episode is a no-noise, no-training evaluation episode.
    evaluating: bool,
    /// Online normalization
    normalization: OnlineStateNormalization,
    /// Experience replay buffer.
    replay_buffer: Option<ReplayBuffer>,

    /* Exported to the inspector */
    #[export]
    /// Reference to the game manager.
    game: Option<Gd<Game>>,

    #[export_group(name = "Model saving and loading")]
    #[export]
    #[var(hint = DIR)]
    /// Output directory to write runs data to.
    output_directory: GString,
    #[export(file)]
    #[var(hint = FILE)]
    /// Saved critic model. Will load it if specified.
    saved_critic_model: GString,
    #[export(file)]
    #[var(hint = FILE)]
    /// Saved second (TD3 twin) critic model. Will load it if specified.
    /// Safe to leave unset when warm-starting from a pre-TD3 checkpoint.
    saved_critic_2_model: GString,
    #[export(file)]
    #[var(hint = FILE)]
    /// Saved actor model. Will load it if specified.
    saved_actor_model: GString,
    #[export(file)]
    #[var(hint = FILE)]
    /// Saved state normalization model. Will load it if specified.
    saved_normalization_model: GString,

    #[export_group(name = "Reinforcement Learning")]
    #[export]
    /// Trains the model every specified amount of physics frames.
    train_every_n_frames: i32,
    #[export(range = (0.0, 1.0, 0.001))]
    /// Gamma parameter of the Bellman function.
    gamma: f32,
    #[export]
    /// Maximum duration of each episode in seconds.
    max_episode_time: f32,
    #[export]
    /// Run a noise-free, training-free evaluation episode every N episodes, and use it
    /// to decide whether to save the best policy. Set to 0 to disable evaluation, which
    /// falls back to checkpointing on the best training episode.
    eval_every_n_episodes: i32,
    #[export]
    critic_hidden_layers: Array<i64>,
    #[export]
    actor_hidden_layers: Array<i64>,

    #[export_subgroup(name = "Learning Rates")]
    #[export]
    #[var(hint = NONE)]
    /// Actor model learning rate.
    actor_learning_rate: f64,
    #[export]
    #[var(hint = NONE)]
    /// Critic model learning rate.
    critic_learning_rate: f64,

    #[export_subgroup(name = "Noise")]
    #[export]
    /// Whether to use noise.
    use_noise: bool,
    #[export]
    /// Seed for the noise generator.
    noise_seed: i64,
    #[export]
    /// Initial exploration noise scale (at training start).
    noise_start: f32,
    #[export]
    /// Minimum exploration noise scale (floor after decay).
    noise_min: f32,
    #[export]
    /// Exploration noise decay time constant in seconds.
    noise_decay: f32,
    #[export]
    /// Ornstein-Uhlenbeck mean-reversion rate, in units of 1/second. The noise
    /// decorrelates over roughly `1 / noise_theta` seconds, which is what sets how
    /// local the exploration is.
    noise_theta: f32,
    #[export]
    /// Ornstein-Uhlenbeck volatility. Together with theta this fixes the noise
    /// magnitude: the stationary standard deviation is `noise_sigma / sqrt(2 * noise_theta)`.
    noise_sigma: f32,

    #[export_subgroup(name = "Target Networks")]
    #[export]
    /// Whether to use target networks for the actor and critic models.
    use_target_networks: bool,
    #[export(range = (0.0, 1.0, 0.001))]
    /// Polyak averaging coefficient for target networks.
    tau: f32,

    #[export_subgroup(name = "TD3")]
    #[export]
    /// Number of critic updates between each actor and target network update.
    policy_delay: i32,
    #[export]
    /// Standard deviation of the target policy smoothing noise.
    target_noise_std: f32,
    #[export]
    /// Absolute bound applied to the target policy smoothing noise.
    target_noise_clip: f32,

    #[export_subgroup(name = "Replay Buffer")]
    #[export]
    /// Maximum number of transitions kept in the replay buffer.
    replay_buffer_capacity: i32,
    #[export]
    /// Number of transitions sampled per training step.
    batch_size: i32,
    #[export]
    /// Minimum number of transitions collected before training starts.
    min_replay_size: i32,
}

#[godot_api]
impl INode3D for Agent {
    fn init(base: Base<Node3D>) -> Self {
        Self {
            base,

            /* Internal states */
            total_training_time: 0.0,
            episode_count: 0,
            adhdp: None,
            episode: Episode::new(0),
            previous_episode: None,
            previous_step: None,
            run_directory: "".into(),
            noise_source: None,
            ou_noise: None,
            best_episode_reward: f32::NEG_INFINITY,
            evaluating: false,
            noise_seed: 1,
            normalization: OnlineStateNormalization::new(),
            replay_buffer: None,

            /* Exported to the inspector */
            game: None,

            output_directory: "".into(),
            saved_critic_model: "".into(),
            saved_critic_2_model: "".into(),
            saved_actor_model: "".into(),
            saved_normalization_model: "".into(),

            train_every_n_frames: 1,
            // At 30 Hz physics, the effective horizon is 1/(1 - gamma) steps: 0.95 only
            // looks 0.67 s ahead, which is far too myopic for position control. 0.99
            // gives ~3.3 s.
            gamma: 0.99,
            max_episode_time: 10.0,
            eval_every_n_episodes: 25,
            critic_hidden_layers: Array::from_iter(vec![128, 128]),
            actor_hidden_layers: Array::from_iter(vec![128, 128]),

            critic_learning_rate: 3e-4,
            actor_learning_rate: 1e-4,

            use_noise: true,
            noise_start: 0.4,
            noise_min: 0.05,
            // The classic DDPG values (theta 0.15, sigma 0.2) assume dt = 1 per env
            // step; this integrates in seconds at 30 Hz, so theta 0.15 would decorrelate
            // over 6.7 s — longer than an episode, making the noise a constant bias
            // rather than exploration. 4.5 restores the intended ~0.22 s, and sigma is
            // raised to keep the same stationary magnitude (sigma / sqrt(2*theta)).
            noise_theta: 4.5,
            noise_sigma: 1.1,
            // Measured in sim seconds. Learning needs on the order of 10^5 steps
            // (~1 hour at 30 Hz), so decaying over ~2 minutes would leave almost the
            // entire run with no exploration.
            noise_decay: 1200.0,

            use_target_networks: true,
            tau: 0.005,

            policy_delay: 2,
            target_noise_std: 0.2,
            target_noise_clip: 0.5,

            replay_buffer_capacity: 100_000,
            batch_size: 128,
            min_replay_size: 1000,
        }
    }

    fn ready(&mut self) {
        if self.output_directory.is_empty() {
            panic!("No output directory set!");
        }

        if self.game.is_none() {
            panic!("No game set!");
        }

        // Set ADHDP properties
        let mut adhdp = ADHDP::new(ADHDPConfig {
            gamma: self.gamma,
            actor_learning_rate: self.actor_learning_rate,
            critic_learning_rate: self.critic_learning_rate,
            use_target_networks: self.use_target_networks,
            tau: self.tau,
            policy_delay: self.policy_delay.max(1) as usize,
            target_noise_std: self.target_noise_std,
            target_noise_clip: self.target_noise_clip,
            actor_hidden_layers: self
                .actor_hidden_layers
                .iter_shared()
                .map(|x| x as usize)
                .collect(),
            critic_hidden_layers: self
                .critic_hidden_layers
                .iter_shared()
                .map(|x| x as usize)
                .collect(),
        });

        // Load models if specified
        if !self.saved_actor_model.is_empty() {
            adhdp.load_actor(&self.saved_actor_model.to_string());
        }

        if !self.saved_critic_model.is_empty() {
            adhdp.load_critic(&self.saved_critic_model.to_string());
        }

        if !self.saved_critic_2_model.is_empty() {
            adhdp.load_critic_2(&self.saved_critic_2_model.to_string());
        }

        if !self.saved_normalization_model.is_empty() {
            self.normalization
                .load(&self.saved_normalization_model.to_string());
        }

        // Determine run output directory
        self.run_directory = chrono::Local::now()
            .format(&format!(
                "{}/run_%Y_%m_%d__%H_%M_%S/",
                self.output_directory
            ))
            .to_string();

        if let Err(e) = std::fs::create_dir_all(&self.run_directory) {
            godot_error!(
                "Failed to create run directory \"{}\": {}",
                self.run_directory,
                e
            );
        }

        // Generate noise source and exploration process
        self.noise_source = Some(source::default(self.noise_seed as u64));
        self.ou_noise = Some(OuNoise::new(self.noise_theta, self.noise_sigma));

        self.replay_buffer = Some(ReplayBuffer::new(self.replay_buffer_capacity as usize));

        self.adhdp = Some(adhdp);
    }

    fn physics_process(&mut self, delta: f32) {
        self.total_training_time += delta;
        self.episode.time += delta;
        self.episode.steps += 1;

        let adhdp = self.adhdp.as_mut().unwrap();

        if self.episode.time > self.max_episode_time {
            self.episode.log(&self.run_directory);
            godot_print!("{}", self.episode);
            self.reset_episode();

            return;
        }

        let game = self.game.clone().unwrap();
        let game_bind = game.bind();
        let helicopter = game_bind.helicopter.clone().unwrap();

        let state = get_agent_state(game.clone());

        self.normalization.update(&state);
        let x = self.normalization.normalize(&state);

        if let Some(prev_step) = self.previous_step.as_ref() {
            let reward_value = stability_reward_function(&state);
            // godot_print!("Reward: {}", reward_value);

            // Store raw states: normalization statistics keep drifting, so a state
            // normalized at collection time would disagree with one normalized later.
            // Normalizing at sample time keeps every batch internally consistent.
            let replay_buffer = self.replay_buffer.as_mut().unwrap();
            replay_buffer.push(Transition {
                state: prev_step.state,
                action: prev_step.u,
                reward: reward_value,
                next_state: state,
            });

            // Hold the policy fixed across an evaluation episode, so its score measures
            // one policy rather than a moving one.
            if !self.evaluating
                && self.episode.steps % (self.train_every_n_frames as usize) == 0
                && replay_buffer.len() >= self.min_replay_size as usize
            {
                if let Some((x, u, reward, x_next)) =
                    replay_buffer.sample(self.batch_size as usize, &self.normalization)
                {
                    let losses = adhdp.train(ADHDPTrainData::Step(ADHDPStepTrainData {
                        x,
                        u,
                        reward,
                        x_next,
                    }));

                    self.episode.critic_loss_sum += losses.critic_loss;
                    self.episode.train_steps += 1;
                    // The actor only updates every `policy_delay` steps.
                    if let Some(actor_loss) = losses.actor_loss {
                        self.episode.actor_loss_sum += actor_loss;
                        self.episode.actor_train_steps += 1;
                    }
                }
            }

            self.episode.accumulated_reward += reward_value;
        }

        let mut u = adhdp.act(x.clone());
        // Add temporally correlated exploration noise, decayed over training time.
        if self.use_noise && !self.evaluating {
            let scale = self.noise_min
                + (self.noise_start - self.noise_min)
                    * (-self.total_training_time / self.noise_decay).exp();
            self.episode.noise = scale;

            if let (Some(ou), Some(noise_source)) =
                (self.ou_noise.as_mut(), self.noise_source.as_mut())
            {
                let noise = ou.sample(delta, scale, noise_source);
                u = u + noise;
            }
        }
        let u = u.clamp(-1.0, 1.0);
        let u_values = to_array(&u);
        perform_action(u, helicopter);

        self.previous_step = Some(StepData {
            state,
            u: u_values,
        });
    }
}

#[godot_api]
impl Agent {
    /// Reset the episode.
    #[func]
    fn reset_episode(&mut self) {
        let adhdp = self.adhdp.as_mut().unwrap();

        self.episode_count += 1;
        if self.episode_count % 100 == 0 {
            adhdp.save(&self.run_directory, &self.episode_count.to_string());
            self.normalization.save(&format!(
                "{}normalization_{}.bin",
                self.run_directory, self.episode_count
            ));
        }

        // Save the best policy seen so far, so a later divergence can't destroy it.
        // (actor_best / critic_best in the run directory.)
        //
        // Only episodes that were measured without exploration noise are eligible. A
        // noisy episode's score says as much about the noise it happened to draw as it
        // does about the policy, so comparing those would checkpoint lucky draws.
        let eligible = self.evaluating || self.eval_every_n_episodes <= 0;
        if eligible && self.episode.accumulated_reward > self.best_episode_reward {
            self.best_episode_reward = self.episode.accumulated_reward;
            adhdp.save(&self.run_directory, "best");
            // The weights are meaningless without the statistics used to normalize
            // their inputs, so this must be saved alongside them to be reloadable.
            self.normalization
                .save(&format!("{}normalization_best.bin", self.run_directory));
        }

        // Decide whether the episode starting now is an evaluation episode. This has to
        // happen before the reset, which needs the flag for the episode about to start
        // rather than the one that just ended.
        self.evaluating = self.eval_every_n_episodes > 0
            && self.episode_count % (self.eval_every_n_episodes as usize) == 0;

        // Evaluation episodes start from the nominal pose, so their reward stays
        // comparable both across a run and against earlier runs.
        let mut game = self.game.clone().unwrap();
        game.bind_mut().reset(!self.evaluating);

        // Restart the exploration process so each episode explores independently.
        if let Some(ou) = self.ou_noise.as_mut() {
            ou.reset();
        }

        self.previous_episode = Some(self.episode.clone());
        self.episode = Episode::new(self.episode_count);
        self.episode.evaluation = self.evaluating;
        self.previous_step = None;
    }
}
