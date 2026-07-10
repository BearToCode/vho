use burn::{
    Tensor,
    backend::{Autodiff, flex::Flex},
};
use godot::prelude::*;
use probability::source;

use crate::{
    game::Game,
    rl::{
        DEVICE,
        action::{PerformActionConfig, get_noise, perform_action},
        adhdp::{ADHDP, ADHDPStepTrainData, ADHDPTrainData},
        episode::Episode,
        reward::{fwd_stability_reward_field, reward_function_from_field},
        state::{AgentStateVector, StateNormalizationConfig, get_agent_state, normalize_state},
    },
};

/// The Burn backend to use. Flex is a lightweight Rust backend that runs on the CPU.
type Backend = Autodiff<Flex>;

struct StepData {
    state: AgentStateVector,
    x: Tensor<Backend, 2>,
    u: Tensor<Backend, 2>,
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
    adhdp: ADHDP,
    /// Episode statistics.
    episode: Episode,
    /// Previous step data.
    previous_step: Option<StepData>,
    /// Specific subdirectory for this run.
    run_directory: String,
    /// Noise source generator
    noise_source: Option<source::Default>,

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
    /// Saved actor model. Will load it if specified.
    saved_actor_model: GString,

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

    #[export_subgroup(name = "Rewards")]
    #[export]
    /// Reward for passing a ring.
    reward_per_ring: f32,
    #[export]
    /// Reward for progressing along the track.
    reward_per_progress: f32,
    #[export]
    /// Penalty for crashing the helicopter (flying in an invalid state).
    penalty_per_crash: f32,

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
    /// How many episodes to decrease noise in.
    noise_episodes_decay: i64,
    #[export]
    /// Seed for the noise generator.
    noise_seed: i64,

    #[export_subgroup(name = "State Normalization")]
    #[export]
    /// Angular velocity scale for state normalization.
    angular_velocity_scale: f32,
    #[export]
    /// Linear velocity scale for state normalization.
    linear_velocity_scale: f32,
    #[export]
    /// Angle scale for state normalization.
    angle_scale: f32,
    #[export]
    /// Position scale for state normalization.
    position_scale: f32,

    #[export_group(name = "Input Ranges")]
    #[export]
    #[var(hint = NONE)]
    /// Range of allowed collective action values.
    collective_range: f32,
    #[export]
    #[var(hint = NONE)]
    /// Range of allowed lateral cyclic action values.
    lateral_cyclic_range: f32,
    #[export]
    #[var(hint = NONE)]
    /// Range of allowed longitudinal cyclic action values.
    longitudinal_cyclic_range: f32,
    #[export]
    #[var(hint = NONE)]
    /// Range of allowed tail rotor cyclic action values.
    tail_rotor_cyclic_range: f32,
}

#[godot_api]
impl INode3D for Agent {
    fn init(base: Base<Node3D>) -> Self {
        Self {
            base,

            /* Internal states */
            total_training_time: 0.0,
            episode_count: 0,
            adhdp: ADHDP::new(),
            episode: Episode::new(),
            previous_step: None,
            run_directory: "".into(),
            noise_source: None,
            noise_seed: 1,

            /* Exported to the inspector */
            game: None,

            output_directory: "".into(),
            saved_critic_model: "".into(),
            saved_actor_model: "".into(),

            train_every_n_frames: 1,
            gamma: 0.95,
            max_episode_time: 10.0,

            reward_per_ring: 10.0,
            reward_per_progress: 3.0,
            penalty_per_crash: -10.0,

            critic_learning_rate: 1e-3,
            actor_learning_rate: 1e-4,

            use_noise: true,
            noise_episodes_decay: 500,

            linear_velocity_scale: 1.0 / 50.0,
            angular_velocity_scale: 2.0,
            angle_scale: 1.0,
            position_scale: 1.0 / 100.0,

            collective_range: 5.0,
            lateral_cyclic_range: 1.0,
            longitudinal_cyclic_range: 1.0,
            tail_rotor_cyclic_range: 0.3,
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
        self.adhdp.config.gamma = self.gamma;
        self.adhdp.config.actor_learning_rate = self.actor_learning_rate;
        self.adhdp.config.critic_learning_rate = self.critic_learning_rate;

        // Load models if specified
        if !self.saved_actor_model.is_empty() {
            self.adhdp.load_actor(&self.saved_actor_model.to_string());
        }

        if !self.saved_critic_model.is_empty() {
            self.adhdp.load_critic(&self.saved_critic_model.to_string());
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

        // Generate noise source
        self.noise_source = Some(source::default(self.noise_seed as u64));
    }

    fn physics_process(&mut self, delta: f32) {
        self.total_training_time += delta;
        self.episode.time += delta;
        self.episode.steps += 1;

        if self.episode.time > self.max_episode_time {
            self.episode.log(&self.run_directory);
            godot_print!("{}", self.episode);
            self.reset_episode();

            return;
        }

        let game = self.game.clone().unwrap();
        let game_bind = game.bind();
        let helicopter = game_bind.helicopter.clone().unwrap();

        let normalization_config = StateNormalizationConfig {
            angle_scale: self.angle_scale,
            linear_velocity_scale: self.linear_velocity_scale,
            angular_velocity_scale: self.angular_velocity_scale,
            position_scale: self.position_scale,
        };

        let action_config = PerformActionConfig {
            collective_range: self.collective_range,
            lateral_cyclic_range: self.lateral_cyclic_range,
            longitudinal_cyclic_range: self.longitudinal_cyclic_range,
            tail_rotor_cyclic_range: self.tail_rotor_cyclic_range,
        };

        let state = get_agent_state(game.clone());
        let x = normalize_state(&state, &normalization_config);

        if let Some(prev_step) = self.previous_step.as_ref() {
            let reward_function = reward_function_from_field(fwd_stability_reward_field);
            let reward_value = reward_function(&prev_step.state, &state);
            let reward = Tensor::<Backend, 1>::from_data([reward_value], &DEVICE).reshape([1, 1]);

            if self.episode.steps % (self.train_every_n_frames as usize) == 0 {
                let losses = self.adhdp.train(ADHDPTrainData::Step(ADHDPStepTrainData {
                    x: prev_step.x.clone(),
                    u: prev_step.u.clone(),
                    reward,
                    x_next: x.clone(),
                }));

                self.episode.critic_loss_sum += losses.critic_loss;
                self.episode.actor_loss_sum += losses.actor_loss;
                self.episode.train_steps += 1;
            }

            self.episode.accumulated_reward += reward_value;
        }

        let mut u = self.adhdp.act(x.clone());
        // Add noise to the action if enabled
        if self.use_noise && self.episode_count < self.noise_episodes_decay as usize {
            self.episode.noise = (-((128.0 as f32).log(std::f32::consts::E)
                * self.episode_count as f32
                / self.noise_episodes_decay as f32))
                .exp();

            if let Some(noise_source) = self.noise_source.as_mut() {
                let noise = get_noise(self.episode.noise, noise_source);
                u = u + noise;
            }
        }

        perform_action(u.clone(), helicopter, &action_config);

        self.previous_step = Some(StepData { state, x, u });
    }
}

#[godot_api]
impl Agent {
    /// Reset the episode.
    #[func]
    fn reset_episode(&mut self) {
        let mut game = self.game.clone().unwrap();
        game.bind_mut().reset();

        self.episode_count += 1;
        if self.episode_count % 100 == 0 {
            self.adhdp
                .save(&self.run_directory, &self.episode_count.to_string());
        }

        self.episode = Episode::new();
        self.previous_step = None;
    }
}
