use burn::Tensor;
use godot::prelude::*;
use probability::prelude::*;

use crate::{
    game::Game,
    rl::{
        Backend, DEVICE,
        action::{OuNoise, perform_action},
        adhdp::{ADHDP, ADHDPConfig, ADHDPStepTrainData, ADHDPTerminalTrainData, ADHDPTrainData},
        episode::Episode,
        reward::{RewardConfig, stability_reward_function},
        state::{StateNormalizationConfig, get_agent_state, is_tumbling, normalize_state},
    },
};

struct StepData {
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
    /// Best episode reward seen so far (for saving the best policy).
    best_episode_reward: f32,

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
    /// Enables training of the agent.
    train: bool,
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

    #[export_subgroup(name = "Spawn Attitude")]
    #[export]
    /// Initial roll angle ranges of the helicopter in degrees.
    initial_roll_range_deg: f32,
    #[export]
    /// Initial pitch angle ranges of the helicopter in degrees.
    initial_pitch_range_deg: f32,
    #[export]
    /// Initial linear velocity ranges of the helicopter in meters per second.
    initial_linear_velocity_range: f32,
    #[export]
    /// Initial angular velocity ranges of the helicopter in degrees per second.
    initial_angular_velocity_range_deg: f32,

    #[export_subgroup(name = "Reward")]
    #[export]
    /// Roll range for reward calculation.
    roll_range_deg: f32,
    #[export]
    /// Pitch range for reward calculation.
    pitch_range_deg: f32,
    #[export]
    /// Angular velocity range for reward calculation.
    angular_velocity_range_deg: f32,
    #[export]
    /// Linear velocity range for reward calculation.
    linear_velocity_range: f32,

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

    #[export_subgroup(name = "Target Networks")]
    #[export]
    /// Whether to use target networks for the actor and critic models.
    use_target_networks: bool,
    #[export(range = (0.0, 1.0, 0.001))]
    /// Polyak averaging coefficient for target networks.
    tau: f32,

    #[export_subgroup(name = "State Normalization")]
    #[export]
    /// Linear velocity scale for state normalization.
    linear_velocity_scale: f32,
    #[export]
    /// Angular velocity scale for state normalization.
    angular_velocity_scale: f32,
    #[export]
    /// Angle scale for state normalization.
    angle_scale: f32,
    #[export]
    /// Flap angle scale for state normalization.
    flap_angle_scale: f32,
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
            noise_seed: 1,

            /* Exported to the inspector */
            game: None,

            output_directory: "".into(),
            saved_critic_model: "".into(),
            saved_actor_model: "".into(),

            train: true,
            train_every_n_frames: 1,
            gamma: 0.95,
            max_episode_time: 10.0,
            critic_hidden_layers: Array::from_iter(vec![128, 128]),
            actor_hidden_layers: Array::from_iter(vec![128, 128]),

            critic_learning_rate: 3e-4,
            actor_learning_rate: 1e-4,

            initial_pitch_range_deg: 20.0,
            initial_roll_range_deg: 20.0,
            initial_linear_velocity_range: 1.0,
            initial_angular_velocity_range_deg: 10.0,

            roll_range_deg: 45.0,
            pitch_range_deg: 45.0,
            angular_velocity_range_deg: 45.0,
            linear_velocity_range: 2.0,

            use_noise: true,
            noise_start: 0.4,
            noise_min: 0.05,
            noise_decay: 120.0,

            use_target_networks: true,
            tau: 0.005,

            linear_velocity_scale: 1.0,
            angular_velocity_scale: 1.0,
            angle_scale: 1.0,
            flap_angle_scale: 1.0,
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
        self.ou_noise = Some(OuNoise::new(0.15, 0.2));

        self.adhdp = Some(adhdp);
    }

    fn physics_process(&mut self, delta: f32) {
        self.total_training_time += delta;
        self.episode.time += delta;
        self.episode.steps += 1;

        let adhdp = self.adhdp.as_mut().unwrap();

        let game = self.game.clone().unwrap();
        let game_bind = game.bind();
        let helicopter = game_bind.helicopter.clone().unwrap();

        let state = get_agent_state(game.clone());

        if self.episode.time > self.max_episode_time || is_tumbling(&state) {
            if is_tumbling(&state) {
                // Perform one training step for the last state
                if let Some(prev_step) = self.previous_step.as_ref() {
                    let time_left = self.max_episode_time - self.episode.time;
                    let reward_value = -time_left; // Penalize for tumbling, scaled by time left
                    let reward =
                        Tensor::<Backend, 1>::from_data([reward_value], &DEVICE).reshape([1, 1]);
                    self.episode.accumulated_reward += reward_value;

                    if self.train {
                        let losses =
                            adhdp.train(ADHDPTrainData::Terminal(ADHDPTerminalTrainData {
                                x: prev_step.x.clone(),
                                u: prev_step.u.clone(),
                                reward,
                            }));

                        self.episode.critic_loss_sum += losses.critic_loss;
                        self.episode.actor_loss_sum += losses.actor_loss;
                        self.episode.train_steps += 1;
                    }
                }
            }

            self.episode.log(&self.run_directory);
            godot_print!("{}", self.episode);
            drop(game_bind);
            self.reset_episode();

            return;
        }

        let normalization_config = StateNormalizationConfig {
            angular_velocity_scale: self.angular_velocity_scale,
            linear_velocity_scale: self.linear_velocity_scale,
            angle_scale: self.angle_scale,
            flap_angle_scale: self.flap_angle_scale,
        };

        let x = normalize_state(&state, &normalization_config);

        if let Some(prev_step) = self.previous_step.as_ref() {
            let reward_config = RewardConfig {
                roll_range: self.roll_range_deg.to_radians(),
                pitch_range: self.pitch_range_deg.to_radians(),
                angular_velocity_range: self.angular_velocity_range_deg.to_radians(),
                linear_velocity_range: self.linear_velocity_range,
            };

            let reward_value = stability_reward_function(&state, &reward_config);
            // godot_print!("Reward: {}", reward_value);
            let reward = Tensor::<Backend, 1>::from_data([reward_value], &DEVICE).reshape([1, 1]);

            if self.episode.steps % (self.train_every_n_frames as usize) == 0 && self.train {
                let losses = adhdp.train(ADHDPTrainData::Step(ADHDPStepTrainData {
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

        let mut u = adhdp.act(x.clone());
        // Add temporally correlated exploration noise, decayed over training time.
        if self.use_noise {
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
        perform_action(u.clone(), helicopter);

        self.previous_step = Some(StepData { x, u });
    }
}

#[godot_api]
impl Agent {
    /// Reset the episode.
    #[func]
    fn reset_episode(&mut self) {
        let (helicopter_rotation, helicopter_linear_velocity, helicopter_angular_velocity) =
            self.get_helicopter_starting_state();

        let adhdp = self.adhdp.as_mut().unwrap();
        let mut game = self.game.clone().unwrap();
        game.bind_mut().reset(
            helicopter_rotation,
            helicopter_linear_velocity,
            helicopter_angular_velocity,
        );

        self.episode_count += 1;
        if self.episode_count % 100 == 0 {
            adhdp.save(&self.run_directory, &self.episode_count.to_string());
        }

        // Save the best policy seen so far, so a later divergence can't destroy it.
        // (actor_best / critic_best in the run directory.)
        if self.episode.accumulated_reward > self.best_episode_reward {
            self.best_episode_reward = self.episode.accumulated_reward;
            adhdp.save(&self.run_directory, "best");
        }

        // Restart the exploration process so each episode explores independently.
        if let Some(ou) = self.ou_noise.as_mut() {
            ou.reset();
        }

        self.previous_episode = Some(self.episode.clone());
        self.episode = Episode::new(self.episode_count);
        self.previous_step = None;
    }

    fn get_helicopter_starting_state(&mut self) -> (Vector3, Vector3, Vector3) {
        // Sample initial ranges from uniform distributions within the specified ranges
        let roll_range_rad = self.initial_roll_range_deg.to_radians();
        let pitch_range_rad = self.initial_pitch_range_deg.to_radians();
        let linear_velocity_range = self.initial_linear_velocity_range;
        let angular_velocity_range_rad = self.initial_angular_velocity_range_deg.to_radians();

        let roll_distribution = Uniform::new(-roll_range_rad as f64, roll_range_rad as f64);
        let pitch_distribution = Uniform::new(-pitch_range_rad as f64, pitch_range_rad as f64);
        let linear_velocity_distribution =
            Uniform::new(-linear_velocity_range as f64, linear_velocity_range as f64);
        let angular_velocity_distribution = Uniform::new(
            -angular_velocity_range_rad as f64,
            angular_velocity_range_rad as f64,
        );

        // Sample from distributions using the noise source directly.
        // probability::prelude::Uniform provides a sample method that accepts a Source.
        let noise_source = self.noise_source.as_mut().unwrap();

        let roll = roll_distribution.sample(noise_source) as f32;
        let pitch = pitch_distribution.sample(noise_source) as f32;
        let linear_velocity_x = linear_velocity_distribution.sample(noise_source) as f32;
        let linear_velocity_y = linear_velocity_distribution.sample(noise_source) as f32;
        let linear_velocity_z = linear_velocity_distribution.sample(noise_source) as f32;
        let angular_velocity_x = angular_velocity_distribution.sample(noise_source) as f32;
        let angular_velocity_y = angular_velocity_distribution.sample(noise_source) as f32;
        let angular_velocity_z = angular_velocity_distribution.sample(noise_source) as f32;

        let rotation = Vector3::new(roll, 0.0, pitch);
        let linear_velocity = Vector3::new(linear_velocity_x, linear_velocity_y, linear_velocity_z);
        let angular_velocity =
            Vector3::new(angular_velocity_x, angular_velocity_y, angular_velocity_z);

        (rotation, linear_velocity, angular_velocity)
    }
}
