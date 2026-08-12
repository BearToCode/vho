use burn::{
    module::Module,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
};
use godot::prelude::*;
use probability::prelude::*;

use crate::{
    game::Game,
    rl::{
        Backend, DEVICE,
        action::{ACTION_DIM, perform_action},
        networks::ActorModel,
        state::{
            AgentStateComponent, AgentStateVector, STATE_DIM, StateNormalizationConfig,
            get_agent_state, normalize_state,
        },
    },
};

#[derive(GodotClass)]
#[class(base=Node3D)]
pub struct Evaluator {
    base: Base<Node3D>,
    episode_count: u32,
    successful_episodes: u32,
    episode_time: f32,
    noise_source: source::Default,
    actor: Option<ActorModel<Backend>>,

    #[export]
    /// Reference to the game manager.
    game: Option<Gd<Game>>,
    #[export]
    max_episode_time: f32,
    #[export(file)]
    #[var(hint = FILE)]
    actor_model_path: GString,
    #[export]
    actor_hidden_layers: Array<i64>,

    #[export_group(name = "Spawn Attitude")]
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

    #[export_group(name = "Stability Thresholds")]
    #[export]
    /// Maximum allowed roll angle in degrees before the episode is considered failed.
    max_roll_deg: f32,
    #[export]
    /// Maximum allowed pitch angle in degrees before the episode is considered failed.
    max_pitch_deg: f32,
    #[export]
    /// Maximum allowed linear velocity in meters per second before the episode is considered failed.
    max_linear_velocity: f32,
    #[export]
    /// Maximum allowed angular velocity in degrees per second before the episode is considered failed.
    max_angular_velocity_deg: f32,

    #[export_group(name = "State Normalization")]
    #[export]
    /// Linear velocity scale for state normalization.
    linear_velocity_scale: f32,
    #[export]
    /// Angular velocity scale for state normalization.
    angular_velocity_scale: f32,
    #[export]
    /// Position error scale for state normalization.
    position_error_scale: f32,
    #[export]
    /// Angle scale for state normalization.
    angle_scale: f32,
    #[export]
    /// Flap angle scale for state normalization.
    flap_angle_scale: f32,
}

#[godot_api]
impl INode3D for Evaluator {
    fn init(base: Base<Node3D>) -> Self {
        Self {
            base,
            episode_count: 0,
            successful_episodes: 0,
            episode_time: 0.0,
            noise_source: source::default(42),
            actor: None,

            actor_model_path: GString::new(),
            actor_hidden_layers: Array::new(),
            game: None,
            max_episode_time: 0.0,

            initial_roll_range_deg: 0.0,
            initial_pitch_range_deg: 0.0,
            initial_linear_velocity_range: 0.0,
            initial_angular_velocity_range_deg: 0.0,

            max_roll_deg: 0.0,
            max_pitch_deg: 0.0,
            max_linear_velocity: 0.0,
            max_angular_velocity_deg: 0.0,

            linear_velocity_scale: 0.0,
            angular_velocity_scale: 0.0,
            position_error_scale: 0.0,
            angle_scale: 0.0,
            flap_angle_scale: 0.0,
        }
    }

    fn ready(&mut self) {
        if self.game.is_none() {
            panic!("No game set!");
        }

        if self.actor_model_path.is_empty() {
            panic!("No actor model set!");
        }

        self.actor = Some(ActorModel::<Backend>::new(
            STATE_DIM,
            ACTION_DIM,
            &self
                .actor_hidden_layers
                .iter_shared()
                .map(|v| v as usize)
                .collect::<Vec<usize>>(),
            &DEVICE,
        ));

        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

        self.actor = Some(
            self.actor
                .clone()
                .unwrap()
                .load_file(
                    self.actor_model_path.to_string().as_str(),
                    &recorder,
                    &DEVICE,
                )
                .unwrap_or_else(|e| {
                    panic!(
                        "Failed to load actor model from {}: {}",
                        self.actor_model_path, e
                    )
                }),
        );
        godot_print!("Loaded actor model from {}", self.actor_model_path);
    }

    fn physics_process(&mut self, delta: f32) {
        self.episode_time += delta;

        let game = self.game.clone().unwrap();
        let helicopter = game.bind().helicopter.clone().unwrap();
        let state = get_agent_state(game);

        if self.episode_time >= self.max_episode_time {
            if self.is_stable(&state) {
                self.successful_episodes += 1;
            }

            self.reset_episode();
        }

        let normalization_config = StateNormalizationConfig {
            angular_velocity_scale: self.angular_velocity_scale,
            linear_velocity_scale: self.linear_velocity_scale,
            angle_scale: self.angle_scale,
            flap_angle_scale: self.flap_angle_scale,
            position_error_scale: self.position_error_scale,
        };

        let x = normalize_state(&state, &normalization_config, &DEVICE);
        let actor = self.actor.as_ref().unwrap();
        let u = actor.forward(x);
        let u = u.clamp(-1.0, 1.0);
        perform_action(u, helicopter);
    }
}

#[godot_api]
impl Evaluator {
    #[func]
    fn reset_episode(&mut self) {
        let (helicopter_rotation, helicopter_linear_velocity, helicopter_angular_velocity) =
            self.get_helicopter_starting_state();

        let mut game = self.game.clone().unwrap();
        game.bind_mut().reset(
            helicopter_rotation,
            helicopter_linear_velocity,
            helicopter_angular_velocity,
        );

        self.episode_count += 1;
        self.episode_time = 0.0;

        godot_print!(
            "Evaluation: {} | Stable: {} | Success rate: {:.2}%",
            self.episode_count,
            self.successful_episodes,
            (self.successful_episodes as f32 / self.episode_count as f32) * 100.0
        );
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
        let noise_source = &mut self.noise_source;
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

    fn is_stable(&self, state: &AgentStateVector) -> bool {
        let roll_deg = state[AgentStateComponent::RotationAngleX].to_degrees();
        let pitch_deg = state[AgentStateComponent::RotationAngleZ].to_degrees();

        let linear_velocity_x = state[AgentStateComponent::LinearVelocityX];
        let linear_velocity_y = state[AgentStateComponent::LinearVelocityY];
        let linear_velocity_z = state[AgentStateComponent::LinearVelocityZ];
        let linear_velocity =
            (linear_velocity_x.powi(2) + linear_velocity_y.powi(2) + linear_velocity_z.powi(2))
                .sqrt();

        let angular_velocity_x = state[AgentStateComponent::AngularVelocityX].to_degrees();
        let angular_velocity_y = state[AgentStateComponent::AngularVelocityY].to_degrees();
        let angular_velocity_z = state[AgentStateComponent::AngularVelocityZ].to_degrees();
        let angular_velocity_deg =
            (angular_velocity_x.powi(2) + angular_velocity_y.powi(2) + angular_velocity_z.powi(2))
                .sqrt();

        let is_stable = roll_deg.abs() <= self.max_roll_deg
            && pitch_deg.abs() <= self.max_pitch_deg
            && linear_velocity <= self.max_linear_velocity
            && angular_velocity_deg <= self.max_angular_velocity_deg;

        if !is_stable {
            godot_print!(
                "Unstable! Roll: {:.2}, Pitch: {:.2}, Linear Velocity: {:.2}, Angular Velocity: {:.2}",
                roll_deg,
                pitch_deg,
                linear_velocity,
                angular_velocity_deg
            );
        }

        is_stable
    }
}
