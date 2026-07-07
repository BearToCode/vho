use burn::{
    Tensor,
    backend::{
        Autodiff,
        flex::{Flex, FlexDevice},
    },
    optim::{Adam, AdamConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
};
use godot::{global::rad_to_deg, prelude::*};

use crate::{
    game::Game,
    helicopter::{Helicopter, HelicopterStateVectorComponent},
    networks::{ActorModel, CriticModel},
    ring::Ring,
};

/// The Burn backend to use. Flex is a lightweight Rust backend that runs on the CPU.
type Backend = Autodiff<Flex>;

/// Dimension of the state input of the model.
const STATE_DIM: usize = 14;
/// Dimension of the action output of the model.
const ACTION_DIM: usize = 4;

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

    // RL params
    /// Actor model learning rate.
    actor_lr: f64,
    /// Critic model learning rate.
    critic_lr: f64,
    /// Gamma parameter of the Bellman function.
    gamma: f32,
    /// Tau parameter for Polyak updates.
    tau: f32,
    /// Maximum duration of each episode in seconds.
    max_episode_time: f32,

    // RL states
    /// Current episode time.
    episode_time: f32,
    /// Current progression along the track.
    track_progress: f32,
    /// Current number of rings passed.
    rings_passed: usize,
    /// Previous state tensor.
    prev_x: Option<Tensor<Backend, 2>>,
    /// Previous action tensor.
    prev_u: Option<Tensor<Backend, 2>>,

    // Godot params
    #[export]
    game: Option<Gd<Game>>,
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
        let device = FlexDevice;

        let actor = ActorModel::new(STATE_DIM, ACTION_DIM, &device);
        let critic = CriticModel::new(STATE_DIM, ACTION_DIM, &device);

        let actor_target = actor.clone();
        let critic_target = critic.clone();

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

            // RL states
            episode_time: 0.0,
            track_progress: 0.0,
            rings_passed: 0,
            prev_u: None,
            prev_x: None,

            // Godot params
            game: None,
            collective_range: 5.0,
            lateral_cyclic_range: 1.0,
            longitudinal_cyclic_range: 1.0,
            tail_rotor_cyclic_range: 0.3,
        }
    }

    fn physics_process(&mut self, delta: f32) {
        self.episode_time += delta;
        let mut game: Gd<Game> = self.game.clone().unwrap();
        let helicopter: Gd<Helicopter> = game.bind().helicopter.clone().unwrap();

        if self.episode_time > self.max_episode_time {
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

            self.act(u.clone(), helicopter.clone());

            (x, u)
        };

        // Check if the helicopter is crashing
        if self.crashing(helicopter.clone()) {
            let crash_penalty = 100.0;
            let reward =
                Tensor::<Backend, 1>::from_floats([-crash_penalty], &self.device).reshape([1, 1]);

            if let (Some(prev_x), Some(prev_u)) = (&self.prev_x, &self.prev_u) {
                self.train_step(
                    prev_x.clone(),
                    prev_u.clone(),
                    reward,
                    CriticUpdate::Terminal,
                );
            }

            game.bind_mut().reset();
            self.reset_episode();
            return;
        }

        // Train all the networks
        if let Some(prev_x) = self.prev_x.as_ref()
            && let Some(prev_u) = self.prev_u.as_ref()
        {
            // Compute reward
            let new_progress = game.bind().track_progress();
            let new_rings_passed = game.bind().rings_passed();

            let progress_reward = (new_progress - self.track_progress) * 5.0;
            let rings_reward = (new_rings_passed - self.rings_passed) as f32 * 10.0;

            let living_penalty = 0.05 * delta;

            let reward = progress_reward + rings_reward - living_penalty;
            let reward = Tensor::<Backend, 1>::from_floats([reward], &self.device).reshape([1, 1]);

            // Perform normal train step
            self.train_step(
                prev_x.clone(),
                prev_u.clone(),
                reward,
                CriticUpdate::Normal(x.clone()),
            );

            self.track_progress = new_progress;
            self.rings_passed = new_rings_passed;
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
        // Scale angular velocities for better gradients
        type SV = HelicopterStateVectorComponent;
        helicopter_state[SV::Q] *= 10.0;
        helicopter_state[SV::P] *= 10.0;
        helicopter_state[SV::R] *= 10.0;

        // Helicopter necessary transform data
        let helicopter_position = helicopter.get_global_position();
        let global_to_local = helicopter.get_transform().basis.inverse();

        // Second 3 components: Helicopter position relative to fist ring, in local reference frame
        let current_ring_position = current_ring.get_global_position();
        let current_ring_relative_position =
            global_to_local * (current_ring_position - helicopter_position);

        // Third 3 components: Helicopter position relative to second ring, in local reference frame
        let next_ring_relative_position = if let Some(next_ring) = next_ring {
            let next_ring_position = next_ring.get_global_position();
            global_to_local * (next_ring_position - helicopter_position)
        } else {
            // Use same location as current ring if it's the last one
            current_ring_relative_position
        };

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

    /// Perform one training step.
    pub fn train_step(
        &mut self,
        x: Tensor<Backend, 2>,
        u: Tensor<Backend, 2>,
        reward: Tensor<Backend, 2>,
        critic_update: CriticUpdate,
    ) {
        // 1. critic update
        let target = match critic_update {
            CriticUpdate::Normal(x_next) => {
                let u_next = self.actor_target.forward(x_next.clone()).detach();
                let j_next = self.critic_target.forward(x_next, u_next).detach();
                reward + j_next.mul_scalar(self.gamma)
            }
            CriticUpdate::Terminal => reward,
        };

        let j_pred = self.critic.forward(x.clone(), u.clone());
        let critic_loss = (target - j_pred).powf_scalar(2.0).mean();

        let grads = critic_loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.critic);
        self.critic = self
            .critic_optimizer
            .step(self.critic_lr, self.critic.clone(), grads);

        // 2. actor update (maximize J)
        let u_pred = self.actor.forward(x.clone());
        let j_for_actor = self.critic.forward(x, u_pred).mean();
        let actor_loss = j_for_actor.neg();

        let grads = actor_loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.actor);
        self.actor = self
            .actor_optimizer
            .step(self.actor_lr, self.actor.clone(), grads);

        // 3. Polyak averaging
        self.actor_target.polyak_update(&self.actor, self.tau);
        self.critic_target.polyak_update(&self.critic, self.tau);
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

    /// Reset the episode.
    fn reset_episode(&mut self) {
        godot_print!("Episode completed. Final progress {0}", self.track_progress);

        self.episode_time = 0.0;
        self.track_progress = 0.0;
        self.rings_passed = 0;
        self.prev_u = None;
        self.prev_x = None;
    }
}
