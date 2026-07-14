use crate::rl::{
    Backend, DEVICE,
    action::ACTION_DIM,
    networks::{ActorModel, CriticModel},
    state::STATE_DIM,
};
use burn::{
    Tensor,
    grad_clipping::GradientClippingConfig,
    module::Module,
    optim::{Adam, AdamConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
};
use godot::prelude::*;

pub struct ADHDPConfig {
    /// Bellman equation gamma parameter.
    pub gamma: f32,
    /// Critic model learning rate.
    pub critic_learning_rate: f64,
    /// Actor model learning rate.
    pub actor_learning_rate: f64,
    /// Target network update rate.
    pub tau: f32,
    /// Whether to use target networks for the actor and critic models.
    pub use_target_networks: bool,
    /// Hidden layer sizes for the actor model.
    pub actor_hidden_layers: Vec<usize>,
    /// Hidden layer sizes for the critic model.
    pub critic_hidden_layers: Vec<usize>,
}

/// Losses from one ADHDP train step.
pub struct ADHDPLosses {
    pub critic_loss: f32,
    pub actor_loss: f32,
}

pub struct ADHDPTerminalTrainData {
    pub x: Tensor<Backend, 2>,
    pub u: Tensor<Backend, 2>,
    pub reward: Tensor<Backend, 2>,
}

pub struct ADHDPStepTrainData {
    pub x: Tensor<Backend, 2>,
    pub u: Tensor<Backend, 2>,
    pub reward: Tensor<Backend, 2>,
    pub x_next: Tensor<Backend, 2>,
}

#[allow(dead_code)]
pub enum ADHDPTrainData {
    #[allow(dead_code)]
    Step(ADHDPStepTrainData),
    #[allow(dead_code)]
    Terminal(ADHDPTerminalTrainData),
}

/// Action Dependent Heuristic Dynamic Programming RL implementation.
pub struct ADHDP {
    /// The online actor model.
    actor: ActorModel<Backend>,
    /// The online critic model.
    critic: CriticModel<Backend>,
    /// The target actor model, used for stabilizing training.
    target_actor: ActorModel<Backend>,
    /// The target critic model, used for stabilizing training.
    target_critic: CriticModel<Backend>,

    /// Adam optimizer for the actor model.
    actor_optimizer: OptimizerAdaptor<Adam, ActorModel<Backend>, Backend>,
    /// Adam optimizer for the critic model.
    critic_optimizer: OptimizerAdaptor<Adam, CriticModel<Backend>, Backend>,

    /// Configuration of ADHDP.
    pub config: ADHDPConfig,
}

impl ADHDP {
    pub fn new(config: ADHDPConfig) -> Self {
        let actor =
            ActorModel::<Backend>::new(STATE_DIM, ACTION_DIM, &config.actor_hidden_layers, &DEVICE);
        let critic = CriticModel::<Backend>::new(
            STATE_DIM,
            ACTION_DIM,
            &config.critic_hidden_layers,
            &DEVICE,
        );

        let target_actor = actor.clone();
        let target_critic = critic.clone();

        let actor_optimizer = AdamConfig::new()
            .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
            .build()
            .into();
        let critic_optimizer = AdamConfig::new()
            .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
            .build()
            .into();

        return Self {
            actor,
            critic,
            target_actor,
            target_critic,
            actor_optimizer,
            critic_optimizer,
            config,
        };
    }

    /// Train the models, returning their losses for the provided data.
    pub fn train(&mut self, data: ADHDPTrainData) -> ADHDPLosses {
        // 1. critic update
        let critic_update_actor = if self.config.use_target_networks {
            &self.target_actor
        } else {
            &self.actor
        };
        let critic_update_critic = if self.config.use_target_networks {
            &self.target_critic
        } else {
            &self.critic
        };

        let (x, u, target) = match data {
            ADHDPTrainData::Step(step) => {
                let u_next = critic_update_actor.forward(step.x_next.clone()).detach();
                let j_next = critic_update_critic.forward(step.x_next, u_next).detach();
                let target = step.reward + j_next.mul_scalar(self.config.gamma);

                (step.x, step.u, target)
            }
            ADHDPTrainData::Terminal(terminal) => {
                let target = terminal.reward;

                (terminal.x, terminal.u, target)
            }
        };

        let j_pred = self.critic.forward(x.clone(), u.clone());
        let critic_loss = (target - j_pred).powf_scalar(2.0).mean();
        let critic_loss_value = critic_loss
            .clone()
            .into_data()
            .to_vec::<f32>()
            .expect("Failed to read critic loss")[0];

        let grads = critic_loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.critic);
        self.critic = self.critic_optimizer.step(
            self.config.critic_learning_rate,
            self.critic.clone(),
            grads,
        );

        // 2. actor update (maximize J)
        let u_pred = self.actor.forward(x.clone());
        let j_for_actor = self.critic.forward(x, u_pred).mean();
        let actor_loss = j_for_actor.neg();
        let actor_loss_value = actor_loss
            .clone()
            .into_data()
            .to_vec::<f32>()
            .expect("Failed to read actor loss")[0]
            .abs();

        let grads = actor_loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.actor);
        self.actor =
            self.actor_optimizer
                .step(self.config.actor_learning_rate, self.actor.clone(), grads);

        // 3. Polyak averaging of target networks
        if self.config.use_target_networks {
            self.target_actor
                .polyak_update(&self.actor, self.config.tau);
            self.target_critic
                .polyak_update(&self.critic, self.config.tau);
        }

        return ADHDPLosses {
            actor_loss: actor_loss_value,
            critic_loss: critic_loss_value,
        };
    }

    /// Perform one action based on a state.
    pub fn act(&self, x: Tensor<Backend, 2>) -> Tensor<Backend, 2> {
        self.actor.forward(x)
    }

    /// Save the current actor and critic model weights to disk.
    pub fn save(&self, dir: &str, suffix: &str) {
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

        if let Err(e) = self
            .actor
            .clone()
            .save_file(format!("{}/actor_{}", dir, suffix), &recorder)
        {
            godot_print!("Failed to save actor: {e}");
        }
        if let Err(e) = self
            .critic
            .clone()
            .save_file(format!("{}/critic_{}", dir, suffix), &recorder)
        {
            godot_print!("Failed to save critic: {e}");
        }

        godot_print!("Saved model to file");
    }

    /// Load actor weights from disk, replacing the current model
    pub fn load_actor(&mut self, path: &str) {
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

        match self.actor.clone().load_file(path, &recorder, &DEVICE) {
            Ok(actor) => self.actor = actor,
            Err(e) => godot_print!("Failed to load actor: {e}"),
        }

        godot_print!("Loaded critic model from {path}");
    }

    /// Load critic weights from disk, replacing the current model
    pub fn load_critic(&mut self, path: &str) {
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        match self.critic.clone().load_file(path, &recorder, &DEVICE) {
            Ok(critic) => self.critic = critic,
            Err(e) => godot_print!("Failed to load critic: {e}"),
        }

        godot_print!("Loaded critic model from {path}");
    }
}
