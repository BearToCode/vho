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
    tensor::Distribution,
};
use godot::prelude::*;

type ActorOptimizer = OptimizerAdaptor<Adam, ActorModel<Backend>, Backend>;
type CriticOptimizer = OptimizerAdaptor<Adam, CriticModel<Backend>, Backend>;

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
    /// TD3: number of critic updates between each actor and target network update.
    pub policy_delay: usize,
    /// TD3: standard deviation of the target policy smoothing noise.
    pub target_noise_std: f32,
    /// TD3: absolute bound applied to the target policy smoothing noise.
    pub target_noise_clip: f32,
}

/// Losses from one TD3 train step.
pub struct ADHDPLosses {
    /// Mean of the two critic losses.
    pub critic_loss: f32,
    /// `None` on steps where the policy delay skipped the actor update.
    pub actor_loss: Option<f32>,
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

/// Read a scalar loss tensor back to the CPU.
fn loss_value(loss: Tensor<Backend, 1>) -> f32 {
    loss.into_data().to_vec::<f32>().expect("Failed to read loss")[0]
}

/// TD3 (Twin Delayed DDPG) actor-critic implementation.
///
/// Three differences from plain DDPG, all aimed at the failure where the actor learns
/// to exploit errors in the critic and drags both networks down with it:
///   1. Two independently initialized critics, whose *minimum* forms the Bellman
///      target, so an error in one of them cannot inflate the target on its own.
///   2. The actor and target networks update only every `policy_delay` critic updates,
///      so the actor chases a value estimate that has had time to settle.
///   3. Noise on the target action, so a sharp peak in the critic's approximation
///      cannot be exploited by the actor.
pub struct ADHDP {
    /// The online actor model.
    actor: ActorModel<Backend>,
    /// First online critic.
    critic_1: CriticModel<Backend>,
    /// Second online critic. Its *independent* random initialization is what makes its
    /// errors differ from `critic_1`, which is the whole point of taking the minimum.
    critic_2: CriticModel<Backend>,

    /// The target actor model, used for stabilizing training.
    target_actor: ActorModel<Backend>,
    /// Target counterpart of `critic_1`.
    target_critic_1: CriticModel<Backend>,
    /// Target counterpart of `critic_2`.
    target_critic_2: CriticModel<Backend>,

    /// Adam optimizer for the actor model.
    actor_optimizer: ActorOptimizer,
    /// Adam optimizer for `critic_1`.
    critic_1_optimizer: CriticOptimizer,
    /// Adam optimizer for `critic_2`.
    critic_2_optimizer: CriticOptimizer,

    /// Number of `train` calls so far, used to apply the policy delay.
    train_steps: usize,

    /// Configuration of TD3.
    pub config: ADHDPConfig,
}

impl ADHDP {
    pub fn new(config: ADHDPConfig) -> Self {
        let actor =
            ActorModel::<Backend>::new(STATE_DIM, ACTION_DIM, &config.actor_hidden_layers, &DEVICE);
        let critic_1 = CriticModel::<Backend>::new(
            STATE_DIM,
            ACTION_DIM,
            &config.critic_hidden_layers,
            &DEVICE,
        );
        let critic_2 = CriticModel::<Backend>::new(
            STATE_DIM,
            ACTION_DIM,
            &config.critic_hidden_layers,
            &DEVICE,
        );

        let target_actor = actor.clone();
        let target_critic_1 = critic_1.clone();
        let target_critic_2 = critic_2.clone();

        let actor_optimizer = AdamConfig::new()
            .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
            .build()
            .into();
        let critic_1_optimizer = AdamConfig::new()
            .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
            .build()
            .into();
        let critic_2_optimizer = AdamConfig::new()
            .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
            .build()
            .into();

        return Self {
            actor,
            critic_1,
            critic_2,
            target_actor,
            target_critic_1,
            target_critic_2,
            actor_optimizer,
            critic_1_optimizer,
            critic_2_optimizer,
            train_steps: 0,
            config,
        };
    }

    /// Train the models, returning their losses for the provided data.
    pub fn train(&mut self, data: ADHDPTrainData) -> ADHDPLosses {
        self.train_steps += 1;

        // 1. Build the Bellman target.
        let (x, u, target) = match data {
            ADHDPTrainData::Step(step) => {
                let (target_actor, target_critic_1, target_critic_2) =
                    if self.config.use_target_networks {
                        (
                            &self.target_actor,
                            &self.target_critic_1,
                            &self.target_critic_2,
                        )
                    } else {
                        (&self.actor, &self.critic_1, &self.critic_2)
                    };

                // Target policy smoothing.
                let mut u_next = target_actor.forward(step.x_next.clone());
                if self.config.target_noise_std > 0.0 {
                    let noise = u_next
                        .random_like(Distribution::Normal(
                            0.0,
                            self.config.target_noise_std as f64,
                        ))
                        .clamp(-self.config.target_noise_clip, self.config.target_noise_clip);
                    u_next = u_next + noise;
                }
                let u_next = u_next.clamp(-1.0, 1.0).detach();

                // Clipped double-Q: take the more pessimistic of the two estimates.
                let j_1 = target_critic_1.forward(step.x_next.clone(), u_next.clone());
                let j_2 = target_critic_2.forward(step.x_next, u_next);
                let j_next = j_1.min_pair(j_2).detach();

                let target = step.reward + j_next.mul_scalar(self.config.gamma);

                (step.x, step.u, target)
            }
            ADHDPTrainData::Terminal(terminal) => (terminal.x, terminal.u, terminal.reward),
        };

        // 2. Update both critics against that same target.
        let j_1_pred = self.critic_1.forward(x.clone(), u.clone());
        let critic_1_loss = (target.clone() - j_1_pred).powf_scalar(2.0).mean();
        let critic_1_loss_value = loss_value(critic_1_loss.clone());

        let grads = critic_1_loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.critic_1);
        self.critic_1 = self.critic_1_optimizer.step(
            self.config.critic_learning_rate,
            self.critic_1.clone(),
            grads,
        );

        let j_2_pred = self.critic_2.forward(x.clone(), u);
        let critic_2_loss = (target - j_2_pred).powf_scalar(2.0).mean();
        let critic_2_loss_value = loss_value(critic_2_loss.clone());

        let grads = critic_2_loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.critic_2);
        self.critic_2 = self.critic_2_optimizer.step(
            self.config.critic_learning_rate,
            self.critic_2.clone(),
            grads,
        );

        // 3. Delayed actor update (maximize J), followed by the target networks.
        let actor_loss = if self.train_steps % self.config.policy_delay.max(1) == 0 {
            let u_pred = self.actor.forward(x.clone());
            let j_for_actor = self.critic_1.forward(x, u_pred).mean();
            let actor_loss = j_for_actor.neg();
            let actor_loss_value = loss_value(actor_loss.clone()).abs();

            let grads = actor_loss.backward();
            let grads = GradientsParams::from_grads(grads, &self.actor);
            self.actor =
                self.actor_optimizer
                    .step(self.config.actor_learning_rate, self.actor.clone(), grads);

            if self.config.use_target_networks {
                self.target_actor
                    .polyak_update(&self.actor, self.config.tau);
                self.target_critic_1
                    .polyak_update(&self.critic_1, self.config.tau);
                self.target_critic_2
                    .polyak_update(&self.critic_2, self.config.tau);
            }

            Some(actor_loss_value)
        } else {
            None
        };

        return ADHDPLosses {
            critic_loss: (critic_1_loss_value + critic_2_loss_value) / 2.0,
            actor_loss,
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
            .critic_1
            .clone()
            .save_file(format!("{}/critic_{}", dir, suffix), &recorder)
        {
            godot_print!("Failed to save critic 1: {e}");
        }
        if let Err(e) = self
            .critic_2
            .clone()
            .save_file(format!("{}/critic2_{}", dir, suffix), &recorder)
        {
            godot_print!("Failed to save critic 2: {e}");
        }

        godot_print!("Saved model to file");
    }

    /// Load actor weights from disk, replacing the current model.
    pub fn load_actor(&mut self, path: &str) {
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

        match self.actor.clone().load_file(path, &recorder, &DEVICE) {
            Ok(actor) => {
                self.actor = actor;
                // The target has to start out equal to the online network. Leaving it
                // at its random initialization feeds garbage into every Bellman target
                // and destroys the weights that were just loaded.
                self.target_actor = self.actor.clone();
                godot_print!("Loaded actor model from {path}");
            }
            Err(e) => godot_print!("Failed to load actor: {e}"),
        }
    }

    /// Load the first critic's weights from disk, replacing the current model.
    pub fn load_critic(&mut self, path: &str) {
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

        match self.critic_1.clone().load_file(path, &recorder, &DEVICE) {
            Ok(critic) => {
                self.critic_1 = critic;
                self.target_critic_1 = self.critic_1.clone();
                godot_print!("Loaded critic model from {path}");
            }
            Err(e) => godot_print!("Failed to load critic: {e}"),
        }
    }

    /// Load the second critic's weights from disk, replacing the current model.
    /// Leaving this unset is safe: a freshly initialized `critic_2` predicts near zero,
    /// which is above any real (negative) return, so `min` simply defers to `critic_1`
    /// until `critic_2` has caught up.
    pub fn load_critic_2(&mut self, path: &str) {
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

        match self.critic_2.clone().load_file(path, &recorder, &DEVICE) {
            Ok(critic) => {
                self.critic_2 = critic;
                self.target_critic_2 = self.critic_2.clone();
                godot_print!("Loaded second critic model from {path}");
            }
            Err(e) => godot_print!("Failed to load second critic: {e}"),
        }
    }
}
