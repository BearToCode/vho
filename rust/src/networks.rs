use burn::module::Param;
use burn::nn::{Linear, LinearConfig, Relu, Tanh};
use burn::prelude::*;

#[derive(Module, Debug)]
pub struct CriticModel<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
    l3: Linear<B>,
    relu: Relu,
}

impl<B: Backend> CriticModel<B> {
    pub fn new(state_dim: usize, action_dim: usize, device: &B::Device) -> Self {
        Self {
            l1: LinearConfig::new(state_dim + action_dim, 8).init(device),
            l2: LinearConfig::new(8, 8).init(device),
            l3: LinearConfig::new(8, 1).init(device),
            relu: Relu::new(),
        }
    }

    /// Propagates the model.
    pub fn forward(&self, state: Tensor<B, 2>, action: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = Tensor::cat(vec![state, action], 1);
        let x = self.relu.forward(self.l1.forward(x));
        let x = self.relu.forward(self.l2.forward(x));
        self.l3.forward(x) // scalar J
    }

    /// Performs a Polyak update of the model, with respect to an online model.
    #[allow(dead_code)]
    pub fn polyak_update(&mut self, online: &Self, tau: f32) {
        self.l1 = polyak_linear(&self.l1, &online.l1, tau);
        self.l2 = polyak_linear(&self.l2, &online.l2, tau);
        self.l3 = polyak_linear(&self.l3, &online.l3, tau);
    }
}

#[derive(Module, Debug)]
pub struct ActorModel<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
    l3: Linear<B>,
    relu: Relu,
    tanh: Tanh,
}

impl<B: Backend> ActorModel<B> {
    pub fn new(state_dim: usize, action_dim: usize, device: &B::Device) -> Self {
        Self {
            l1: LinearConfig::new(state_dim, 8).init(device),
            l2: LinearConfig::new(8, 8).init(device),
            l3: LinearConfig::new(8, action_dim).init(device),
            relu: Relu::new(),
            tanh: Tanh::new(),
        }
    }

    /// Propagates the model.
    pub fn forward(&self, state: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.relu.forward(self.l1.forward(state));
        let x = self.relu.forward(self.l2.forward(x));
        self.tanh.forward(self.l3.forward(x)) // in [-1, 1]
    }

    /// Performs a Polyak update of the model, with respect to an online model.
    #[allow(dead_code)]
    pub fn polyak_update(&mut self, online: &Self, tau: f32) {
        self.l1 = polyak_linear(&self.l1, &online.l1, tau);
        self.l2 = polyak_linear(&self.l2, &online.l2, tau);
        self.l3 = polyak_linear(&self.l3, &online.l3, tau);
    }
}

/// Perform a Polyak linear update on a target and online linear layers.
/// The final layer has theta = theta_target * (1 - tau) + theta_online * tau
///
/// * `target` - Layer from the target model.
/// * `online` - Layer from the online model.
/// * `tau` - Linear interpolation coefficient.
fn polyak_linear<B: Backend>(target: &Linear<B>, online: &Linear<B>, tau: f32) -> Linear<B> {
    let weight = target.weight.val().mul_scalar(1.0 - tau) + online.weight.val().mul_scalar(tau);
    let weight = weight.set_require_grad(false);

    let weight = Param::from_tensor(weight);

    let bias = match (&target.bias, &online.bias) {
        (Some(tb), Some(ob)) => {
            let b = tb.val().mul_scalar(1.0 - tau) + ob.val().mul_scalar(tau);
            let b = b.set_require_grad(false);

            Some(Param::from_tensor(b))
        }
        _ => None,
    };

    Linear { weight, bias }
}
