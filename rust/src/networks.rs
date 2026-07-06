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
            l1: LinearConfig::new(state_dim + action_dim, 128).init(device),
            l2: LinearConfig::new(128, 128).init(device),
            l3: LinearConfig::new(128, 1).init(device),
            relu: Relu::new(),
        }
    }

    pub fn forward(&self, state: Tensor<B, 2>, action: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = Tensor::cat(vec![state, action], 1);
        let x = self.relu.forward(self.l1.forward(x));
        let x = self.relu.forward(self.l2.forward(x));
        self.l3.forward(x) // scalar J
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
            l1: LinearConfig::new(state_dim, 128).init(device),
            l2: LinearConfig::new(128, 128).init(device),
            l3: LinearConfig::new(128, action_dim).init(device),
            relu: Relu::new(),
            tanh: Tanh::new(),
        }
    }

    pub fn forward(&self, state: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.relu.forward(self.l1.forward(state));
        let x = self.relu.forward(self.l2.forward(x));
        self.tanh.forward(self.l3.forward(x)) // in [-1, 1]
    }
}
