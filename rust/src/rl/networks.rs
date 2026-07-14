use burn::module::Param;
use burn::nn::{Gelu, Initializer, Linear, LinearConfig};
use burn::prelude::*;

#[derive(Module, Debug)]
pub struct CriticModel<B: Backend> {
    layers: Vec<Linear<B>>,
    activation: Gelu,
}

impl<B: Backend> CriticModel<B> {
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        hidden_layers: &Vec<usize>,
        device: &B::Device,
    ) -> Self {
        let mut layers = Vec::new();
        let mut prev_dim = state_dim + action_dim;

        for dim in hidden_layers {
            layers.push(
                LinearConfig::new(prev_dim, *dim)
                    // Xavier/Glorot is the standard pairing for tanh/GELU-family
                    // activations; He/Kaiming (burn's default) assumes ReLU.
                    .with_initializer(Initializer::XavierUniform { gain: 1.0 })
                    .init(device),
            );
            prev_dim = *dim;
        }

        // Output layer: linear, no activation, Q can be any real value.
        // Small init gain here keeps early Q-values near 0 rather than
        // starting with a large-magnitude, arbitrary bias.
        layers.push(
            LinearConfig::new(prev_dim, 1)
                .with_initializer(Initializer::XavierUniform { gain: 0.1 })
                .init(device),
        );

        Self {
            layers,
            activation: Gelu::new(),
        }
    }

    pub fn forward(&self, state: Tensor<B, 2>, action: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = Tensor::cat(vec![state, action], 1);
        for layer in &self.layers[..self.layers.len() - 1] {
            x = self.activation.forward(layer.forward(x));
        }
        self.layers.last().unwrap().forward(x)
    }

    #[allow(dead_code)]
    pub fn polyak_update(&mut self, online: &Self, tau: f32) {
        for (target_layer, online_layer) in self.layers.iter_mut().zip(online.layers.iter()) {
            *target_layer = polyak_linear(target_layer, online_layer, tau);
        }
    }
}
use burn::nn::Tanh;

#[derive(Module, Debug)]
pub struct ActorModel<B: Backend> {
    layers: Vec<Linear<B>>,
    activation: Gelu,
    output_activation: Tanh,
}

impl<B: Backend> ActorModel<B> {
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        hidden_layers: &Vec<usize>,
        device: &B::Device,
    ) -> Self {
        let mut layers = Vec::new();
        let mut prev_dim = state_dim;

        for dim in hidden_layers {
            layers.push(
                LinearConfig::new(prev_dim, *dim)
                    .with_initializer(Initializer::Zeros)
                    .init(device),
            );
            prev_dim = *dim;
        }

        layers.push(
            LinearConfig::new(prev_dim, action_dim)
                .with_initializer(Initializer::Zeros)
                .init(device),
        );

        Self {
            layers,
            activation: Gelu::new(),
            output_activation: Tanh::new(),
        }
    }

    pub fn forward(&self, state: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = state;
        for layer in &self.layers[..self.layers.len() - 1] {
            x = self.activation.forward(layer.forward(x));
        }
        let raw = self.layers.last().unwrap().forward(x);
        self.output_activation.forward(raw)
    }

    #[allow(dead_code)]
    pub fn polyak_update(&mut self, online: &Self, tau: f32) {
        for (target_layer, online_layer) in self.layers.iter_mut().zip(online.layers.iter()) {
            *target_layer = polyak_linear(target_layer, online_layer, tau);
        }
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
