mod action;
mod adhdp;
mod agent;
mod episode;
mod networks;
mod replay;
mod reward;
mod state;

use burn::backend::{Autodiff, Flex, flex::FlexDevice};
// use burn::backend::{Autodiff, Wgpu, wgpu::WgpuDevice};

/// The Burn backend to use. Wgpu is a backend that runs on the GPU.
pub type Backend = Autodiff<Flex>;
// pub type Backend = Autodiff<Wgpu>;

pub const DEVICE: FlexDevice = FlexDevice;
// pub const DEVICE: WgpuDevice = WgpuDevice::DiscreteGpu(0);
