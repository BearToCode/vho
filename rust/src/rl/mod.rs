mod action;
mod adhdp;
mod agent;
mod episode;
mod networks;
mod reward;
mod state;

use burn::backend::{Autodiff, Wgpu, wgpu::WgpuDevice};

/// The Burn backend to use. Wgpu is a backend that runs on the GPU.
pub type Backend = Autodiff<Wgpu>;

pub const DEVICE: WgpuDevice = WgpuDevice::DiscreteGpu(0);
