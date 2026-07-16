mod action;
mod adhdp;
mod agent;
mod episode;
mod networks;
mod replay_buffer;
mod reward;
mod state;

use burn::backend::{Autodiff, Flex, flex::FlexDevice};

/// The Burn backend to use. Flex is a lightweight Rust backend that runs on the CPU.
pub type Backend = Autodiff<Flex>;

pub const DEVICE: FlexDevice = FlexDevice;
