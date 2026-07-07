#![recursion_limit = "256"]

mod agent;
mod game;
mod helicopter;
mod networks;
mod noise;
mod replay_buffer;
mod ring;
mod track;

use godot::prelude::*;

struct MyExtension;

#[gdextension]
unsafe impl ExtensionLibrary for MyExtension {}
