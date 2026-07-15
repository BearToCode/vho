#![recursion_limit = "256"]

mod game;
mod helicopter;
mod rl;

use godot::prelude::*;

struct MyExtension;

#[gdextension]
unsafe impl ExtensionLibrary for MyExtension {}
