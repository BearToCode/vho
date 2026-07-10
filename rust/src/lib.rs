#![recursion_limit = "256"]

mod game;
mod helicopter;
mod ring;
mod rl;
mod track;

use godot::prelude::*;

struct MyExtension;

#[gdextension]
unsafe impl ExtensionLibrary for MyExtension {}
