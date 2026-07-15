use godot::prelude::*;

use crate::helicopter::Helicopter;

#[derive(GodotClass)]
#[class(base=Node3D)]
pub struct Game {
    base: Base<Node3D>,

    helicopter_initial_position: Vector3,
    helicopter_initial_rotation: Vector3,
    helicopter_initial_linear_velocity: Vector3,
    helicopter_initial_angular_velocity: Vector3,

    #[export]
    pub helicopter: Option<Gd<Helicopter>>,
}

#[godot_api]
impl INode3D for Game {
    fn init(base: Base<Node3D>) -> Self {
        Self {
            base,

            helicopter_initial_position: Vector3::ZERO,
            helicopter_initial_rotation: Vector3::ZERO,
            helicopter_initial_linear_velocity: Vector3::ZERO,
            helicopter_initial_angular_velocity: Vector3::ZERO,

            helicopter: None,
        }
    }

    fn ready(&mut self) {
        let helicopter = self
            .helicopter
            .as_ref()
            .expect("Game is missing the helicopter");

        self.helicopter_initial_position = helicopter.get_global_position();
        self.helicopter_initial_rotation = helicopter.get_global_rotation();
        self.helicopter_initial_linear_velocity = helicopter.get_linear_velocity();
        self.helicopter_initial_angular_velocity = helicopter.get_angular_velocity();
    }
}

impl Game {
    pub fn reset(&mut self) {
        let helicopter = self
            .helicopter
            .as_mut()
            .expect("Game is missing the helicopter");

        helicopter.set_position(self.helicopter_initial_position);
        helicopter.set_linear_velocity(self.helicopter_initial_linear_velocity);
        helicopter.set_rotation(self.helicopter_initial_rotation);
        helicopter.set_angular_velocity(self.helicopter_initial_angular_velocity);
    }
}
