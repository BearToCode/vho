use godot::prelude::*;

use crate::helicopter::Helicopter;

#[derive(GodotClass)]
#[class(base=Node3D)]
pub struct Game {
    base: Base<Node3D>,

    #[export]
    pub helicopter: Option<Gd<Helicopter>>,
}

#[godot_api]
impl INode3D for Game {
    fn init(base: Base<Node3D>) -> Self {
        Self {
            base,

            helicopter: None,
        }
    }

    fn ready(&mut self) {
        let _helicopter = self
            .helicopter
            .as_ref()
            .expect("Game is missing the helicopter");
    }
}

impl Game {
    pub fn reset(
        &mut self,
        helicopter_rotation: Vector3,
        helicopter_linear_velocity: Vector3,
        helicopter_angular_velocity: Vector3,
    ) {
        let helicopter = self
            .helicopter
            .as_mut()
            .expect("Game is missing the helicopter");

        helicopter.set_position(Vector3::ZERO);
        helicopter.set_linear_velocity(helicopter_linear_velocity);
        helicopter.set_rotation(helicopter_rotation);
        helicopter.set_angular_velocity(helicopter_angular_velocity);
        helicopter.bind_mut().lat_flapping = 0.0;
        helicopter.bind_mut().lon_flapping = 0.0;
    }
}
