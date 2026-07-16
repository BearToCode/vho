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
    /// The helicopter's pose at scene start, used as the target hover point.
    pub fn helicopter_initial_position(&self) -> Vector3 {
        self.helicopter_initial_position
    }

    pub fn reset(&mut self) {
        let helicopter = self
            .helicopter
            .as_mut()
            .expect("Game is missing the helicopter");

        helicopter.set_position(self.helicopter_initial_position);
        helicopter.set_linear_velocity(self.helicopter_initial_linear_velocity);
        helicopter.set_rotation(self.helicopter_initial_rotation);
        helicopter.set_angular_velocity(self.helicopter_initial_angular_velocity);

        let mut helicopter_bind = helicopter.bind_mut();

        // Flapping is integrated state, and part of the agent's observation. Leaving it
        // to carry over means each episode starts from wherever the last one ended.
        helicopter_bind.lon_flapping = 0.0;
        helicopter_bind.lat_flapping = 0.0;

        // The agent returns before acting on the frame it resets, but the helicopter
        // still integrates afterwards, so stale controls would apply one frame of the
        // previous episode's final action to the new initial state.
        helicopter_bind.collective = 0.0;
        helicopter_bind.lateral_cyclic = 0.0;
        helicopter_bind.longitudinal_cyclic = 0.0;
        helicopter_bind.tail_rotor_cyclic = 0.0;
    }
}
