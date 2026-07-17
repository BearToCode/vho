use godot::prelude::*;
use rand::Rng;

use crate::helicopter::Helicopter;

/// A uniform random vector with each component drawn from `[-magnitude, magnitude]`.
fn random_offset(rng: &mut impl Rng, magnitude: f32) -> Vector3 {
    if magnitude <= 0.0 {
        return Vector3::ZERO;
    }

    Vector3::new(
        rng.random_range(-magnitude..=magnitude),
        rng.random_range(-magnitude..=magnitude),
        rng.random_range(-magnitude..=magnitude),
    )
}

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

    #[export_group(name = "Episode Randomization")]
    #[export]
    /// Maximum random offset applied to the starting position, per axis, in meters.
    reset_position_noise: f32,
    #[export]
    /// Maximum random starting linear velocity, per axis, in m/s.
    reset_linear_velocity_noise: f32,
    #[export]
    /// Maximum random starting attitude offset, per axis, in radians.
    reset_rotation_noise: f32,
    #[export]
    /// Maximum random starting angular velocity, per axis, in rad/s.
    reset_angular_velocity_noise: f32,
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

            reset_position_noise: 1.0,
            reset_linear_velocity_noise: 0.5,
            reset_rotation_noise: 0.05,
            reset_angular_velocity_noise: 0.1,
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

    /// Reset the helicopter to its starting state.
    ///
    /// * `randomize` - Perturb the starting state within the configured ranges. The
    ///   target stays at the nominal starting pose, so a perturbed start turns "hold
    ///   still" into "fly back", which is the only way the agent ever sees states off
    ///   its own trajectory. Evaluation episodes pass `false` so their score stays
    ///   comparable across a run.
    pub fn reset(&mut self, randomize: bool) {
        let mut rng = rand::rng();
        let scale = |noise: f32| if randomize { noise } else { 0.0 };

        let position = self.helicopter_initial_position
            + random_offset(&mut rng, scale(self.reset_position_noise));
        let rotation = self.helicopter_initial_rotation
            + random_offset(&mut rng, scale(self.reset_rotation_noise));
        let linear_velocity = self.helicopter_initial_linear_velocity
            + random_offset(&mut rng, scale(self.reset_linear_velocity_noise));
        let angular_velocity = self.helicopter_initial_angular_velocity
            + random_offset(&mut rng, scale(self.reset_angular_velocity_noise));

        let helicopter = self
            .helicopter
            .as_mut()
            .expect("Game is missing the helicopter");

        helicopter.set_position(position);
        helicopter.set_linear_velocity(linear_velocity);
        helicopter.set_rotation(rotation);
        helicopter.set_angular_velocity(angular_velocity);

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
