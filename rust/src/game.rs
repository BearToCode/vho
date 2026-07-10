use godot::prelude::*;

use crate::{helicopter::Helicopter, track::Track};

#[derive(GodotClass)]
#[class(base=Node3D)]
pub struct Game {
    base: Base<Node3D>,

    distance_to_ring: f32,
    track_progress: f32,
    rings_passed: usize,
    game_ended: bool,

    helicopter_initial_position: Vector3,
    helicopter_initial_rotation: Vector3,
    helicopter_initial_linear_velocity: Vector3,
    helicopter_initial_angular_velocity: Vector3,

    #[export]
    pub track: Option<Gd<Track>>,
    #[export]
    pub helicopter: Option<Gd<Helicopter>>,
}

#[godot_api]
impl INode3D for Game {
    fn init(base: Base<Node3D>) -> Self {
        Self {
            base,

            distance_to_ring: 0.0,
            track_progress: 0.0,
            rings_passed: 0,
            game_ended: false,

            helicopter_initial_position: Vector3::ZERO,
            helicopter_initial_rotation: Vector3::ZERO,
            helicopter_initial_linear_velocity: Vector3::ZERO,
            helicopter_initial_angular_velocity: Vector3::ZERO,

            track: None,
            helicopter: None,
        }
    }

    fn ready(&mut self) {
        let mut this = self.to_gd();

        let helicopter = self
            .helicopter
            .as_ref()
            .expect("Game is missing the helicopter");

        self.helicopter_initial_position = helicopter.get_global_position();
        self.helicopter_initial_rotation = helicopter.get_global_rotation();
        self.helicopter_initial_linear_velocity = helicopter.get_linear_velocity();
        self.helicopter_initial_angular_velocity = helicopter.get_angular_velocity();

        let track = self.track.as_mut().expect("Game is missing the track");
        let mut track_bind = track.bind_mut();

        let rings_count = track_bind.rings.len();

        track_bind
            .signals()
            .ring_passed()
            .connect(move |_ring_idx| {
                let mut this_bind = this.bind_mut();

                this_bind.distance_to_ring = this_bind.compute_distance_to_next_ring();
                this_bind.rings_passed += 1;

                if this_bind.rings_passed == rings_count {
                    this_bind.game_ended = true;
                }
            });

        drop(track_bind);

        self.distance_to_ring = self.compute_distance_to_next_ring();
    }

    fn physics_process(&mut self, _delta: f32) {
        if self.game_ended {
            return;
        }

        let new_distance_to_ring = self.compute_distance_to_next_ring();

        let difference = self.distance_to_ring - new_distance_to_ring;

        self.track_progress += difference;
        self.distance_to_ring = new_distance_to_ring;
    }
}

impl Game {
    fn compute_distance_to_next_ring(&mut self) -> f32 {
        let track_bind = self
            .track
            .as_mut()
            .expect("Game is missing the track")
            .bind_mut();

        let helicopter = self
            .helicopter
            .as_mut()
            .expect("Game is missing the helicopter");

        let current_ring = &track_bind.current_ring().expect("Current ring is not set");

        return (current_ring.get_global_position() - helicopter.get_global_position()).length();
    }

    pub fn reset(&mut self) {
        let helicopter = self
            .helicopter
            .as_mut()
            .expect("Game is missing the helicopter");
        let track = self.track.as_mut().expect("Game is missing the track");

        helicopter.set_position(self.helicopter_initial_position);
        helicopter.set_linear_velocity(self.helicopter_initial_linear_velocity);
        helicopter.set_rotation(self.helicopter_initial_rotation);
        helicopter.set_angular_velocity(self.helicopter_initial_angular_velocity);

        track.bind_mut().current_ring_index = 0;

        self.distance_to_ring = self.compute_distance_to_next_ring();
        self.track_progress = 0.0;
        self.rings_passed = 0;
        self.game_ended = false;
    }

    pub fn track_progress(&self) -> f32 {
        self.track_progress
    }

    pub fn rings_passed(&self) -> usize {
        self.rings_passed
    }
}
