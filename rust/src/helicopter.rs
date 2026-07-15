use godot::classes::{IRigidBody3D, MeshInstance3D, RigidBody3D};
use godot::prelude::*;

#[derive(GodotClass)]
#[class(base=RigidBody3D)]
pub struct Helicopter {
    base: Base<RigidBody3D>,

    // Flapping dynamics
    lon_flapping: f32,
    lat_flapping: f32,

    #[export_group(name = "Inputs")]
    #[export]
    /// Collective input, range: [0.0, 1.0]
    pub collective: f32,
    #[export]
    /// Lateral cyclic input, range: [-1.0, 1.0]
    pub lateral_cyclic: f32,
    #[export]
    /// Longitudinal cyclic input, range: [-1.0, 1.0]
    pub longitudinal_cyclic: f32,
    #[export]
    /// Tail rotor cyclic input, range: [-1.0, 1.0]
    pub tail_rotor_cyclic: f32,

    #[export_group(name = "Parameters")]
    #[export]
    pub air_density: f32, // [kg/m^3]
    #[export]
    pub longitudinal_gain: f32, // [rad]
    #[export]
    pub lateral_gain: f32, // [rad]
    #[export]
    pub longitudinal_stiffness: f32, // [Nm/rad]
    #[export]
    pub lateral_stiffness: f32, // [Nm/rad]
    #[export_subgroup(name = "Main Rotor")]
    #[export]
    pub main_rotor_radius: f32, // [m]
    #[export]
    pub main_rotor_speed: f32, // [rad/s]
    #[export]
    pub max_thrust: f32, // [N]
    #[export]
    pub torque_coefficient: f32, // [-]
    #[export]
    pub lock_number: f32, // [-]
    #[export_subgroup(name = "Tail Rotor")]
    #[export]
    pub tail_rotor_max_thrust: f32, // [N]
    #[export]
    pub tail_rotor_arm: f32, // [m]

    #[export_group(name = "Meshes")]
    #[export]
    main_rotor_mesh: Option<Gd<MeshInstance3D>>,
    #[export]
    tail_rotor_mesh: Option<Gd<MeshInstance3D>>,

    #[export_group(name = "Extra")]
    #[export]
    animate: bool,
}

#[godot_api]
impl IRigidBody3D for Helicopter {
    fn init(base: Base<RigidBody3D>) -> Self {
        Self {
            base,

            lon_flapping: 0.0,
            lat_flapping: 0.0,

            collective: 0.0,
            lateral_cyclic: 0.0,
            longitudinal_cyclic: 0.0,
            tail_rotor_cyclic: 0.0,

            air_density: 1.225,
            longitudinal_gain: 0.175,
            lateral_gain: 0.175,
            longitudinal_stiffness: 25_000.0,
            lateral_stiffness: 25_000.0,
            main_rotor_radius: 4.0,
            main_rotor_speed: 50.0,
            max_thrust: 18_700.0,
            torque_coefficient: 0.00257,
            lock_number: 6.0,
            tail_rotor_max_thrust: 1_500.0,
            tail_rotor_arm: 5.5,

            main_rotor_mesh: None,
            tail_rotor_mesh: None,

            animate: true,
        }
    }

    fn ready(&mut self) {}

    fn physics_process(&mut self, delta: f32) {
        // Retrieve state from the physics engine
        let local_to_global = self.base().get_transform().basis;
        let global_to_local = local_to_global.inverse();
        let linear_velocity = global_to_local * self.base().get_linear_velocity();
        let angular_velocity = global_to_local * self.base().get_angular_velocity();

        // We need to convert from Godot's coordinate system to the helicopter's coordinate system
        let (_u, _v, _w) = (linear_velocity.x, -linear_velocity.z, -linear_velocity.y);
        let (p, q, _r) = (angular_velocity.x, -angular_velocity.z, -angular_velocity.y);

        // Advance flapping dynamics
        let flap_time_const = 16.0 / (self.lock_number * self.main_rotor_speed);
        let lon_flapping_dot = (-self.lon_flapping - flap_time_const * q
            + self.longitudinal_gain * self.longitudinal_cyclic)
            / flap_time_const;
        let lat_flapping_dot = (-self.lat_flapping - flap_time_const * p
            + self.lateral_gain * self.lateral_cyclic)
            / flap_time_const;
        // Simple Euler integration for flapping dynamics
        self.lon_flapping += lon_flapping_dot * delta;
        self.lat_flapping += lat_flapping_dot * delta;

        // Calculate forces and torques
        let thrust = self.max_thrust * self.collective;
        let reaction_torque = self.torque_coefficient * thrust.powf(1.5);
        let tail_rotor_thrust = self.tail_rotor_max_thrust * self.tail_rotor_cyclic;

        let f_x = -thrust * self.lon_flapping;
        let f_y = thrust * self.lat_flapping + tail_rotor_thrust;
        let f_z = -thrust;

        let m_x = self.lateral_stiffness * self.lat_flapping;
        let m_y = self.longitudinal_stiffness * self.lon_flapping;
        let m_z = -reaction_torque + self.tail_rotor_arm * tail_rotor_thrust;

        let local_force = Vector3::new(f_x, -f_z, -f_y);
        let local_torque = Vector3::new(m_x, -m_z, -m_y);

        self.base_mut()
            .apply_central_force(local_to_global * local_force);
        self.base_mut().apply_torque(local_to_global * local_torque);

        // Animate the rotors if enabled
        if self.animate {
            self.animate_main_rotor_rotation(delta);
            self.animate_tail_rotor_rotation(delta);
        }
    }
}

#[godot_api]
impl Helicopter {
    #[func]
    fn animate_main_rotor_rotation(&mut self, delta: f32) {
        if let Some(main_rotor_mesh) = &mut self.main_rotor_mesh {
            let mut rotation = main_rotor_mesh.get_rotation();
            rotation.y += self.main_rotor_speed * delta;
            main_rotor_mesh.set_rotation(rotation);
        }
    }

    #[func]
    fn animate_tail_rotor_rotation(&mut self, delta: f32) {
        if let Some(tail_rotor_mesh) = &mut self.tail_rotor_mesh {
            let mut rotation = tail_rotor_mesh.get_rotation();
            let tail_rotor_speed = self.main_rotor_speed * 6.0;
            rotation.x += tail_rotor_speed * delta;
            tail_rotor_mesh.set_rotation(rotation);
        }
    }
}
