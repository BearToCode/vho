use std::ops::{Index, IndexMut};

use godot::classes::{IRigidBody3D, MeshInstance3D, PhysicsDirectBodyState3D, RigidBody3D};
use godot::prelude::*;

use nalgebra::{SMatrix, SVector};

type StateMatrix = SMatrix<f32, 8, 8>;
type InputMatrix = SMatrix<f32, 8, 4>;
type StateVector = SVector<f32, 8>;
type InputVector = SVector<f32, 4>;

/// Imperial state vector, used for linear dynamics.
/// All velocities are in a body-fixed reference frame.
pub enum HelicopterStateComponent {
    /// [ft/s] forward velocity
    U = 0,
    // [ft/s]  vertical velocity
    W,
    // [rad/s] pitch rate
    Q,
    // [rad]   pitch angle
    Theta,
    // [ft/s]  lateral velocity
    V,
    // [rad/s] roll rate
    P,
    // [rad/s] yaw rate
    R,
    // [rad]   roll angle
    Phi,
}

/// Helicopter control inputs.
pub enum HelicopterInputComponent {
    /// longitudinal cyclic
    UY = 0,
    /// collective
    UC,
    /// lateral cyclic
    UX,
    /// tail rotor cyclic
    UZ,
}

impl Index<HelicopterStateComponent> for StateVector {
    type Output = f32;
    fn index(&self, index: HelicopterStateComponent) -> &Self::Output {
        &self[index as usize]
    }
}

impl IndexMut<HelicopterStateComponent> for StateVector {
    fn index_mut(&mut self, index: HelicopterStateComponent) -> &mut Self::Output {
        return &mut self[index as usize];
    }
}

impl Index<HelicopterInputComponent> for InputVector {
    type Output = f32;
    fn index(&self, index: HelicopterInputComponent) -> &Self::Output {
        &self[index as usize]
    }
}

impl IndexMut<HelicopterInputComponent> for InputVector {
    fn index_mut(&mut self, index: HelicopterInputComponent) -> &mut Self::Output {
        return &mut self[index as usize];
    }
}

struct HelicopterLinearModel {
    a: StateMatrix,
    b: InputMatrix,
}

impl HelicopterLinearModel {
    #[rustfmt::skip]
    pub fn new() -> Self {
        const G: f32 = 32.174;        // Gravity in ft/s^2
        const U0: f32 = 20.0 * 3.281; // 20 m/s

        Self {
            a: StateMatrix::from_row_slice(&[
                //         |   u   |   w   |   q   | theta |   v   |   p   |   r   |  phi  |
                /*   u   */  -0.01 ,  0.0  ,  0.0  ,   -G  ,  0.0  ,  0.0  ,  0.0  ,  0.0  ,
                /*   w   */   0.0  , -1.0  ,   U0  ,  0.0  ,  0.0  ,  0.0  ,  0.0  ,  0.0  ,
                /*   q   */   0.0  ,  0.0  , -3.0  ,  0.0  ,  0.0  ,  0.0  ,  0.0  ,  0.0  ,
                /* theta */   0.0  ,  0.0  ,  1.0  ,  0.0  ,  0.0  ,  0.0  ,  0.0  ,  0.0  ,
                /*   v   */   0.0  ,  0.0  ,  0.0  ,  0.0  , -0.02 ,  0.0  ,  -U0  ,    G  ,
                /*   p   */   0.0  ,  0.0  ,  0.0  ,  0.0  ,  0.0  , -5.0  ,  0.0  ,  0.0  ,
                /*   r   */   0.0  ,  0.0  ,  0.0  ,  0.0  ,  0.0  ,  0.0  , -1.0  ,  0.0  ,
                /*  phi  */   0.0  ,  0.0  ,  0.0  ,  0.0  ,  0.0  ,  1.0  ,  0.0  ,  0.0  ,
            ]),
            b: InputMatrix::from_row_slice(&[
                //         |  u_y  |  u_c  |  u_x  |  u_z  |
                /*   u   */   0.0  ,  0.0  ,  0.0  ,  0.0  ,
                /*   w   */   0.0  , -7.0  ,  0.0  ,  0.0  ,
                /*   q   */  -0.7  ,  0.0  ,  0.0  ,  0.0  ,
                /* theta */   0.0  ,  0.0  ,  0.0  ,  0.0  ,
                /*   v   */   0.0  ,  0.0  ,  0.0  ,  0.0  ,
                /*   p   */   0.0  ,  0.0  ,  2.0  ,  0.0  ,
                /*   r   */   0.0  ,  0.0  ,  0.0  , -3.0  ,
                /*  phi  */   0.0  ,  0.0  ,  0.0  ,  0.0  ,
            ]),
        }
    }
}

/// Convert fts to meters
pub fn ft_to_m(value: f32) -> f32 {
    value * 0.3048
}

/// Convert meters to fts
pub fn m_to_ft(value: f32) -> f32 {
    value / 0.3048
}

#[derive(GodotClass)]
#[class(base=RigidBody3D)]
pub struct Helicopter {
    base: Base<RigidBody3D>,

    state_vector: StateVector,
    inputs_vector: InputVector,

    linear_model: HelicopterLinearModel,

    #[export_group(name = "Inputs")]
    #[export]
    pub collective: f32,
    #[export]
    pub lateral_cyclic: f32,
    #[export]
    pub longitudinal_cyclic: f32,
    #[export]
    pub tail_rotor_cyclic: f32,
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
            main_rotor_mesh: None,
            tail_rotor_mesh: None,
            collective: 0.0,
            lateral_cyclic: 0.0,
            longitudinal_cyclic: 0.0,
            tail_rotor_cyclic: 0.0,
            state_vector: StateVector::zeros(),
            inputs_vector: InputVector::zeros(),
            linear_model: HelicopterLinearModel::new(),
            animate: true,
        }
    }

    fn ready(&mut self) {}

    fn integrate_forces(&mut self, state: Option<Gd<PhysicsDirectBodyState3D>>) {
        if let Some(s) = state {
            self.retrieve_state(s.clone());
            self.retrieve_inputs();

            let state_vector_derivative: StateVector =
                self.linear_model.a * self.state_vector + self.linear_model.b * self.inputs_vector;

            self.apply_accelerations(s.clone(), state_vector_derivative);

            if self.animate {
                let delta = s.get_step();
                self.animate_main_rotor_rotation(delta);
                self.animate_tail_rotor_rotation(delta);
            }
        }
    }
}

#[godot_api]
impl Helicopter {
    pub fn get_state_vector(&self) -> &StateVector {
        return &self.state_vector;
    }

    fn retrieve_state(&mut self, state: Gd<PhysicsDirectBodyState3D>) {
        let transform = state.get_transform();
        let global_to_local = transform.basis.inverse();

        let local_linear_velocity = global_to_local * state.get_linear_velocity();
        let local_angular_velocity = global_to_local * state.get_angular_velocity();

        let rotation = transform.basis.get_euler();

        //  | Local Reference Frame | Godot | Model |
        //  | --------------------- | ----- | ----- |
        //  | Forward:              | -z    |   U/P |
        //  | Right:                | +x    |   W/Q |
        //  | Up:                   | +y    |  -V/R |

        type SV = HelicopterStateComponent;
        self.state_vector[SV::U] = m_to_ft(-local_linear_velocity.z);
        self.state_vector[SV::V] = m_to_ft(local_linear_velocity.x);
        self.state_vector[SV::W] = m_to_ft(-local_linear_velocity.y);

        self.state_vector[SV::Phi] = -rotation.z;
        self.state_vector[SV::Theta] = rotation.x;

        self.state_vector[SV::P] = -local_angular_velocity.z;
        self.state_vector[SV::Q] = local_angular_velocity.x;
        self.state_vector[SV::R] = -local_angular_velocity.y;
    }

    fn apply_accelerations(
        &mut self,
        state: Gd<PhysicsDirectBodyState3D>,
        state_vector_derivative: StateVector,
    ) {
        let transform = state.get_transform();
        let local_to_global = transform.basis;

        //  | Local Reference Frame | Godot | Model |
        //  | --------------------- | ----- | ----- |
        //  | Forward:              | -z    |   U/P |
        //  | Right:                | +x    |   W/Q |
        //  | Up:                   | +y    |   V/R |

        type SV = HelicopterStateComponent;
        let local_linear_acceleration = Vector3::new(
            ft_to_m(state_vector_derivative[SV::V]),
            ft_to_m(-state_vector_derivative[SV::W]),
            ft_to_m(-state_vector_derivative[SV::U]),
        );

        let local_angular_acceleration = Vector3::new(
            state_vector_derivative[SV::Q],
            -state_vector_derivative[SV::R],
            -state_vector_derivative[SV::P],
        );

        let mut base = self.base_mut();
        let mass = base.get_mass();
        let inertia = base.get_inertia();

        let local_force = local_linear_acceleration * mass;
        let local_torque = local_angular_acceleration * inertia;

        let global_force = local_to_global * local_force;
        let global_torque = local_to_global * local_torque;

        // godot_print!("global force: {global_force}");
        // godot_print!("global torque: {global_torque}");

        base.apply_force(global_force);
        base.apply_torque(global_torque);
    }

    #[func]
    fn retrieve_inputs(&mut self) {
        type Inputs = HelicopterInputComponent;
        self.inputs_vector[Inputs::UX] = self.lateral_cyclic;
        self.inputs_vector[Inputs::UC] = self.collective;
        self.inputs_vector[Inputs::UY] = self.longitudinal_cyclic;
        self.inputs_vector[Inputs::UZ] = self.tail_rotor_cyclic;
    }

    #[func]
    fn animate_main_rotor_rotation(&mut self, delta: f32) {
        const OMEGA: f32 = 10.0;
        if let Some(main_rotor_mesh) = &mut self.main_rotor_mesh {
            let mut rotation = main_rotor_mesh.get_rotation();
            rotation.y += OMEGA * delta;
            main_rotor_mesh.set_rotation(rotation);
        }
    }

    #[func]
    fn animate_tail_rotor_rotation(&mut self, delta: f32) {
        const OMEGA: f32 = 30.0;
        if let Some(tail_rotor_mesh) = &mut self.tail_rotor_mesh {
            let mut rotation = tail_rotor_mesh.get_rotation();
            rotation.x += OMEGA * delta;
            tail_rotor_mesh.set_rotation(rotation);
        }
    }
}
