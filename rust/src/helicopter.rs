use std::ops::{Index, IndexMut};

use godot::classes::{IRigidBody3D, MeshInstance3D, PhysicsDirectBodyState3D, RigidBody3D};
use godot::prelude::*;

use nalgebra::{SMatrix, SVector};

type Matrix8x8f = SMatrix<f32, 8, 8>;
type Matrix8x4f = SMatrix<f32, 8, 4>;
type Vector8f = SVector<f32, 8>;
type Vector4f = SVector<f32, 4>;

/// Imperial state vector, used for linear dynamics.
/// All velocities are in a body-fixed reference frame.
enum HelicopterStateVectorComponent {
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
enum HelicopterInputsComponent {
    /// longitudinal cyclic
    UY = 0,
    /// collective
    UC,
    /// lateral cyclic
    UX,
    /// tail rotor cyclic
    UZ,
}

impl Index<HelicopterStateVectorComponent> for Vector8f {
    type Output = f32;
    fn index(&self, index: HelicopterStateVectorComponent) -> &Self::Output {
        &self[index as usize]
    }
}

impl IndexMut<HelicopterStateVectorComponent> for Vector8f {
    fn index_mut(&mut self, index: HelicopterStateVectorComponent) -> &mut Self::Output {
        return &mut self[index as usize];
    }
}

impl Index<HelicopterInputsComponent> for Vector4f {
    type Output = f32;
    fn index(&self, index: HelicopterInputsComponent) -> &Self::Output {
        &self[index as usize]
    }
}

impl IndexMut<HelicopterInputsComponent> for Vector4f {
    fn index_mut(&mut self, index: HelicopterInputsComponent) -> &mut Self::Output {
        return &mut self[index as usize];
    }
}

struct HelicopterLinearModel {
    a: Matrix8x8f,
    b: Matrix8x4f,
}

impl HelicopterLinearModel {
    #[rustfmt::skip]
    pub fn new() -> Self {
        const G: f32 = 32.174;        // Gravity in ft/s^2
        const U0: f32 = 20.0 * 3.281; // 20 m/s

        Self {
            a: Matrix8x8f::from_row_slice(&[
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
            b: Matrix8x4f::from_row_slice(&[
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

    /// Convert fts to meters
    pub fn ft_to_m(value: f32) -> f32 {
        value * 0.3048
    }

    /// Convert meters to fts
    pub fn m_to_ft(value: f32) -> f32 {
        value / 0.3048
    }
}

#[derive(GodotClass)]
#[class(base=RigidBody3D)]
struct Helicopter {
    base: Base<RigidBody3D>,

    state_vector: Vector8f,
    inputs_vector: Vector4f,

    linear_model: HelicopterLinearModel,

    #[export_group(name = "Inputs")]
    #[export]
    collective: f32,
    #[export]
    lateral_cyclic: f32,
    #[export]
    longitudinal_cyclic: f32,
    #[export]
    tail_rotor_cyclic: f32,
    #[export_group(name = "Meshes")]
    #[export]
    main_rotor_mesh: Option<Gd<MeshInstance3D>>,
    #[export]
    tail_rotor_mesh: Option<Gd<MeshInstance3D>>,
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
            state_vector: Vector8f::zeros(),
            inputs_vector: Vector4f::zeros(),
            linear_model: HelicopterLinearModel::new(),
        }
    }

    fn ready(&mut self) {}

    fn integrate_forces(&mut self, state: Option<Gd<PhysicsDirectBodyState3D>>) {
        if let Some(s) = state {
            self.retrieve_state(s.clone());
            self.retrieve_inputs();

            let state_vector_derivative: Vector8f =
                self.linear_model.a * self.state_vector + self.linear_model.b * self.inputs_vector;

            self.apply_accelerations(s.clone(), state_vector_derivative);
        }
    }
}

#[godot_api]
impl Helicopter {
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

        type SV = HelicopterStateVectorComponent;
        self.state_vector[SV::U] = HelicopterLinearModel::m_to_ft(-local_linear_velocity.z);
        self.state_vector[SV::V] = HelicopterLinearModel::m_to_ft(local_linear_velocity.x);
        self.state_vector[SV::W] = HelicopterLinearModel::m_to_ft(-local_linear_velocity.y);

        self.state_vector[SV::Phi] = -rotation.z;
        self.state_vector[SV::Theta] = rotation.x;

        self.state_vector[SV::P] = -local_angular_velocity.z;
        self.state_vector[SV::Q] = local_angular_velocity.x;
        self.state_vector[SV::R] = -local_angular_velocity.y;
    }

    fn apply_accelerations(
        &mut self,
        state: Gd<PhysicsDirectBodyState3D>,
        state_vector_derivative: Vector8f,
    ) {
        let transform = state.get_transform();
        let local_to_global = transform.basis;

        //  | Local Reference Frame | Godot | Model |
        //  | --------------------- | ----- | ----- |
        //  | Forward:              | -z    |   U/P |
        //  | Right:                | +x    |   W/Q |
        //  | Up:                   | +y    |   V/R |

        type SV = HelicopterStateVectorComponent;
        let local_linear_acceleration = Vector3::new(
            HelicopterLinearModel::ft_to_m(state_vector_derivative[SV::V]),
            HelicopterLinearModel::ft_to_m(-state_vector_derivative[SV::W]),
            HelicopterLinearModel::ft_to_m(-state_vector_derivative[SV::U]),
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
        type Inputs = HelicopterInputsComponent;
        self.inputs_vector[Inputs::UX] = self.lateral_cyclic;
        self.inputs_vector[Inputs::UC] = self.collective;
        self.inputs_vector[Inputs::UY] = self.longitudinal_cyclic;
        self.inputs_vector[Inputs::UZ] = self.tail_rotor_cyclic;
    }

    #[func]
    fn set_main_rotor_rotation(&mut self, angle: f32) {
        if let Some(main_rotor_mesh) = &mut self.main_rotor_mesh {
            main_rotor_mesh.set_rotation(Vector3::new(0.0, angle, 0.0));
        }
    }

    #[func]
    fn set_tail_rotor_rotation(&mut self, angle: f32) {
        if let Some(tail_rotor_mesh) = &mut self.tail_rotor_mesh {
            tail_rotor_mesh.set_rotation(Vector3::new(angle, 0.0, 0.0));
        }
    }
}
