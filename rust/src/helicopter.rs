use godot::classes::{ClassDb, Engine, IRigidBody3D, MeshInstance3D, RigidBody3D};
use godot::prelude::*;

#[derive(GodotClass)]
#[class(base=RigidBody3D, tool)]
struct Helicopter {
    base: Base<RigidBody3D>,
    main_rotor_mesh: Option<Gd<MeshInstance3D>>,
    tail_rotor_mesh: Option<Gd<MeshInstance3D>>,

    #[export]
    tail_rotor_position: Vector3,
    #[export]
    debug: bool,
}

#[godot_api]
impl IRigidBody3D for Helicopter {
    fn init(base: Base<RigidBody3D>) -> Self {
        Self {
            base,
            main_rotor_mesh: None,
            tail_rotor_mesh: None,
            tail_rotor_position: Vector3::ZERO,
            debug: true,
        }
    }

    fn ready(&mut self) {
        let main_rotor_mesh = self.base().get_node_as::<MeshInstance3D>("main-rotor");
        let tail_rotor_mesh = self.base().get_node_as::<MeshInstance3D>("tail-rotor");

        self.main_rotor_mesh = Some(main_rotor_mesh);
        self.tail_rotor_mesh = Some(tail_rotor_mesh);

        godot_print!("Main rotor mesh: {:?}", self.main_rotor_mesh);
        godot_print!("Tail rotor mesh: {:?}", self.tail_rotor_mesh);

        self.set_main_rotor_rotation(0.2);
        self.set_tail_rotor_rotation(0.2);
    }

    fn physics_process(&mut self, delta: f64) {
        if Engine::singleton().is_editor_hint() {
            // let mut draw = ClassDb::singleton()
            //     .instantiate("Draw3D")
            //     .to::<Gd<Node3D>>();

            // self.base_mut().add_child(&draw);

            // draw.call(
            //     "cube",
            //     vslice![
            //         self.tail_rotor_position,
            //         Basis::IDENTITY.scaled(Vector3::ONE * 0.1),
            //     ],
            // );

            // self.draw_debug_boxes();
            return;
        }
        self.set_main_rotor_rotation(1.2);
    }
}

#[godot_api]
impl Helicopter {
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

    #[func]
    fn draw_debug_boxes(&mut self) {}
}
