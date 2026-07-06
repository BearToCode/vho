use godot::prelude::*;

use crate::ring::Ring;

#[derive(GodotClass)]
#[class(base=Node3D)]
pub struct Track {
    base: Base<Node3D>,
    current_ring_index: usize,

    #[export]
    pub rings: Array<Option<Gd<Ring>>>,
}

#[godot_api]
impl INode3D for Track {
    fn init(base: Base<Node3D>) -> Self {
        Self {
            base,
            current_ring_index: 0,
            rings: Array::new(),
        }
    }

    fn ready(&mut self) {
        let this = self.to_gd();

        for (ring_idx, ring) in self.rings.iter_shared().enumerate() {
            let ring = ring.expect("Track is missing a ring");
            let ring_bind = ring.bind();
            let area = ring_bind.area.as_ref().expect("Ring is missing a area 3D");

            let mut this = this.clone(); // clone the handle for this closure

            area.signals().body_entered().connect(move |_area| {
                let mut this_bind = this.bind_mut();

                if ring_idx == this_bind.current_ring_index {
                    this_bind.current_ring_index += 1;
                    drop(this_bind);

                    this.signals().ring_passed().emit(ring_idx as i64);
                }
            });
        }
    }

    fn physics_process(&mut self, _delta: f32) {}
}

#[godot_api]
impl Track {
    #[func]
    pub fn current_ring(&self) -> Option<Gd<Ring>> {
        return self.rings.get(self.current_ring_index).unwrap_or(None);
    }

    #[func]
    pub fn next_ring(&self) -> Option<Gd<Ring>> {
        return self.rings.get(self.current_ring_index + 1).unwrap_or(None);
    }

    #[signal]
    pub fn ring_passed(ring_idx: i64);
}
