use godot::{
    classes::{Area3D, IStaticBody3D, StaticBody3D},
    prelude::*,
};

#[derive(GodotClass)]
#[class(base=StaticBody3D)]
pub struct Ring {
    base: Base<StaticBody3D>,
    #[export]
    pub area: Option<Gd<Area3D>>,
}

#[godot_api]
impl IStaticBody3D for Ring {
    fn init(base: Base<StaticBody3D>) -> Self {
        Self { base, area: None }
    }
}
