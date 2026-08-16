use std::fs::File;
use std::io::{BufWriter, Write};

use burn::{
    module::Module,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
};
use godot::prelude::*;

use crate::{
    game::Game,
    rl::{
        Backend, DEVICE,
        action::{ACTION_DIM, perform_action},
        networks::ActorModel,
        state::{STATE_DIM, StateNormalizationConfig, get_agent_state, normalize_state},
    },
};

/// Runs a trained actor against a helicopter and logs every step's
/// (non-normalized) state, normalized state, and control inputs to a CSV
/// file. Does not evaluate success/failure or reset episodes — purely a
/// data-collection node.
#[derive(GodotClass)]
#[class(base=Node3D)]
pub struct Recorder {
    base: Base<Node3D>,
    actor: Option<ActorModel<Backend>>,
    writer: Option<BufWriter<File>>,
    step: u64,
    time: f32,
    finished: bool,

    #[export]
    /// Reference to the game manager.
    game: Option<Gd<Game>>,
    #[export(file)]
    #[var(hint = FILE)]
    actor_model_path: GString,
    #[export]
    actor_hidden_layers: Array<i64>,
    #[export(file)]
    #[var(hint = FILE, hint_string = "*.csv")]
    /// Where to write the recorded CSV. Overwritten on `ready`.
    output_csv_path: GString,
    #[export]
    /// Stop recording (and freeze the node) after this many seconds.
    /// 0 = record indefinitely, until the node is removed from the tree.
    max_time: f32,

    #[export_group(name = "State Normalization")]
    #[export]
    linear_velocity_scale: f32,
    #[export]
    angular_velocity_scale: f32,
    #[export]
    position_error_scale: f32,
    #[export]
    angle_scale: f32,
    #[export]
    flap_angle_scale: f32,
}

#[godot_api]
impl INode3D for Recorder {
    fn init(base: Base<Node3D>) -> Self {
        Self {
            base,
            actor: None,
            writer: None,
            step: 0,
            time: 0.0,
            finished: false,

            game: None,
            actor_model_path: GString::new(),
            actor_hidden_layers: Array::new(),
            output_csv_path: GString::new(),
            max_time: 0.0,

            linear_velocity_scale: 0.0,
            angular_velocity_scale: 0.0,
            position_error_scale: 0.0,
            angle_scale: 0.0,
            flap_angle_scale: 0.0,
        }
    }

    fn ready(&mut self) {
        if self.game.is_none() {
            panic!("No game set!");
        }
        if self.actor_model_path.is_empty() {
            panic!("No actor model set!");
        }
        if self.output_csv_path.is_empty() {
            panic!("No output CSV path set!");
        }

        self.actor = Some(ActorModel::<Backend>::new(
            STATE_DIM,
            ACTION_DIM,
            &self
                .actor_hidden_layers
                .iter_shared()
                .map(|v| v as usize)
                .collect::<Vec<usize>>(),
            &DEVICE,
        ));

        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        self.actor = Some(
            self.actor
                .clone()
                .unwrap()
                .load_file(
                    self.actor_model_path.to_string().as_str(),
                    &recorder,
                    &DEVICE,
                )
                .unwrap_or_else(|e| {
                    panic!(
                        "Failed to load actor model from {}: {}",
                        self.actor_model_path, e
                    )
                }),
        );
        godot_print!("Loaded actor model from {}", self.actor_model_path);

        let file = File::create(self.output_csv_path.to_string())
            .unwrap_or_else(|e| panic!("Failed to create CSV at {}: {}", self.output_csv_path, e));
        let mut writer = BufWriter::new(file);

        let mut header = vec![
            "step".to_string(),
            "time_s".to_string(),
            // Non-normalized state, matches AgentStateComponent order.
            "lin_vel_x".to_string(),
            "lin_vel_y".to_string(),
            "lin_vel_z".to_string(),
            "ang_vel_x".to_string(),
            "ang_vel_y".to_string(),
            "ang_vel_z".to_string(),
            "roll".to_string(),
            "pitch".to_string(),
            "lon_flap".to_string(),
            "lat_flap".to_string(),
            "pos_err_y".to_string(),
        ];
        for i in 0..STATE_DIM {
            header.push(format!("norm_state_{i}"));
        }
        header.push("collective".to_string());
        header.push("lateral_cyclic".to_string());
        header.push("longitudinal_cyclic".to_string());
        header.push("tail_rotor_cyclic".to_string());

        writeln!(writer, "{}", header.join(","))
            .unwrap_or_else(|e| panic!("Failed to write CSV header: {e}"));

        self.writer = Some(writer);
        godot_print!("Recording to {}", self.output_csv_path);
    }

    fn physics_process(&mut self, delta: f32) {
        if self.finished {
            return;
        }

        self.time += delta;

        let game = self.game.clone().unwrap();
        let helicopter = game.bind().helicopter.clone().unwrap();
        let state = get_agent_state(game);

        let normalization_config = StateNormalizationConfig {
            angular_velocity_scale: self.angular_velocity_scale,
            linear_velocity_scale: self.linear_velocity_scale,
            angle_scale: self.angle_scale,
            flap_angle_scale: self.flap_angle_scale,
            position_error_scale: self.position_error_scale,
        };

        let x = normalize_state(&state, &normalization_config, &DEVICE);
        let actor = self.actor.as_ref().unwrap();
        let u = actor.forward(x.clone());
        let u = u.clamp(-1.0, 1.0);

        self.write_row(&state, &x, &u);

        perform_action(u, helicopter);

        self.step += 1;
        if self.max_time > 0.0 && self.time >= self.max_time {
            self.finish();
        }
    }

    fn exit_tree(&mut self) {
        // Make sure buffered rows hit disk even if max_time was never reached
        // (e.g. scene stopped manually).
        self.finish();
    }
}

#[godot_api]
impl Recorder {
    fn write_row(
        &mut self,
        raw_state: &crate::rl::state::AgentStateVector,
        norm_state: &burn::tensor::Tensor<Backend, 2>,
        action: &burn::tensor::Tensor<Backend, 2>,
    ) {
        // NOTE: verify `to_vec::<f32>()` matches your burn version's TensorData API;
        // some versions use `.into_data().convert::<f32>().value` instead.
        let norm_values: Vec<f32> = norm_state
            .clone()
            .to_data()
            .to_vec::<f32>()
            .expect("Failed to extract normalized state data");
        // Order must match `perform_action`'s use of control_normalized:
        // [collective, lateral_cyclic, longitudinal_cyclic, tail_rotor_cyclic]
        let action_values: Vec<f32> = action
            .clone()
            .to_data()
            .to_vec::<f32>()
            .expect("Failed to extract action data");

        let mut fields: Vec<String> = vec![self.step.to_string(), self.time.to_string()];
        fields.extend(raw_state.as_slice().iter().map(|v| v.to_string()));
        fields.extend(norm_values.iter().map(|v| v.to_string()));
        fields.extend(action_values.iter().map(|v| v.to_string()));

        if let Some(writer) = self.writer.as_mut() {
            if let Err(e) = writeln!(writer, "{}", fields.join(",")) {
                godot_print!("Failed to write CSV row: {e}");
            }
        }
    }

    fn finish(&mut self) {
        if self.finished {
            return;
        }
        self.finished = true;

        if let Some(writer) = self.writer.as_mut() {
            if let Err(e) = writer.flush() {
                godot_print!("Failed to flush CSV: {e}");
            }
        }
        godot_print!(
            "Recording finished after {:.2}s ({} steps).",
            self.time,
            self.step
        );
    }
}
