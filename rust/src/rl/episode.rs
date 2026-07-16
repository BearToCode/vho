use godot::prelude::*;
use std::fmt::Display;

#[derive(Clone, Copy)]
pub struct Episode {
    pub index: usize,
    /// Elapsed time.
    pub time: f32,
    /// Number of steps elapsed.
    pub steps: usize,
    /// Noise factor for this episode, in [0, 1].
    pub noise: f32,
    /// Accumulated reward this episode.
    pub accumulated_reward: f32,
    /// Sum of critic losses this episode (for averaging).
    pub critic_loss_sum: f32,
    /// Sum of actor losses this episode (for averaging).
    pub actor_loss_sum: f32,
    /// Number of train_step calls this episode (for averaging).
    pub train_steps: usize,
    /// Number of those steps that actually updated the actor. TD3's policy delay means
    /// this is a fraction of `train_steps`, so the actor loss needs its own divisor.
    pub actor_train_steps: usize,
}

impl Episode {
    pub fn new(index: usize) -> Self {
        Episode {
            index,
            time: 0.0,
            steps: 0,
            noise: 0.0,
            accumulated_reward: 0.0,
            critic_loss_sum: 0.0,
            actor_loss_sum: 0.0,
            train_steps: 0,
            actor_train_steps: 0,
        }
    }

    /// Append this episode's stats as one row in the training log CSV.
    /// Writes a header row if the file doesn't exist yet.
    pub fn log(&self, dir: &str) {
        use std::io::Write;

        let log_path = format!("{}episodes.csv", dir);

        let file_exists = std::path::Path::new(&log_path).exists();

        let mut file = match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
        {
            Ok(f) => f,
            Err(e) => {
                godot_error!("Failed to open training log: {e}");
                return;
            }
        };

        if !file_exists {
            let header = "index,episode_time,steps,noise,episode_reward,avg_critic_loss,avg_actor_loss\n";
            if let Err(e) = file.write_all(header.as_bytes()) {
                godot_error!("Failed to write CSV header: {e}");
                return;
            }
        }

        let avg_critic_loss = if self.train_steps > 0 {
            self.critic_loss_sum / self.train_steps as f32
        } else {
            0.0
        };
        let avg_actor_loss = if self.actor_train_steps > 0 {
            self.actor_loss_sum / self.actor_train_steps as f32
        } else {
            0.0
        };

        let row = format!(
            "{},{},{},{},{},{},{}\n",
            self.index,
            self.time,
            self.steps,
            self.noise,
            self.accumulated_reward,
            avg_critic_loss,
            avg_actor_loss,
        );

        if let Err(e) = file.write_all(row.as_bytes()) {
            godot_error!("Failed to write CSV row: {e}");
        }
    }
}

impl Display for Episode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let avg_critic_loss = if self.train_steps > 0 {
            self.critic_loss_sum / self.train_steps as f32
        } else {
            0.0
        };
        let avg_actor_loss = if self.actor_train_steps > 0 {
            self.actor_loss_sum / self.actor_train_steps as f32
        } else {
            0.0
        };

        write!(
            f,
            "idx: {} \t| t: {:.2} \t| rwd: {:.2} \t| critic_loss: {:.2} \t| \
             actor_loss: {:.2} \t| noise: {:.2}",
            self.index,
            self.time,
            self.accumulated_reward,
            avg_critic_loss,
            avg_actor_loss,
            self.noise,
        )
    }
}
