# vho

Deep Reinforcement Learning for Helicopter Attitude Stabilization — control of an unstable, simplified helicopter model through Action Dependent Heuristic Dynamic Programming (ADHDP).

This project was developed for the course _Bio-inspired Intelligence and Learning for Aerospace Applications_ at TU Delft.

## Overview

A simplified, nonlinear helicopter model — including main rotor flapping dynamics — is simulated in a 3D physics environment. An Actor-Critic reinforcement learning agent is trained online, without prior knowledge of the system dynamics, to stabilize the helicopter from randomized initial attitudes and velocities.

Key components:

- **Simulation environment** — built in Godot, using the Jolt physics engine to integrate the rigid-body helicopter dynamics and provide a visual training environment.
- **Learning infrastructure** — implemented in Rust using the [Burn](https://github.com/tracel-ai/burn) deep learning framework, compiled to a dynamic library and called from Godot at runtime.
- **Analysis / postprocessing** — Python scripts for parsing training logs and generating the plots used to evaluate training runs and the final controller.

## Method

The agent is trained using **ADHDP**:

- A **Critic** network estimates the value function $J(x, u)$ given the current state and action.
- An **Actor** network outputs an action aimed at maximizing the value predicted by the Critic.
- Both networks are trained online, using target networks (Polyak averaging) and a replay buffer to stabilize learning.

The helicopter model includes 12 rigid-body states plus two rotor flapping states, and is controlled through four inputs: main rotor collective, longitudinal cyclic, lateral cyclic, and tail rotor collective. See the accompanying report for the full dynamics, reward function, and training methodology.

## Repository structure

```
vho/
├── godot/ # Godot project: simulation scenes, helicopter model, training/eval environments
├── rust/ # Rust crate: Actor/Critic networks, ADHDP training loop, Burn integration
├── postprocessing/ # Python scripts for parsing episode logs and plotting training results
└── .vscode/ # Editor configuration
```

## Requirements

- [Godot Engine](https://godotengine.org/) (4.x)
- [Rust toolchain](https://rustup.rs/) (stable)
- Python 3.x with `pandas` and `matplotlib`, for postprocessing

## Getting started

1. Clone the repository:

```bash
   git clone https://github.com/BearToCode/vho.git
   cd vho
```

2. Build the Rust learning library:

```bash
   cd rust
   cargo build --release
```

3. Open the `godot/` project in the Godot editor and run the `Train` scene to start training, or `Eval` to evaluate a trained model.

## Results

Training experiments across network size, learning rate, discount factor (γ), target networks, and replay buffer configuration are documented in the accompanying report. The final trained model (16-neuron hidden layers, learning rate 0.001, γ = 0.975) successfully stabilized the helicopter in 999 out of 1000 validation runs with randomized initial conditions.

## Report

The full methodology, dynamics derivation, and results analysis are available in the project report: _Deep Reinforcement Learning for Helicopter Attitude Stabilization_ (Davide Basso, TU Delft, 2025/2026).

## Author

**Davide Basso** — TU Delft, Aerospace Engineering
