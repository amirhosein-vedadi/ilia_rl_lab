# ILIA RL Lab

| Isaac Lab | MuJoCo sim-to-sim | Real robot |
| :---: | :---: | :---: |
| [![Isaac Lab playback](media/isaac-lab-preview.gif)](media/isaac-lab.mp4) | [![MuJoCo playback](media/mujoco-preview.gif)](media/mujoco.mp4) | [![Real ILIA robot](media/real-robot-preview.gif)](media/real-robot.mp4) |
| [Watch Isaac Lab](media/isaac-lab.mp4) | [Watch MuJoCo](media/mujoco.mp4) | [Watch real robot](media/real-robot.mp4) |

Select a preview to open the full, silent video.

Isaac Lab training for the **ILIA Reduced** 19-joint humanoid velocity task, with
Flat and Rough variants and a separate MuJoCo sim-to-sim evaluator. This
repository contains the robot URDF, task and PPO configurations, validation
scripts, and an ONNX playback path. It does not require the private ILIA SDK
for MuJoCo playback.

| Task | Environment ID | Actor observations | Terrain | RSL-RL experiment |
| --- | --- | ---: | --- | --- |
| Flat | `Isaac-ILIA-Reduced-Velocity-Flat-v0` | 68 | plane | `ilia_reduced_velocity_flat` |
| Rough | `Isaac-ILIA-Reduced-Velocity-Rough-v0` | 255 | generated terrain with a 17 × 11 height scan | `ilia_reduced_velocity_rough` |

Each has a matching `-Play-v0` environment for checkpoint evaluation. The
policy produces 19 joint-position actions at 50 Hz; physics runs at 200 Hz.
The control and observation contract is described in
[docs/ilia_reduced_velocity.md](docs/ilia_reduced_velocity.md).

## Repository map

- [`source/ilia_rl_lab/ilia_rl_lab/assets/robots/ilia_reduced/`](source/ilia_rl_lab/ilia_rl_lab/assets/robots/ilia_reduced/):
  URDF and private mesh placement, canonical joint order, motor parameters,
  and Isaac Lab articulation configuration.
- [`source/ilia_rl_lab/ilia_rl_lab/tasks/velocity/`](source/ilia_rl_lab/ilia_rl_lab/tasks/velocity/):
  shared commands, observations, rewards, terrain and events.
- [`source/ilia_rl_lab/ilia_rl_lab/tasks/velocity/config/ilia_reduced/`](source/ilia_rl_lab/ilia_rl_lab/tasks/velocity/config/ilia_reduced/):
  Flat/Rough task registration, environment specializations, and PPO settings.
- [`scripts/`](scripts/): Isaac Lab asset checks, environment checks, and
  RSL-RL training/playback.
- [`sim2sim/`](sim2sim/): standalone MuJoCo ONNX playback, dependencies, and
  instructions for evaluating your own exports. It does not launch Isaac Sim.

## Private mesh setup

The 22 reduced ILIA STL meshes referenced by the URDF are **not included** in
this public repository or its Git history. Obtain them through an authorized
private source and place them, with their original filenames, in:

```text
source/ilia_rl_lab/ilia_rl_lab/assets/robots/ilia_reduced/urdf/meshes_reduced/
```

This directory is ignored by Git. Isaac Lab import and MuJoCo playback require
these meshes; the URDF and numeric parameters alone are insufficient.

## Isaac Lab setup and validation

Use a Python environment with a working Isaac Lab and Isaac Sim installation.
From the repository root, install this extension into that environment:

```bash
python -m pip install -e source/ilia_rl_lab
python scripts/list_envs.py
python scripts/check_ilia_reduced.py --headless
python scripts/check_velocity.py --task Isaac-ILIA-Reduced-Velocity-Flat-v0 --headless
python scripts/check_velocity.py --task Isaac-ILIA-Reduced-Velocity-Rough-v0 --headless
```

The checks verify the imported robot, ordered joints, observation dimensions,
actuator delay and finite physics steps. They do not measure learned walking
quality.

## Train and play

Training creates runs under ignored `logs/rsl_rl/` folders:

```bash
python scripts/rsl_rl/train.py --task Isaac-ILIA-Reduced-Velocity-Flat-v0 --headless
python scripts/rsl_rl/train.py --task Isaac-ILIA-Reduced-Velocity-Rough-v0 --headless
```

To evaluate a particular checkpoint in Isaac Lab, give its path explicitly.
For example, replace `/path/to/model_XXXX.pt` with one from the corresponding
run:

```bash
python scripts/rsl_rl/play.py --task Isaac-ILIA-Reduced-Velocity-Flat-Play-v0 --checkpoint /path/to/model_XXXX.pt
python scripts/rsl_rl/play.py --task Isaac-ILIA-Reduced-Velocity-Rough-Play-v0 --checkpoint /path/to/model_XXXX.pt
```

The Play script exports `policy.onnx` into that checkpoint's run directory.
Pass the export to [sim2sim/README.md](sim2sim/README.md) for MuJoCo playback.
Keep run logs and policy exports out of Git; the tracked evaluator accepts a
policy path at runtime.

## MuJoCo sim-to-sim

The MuJoCo runner uses the repository's URDF and numeric parameters, the
private meshes placed above, plus MuJoCo, NumPy and ONNX Runtime. Isaac Lab
and MuJoCo have different contact and actuator implementations, so compare fixed
commands in both simulators before drawing conclusions about a policy. See
[sim2sim/README.md](sim2sim/README.md) for setup and commands. Policy exports
are supplied by the user and are not distributed in this repository.
