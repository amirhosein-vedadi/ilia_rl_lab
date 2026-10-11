# ILIA Reduced: Isaac Lab to MuJoCo

This folder replays the **Flat** and **Rough** ILIA Reduced velocity policies
trained in this repository. `ilia_reduced_sim2sim.py` loads an exported ONNX
actor, builds the 19-joint robot in native MuJoCo, reconstructs the actor
observation, and applies joint targets. It uses `mujoco`, `numpy` and
`onnxruntime`, without Isaac Sim, the ILIA SDK, DDS, or a robot connection.
The SDK and any private deployment code are not needed here.

## Files and inputs

| Item | Purpose |
| --- | --- |
| `ilia_reduced_sim2sim.py` | MuJoCo model, observation bridge, ONNX inference, control loop, and metrics |
| `requirements.txt` | Pinned standalone Python dependencies |
| `../source/ilia_rl_lab/ilia_rl_lab/assets/robots/ilia_reduced/` | Shared URDF, private mesh placement, joint order, default pose, gains, limits, and action scales |
| `../logs/rsl_rl/ilia_reduced_velocity_*/.../exported/policy.onnx` | Your own policy export, ignored by Git |

The runner locates the asset from its position in `sim2sim/`. It checks the
URDF SHA-256, 19 motors, floating base, and approximately 30.181 kg total mass
before policy inference. If the asset changes deliberately, review the model
and then update the expected hash in the runner.

The URDF requires 22 STL meshes. They are not part of this public repository;
place the privately supplied files in the directory described in the
[project README](../README.md#private-mesh-setup) before running this script.

## Python environment

Run these commands from the repository root. Use a Python environment that
already has the three packages, or create a small environment outside Git:

```bash
cd /path/to/ilia_rl_lab
python3 -m venv /tmp/ilia-sim2sim-venv
/tmp/ilia-sim2sim-venv/bin/python -m pip install -r sim2sim/requirements.txt
export ILIA_MJ_PY=/tmp/ilia-sim2sim-venv/bin/python
```

`requirements.txt` pins MuJoCo 3.5.0, ONNX Runtime 1.28.0 and NumPy 1.26.4.
An existing Isaac Lab Python environment with these packages also works
without launching Isaac Sim; set `ILIA_MJ_PY` to its Python executable.

## Select policy exports

After evaluating a training checkpoint with `scripts/rsl_rl/play.py`, use the
`exported/policy.onnx` file in that run. Set these variables to your own
absolute export paths:

```bash
export ILIA_FLAT_POLICY=/path/to/flat/exported/policy.onnx
export ILIA_ROUGH_POLICY=/path/to/rough/exported/policy.onnx
```

The repository does not include trained checkpoints or policy exports. The
runner prints each policy's SHA-256 so different exports can be identified
when comparing runs.

Validate the robot and policy interface before a rollout:

```bash
"$ILIA_MJ_PY" sim2sim/ilia_reduced_sim2sim.py --policy "$ILIA_FLAT_POLICY" --terrain flat --validate-only --headless
"$ILIA_MJ_PY" sim2sim/ilia_reduced_sim2sim.py --policy "$ILIA_ROUGH_POLICY" --terrain rough --validate-only --headless
```

Flat must accept `[1, 68]` observations and emit `[1, 19]` actions; Rough
must accept `[1, 255]` and emit `[1, 19]`. The ONNX exports already include
their trained observation normalizers.

## Playback

These commands open MuJoCo's passive viewer for 10 seconds. Flat requests
0.3 m/s forward, while Rough requests 0.5 m/s forward:

```bash
"$ILIA_MJ_PY" sim2sim/ilia_reduced_sim2sim.py --policy "$ILIA_FLAT_POLICY" --terrain flat --command 0.3 0 0 --duration 10
"$ILIA_MJ_PY" sim2sim/ilia_reduced_sim2sim.py --policy "$ILIA_ROUGH_POLICY" --terrain rough --command 0.5 0 0 --duration 10
```

Add `--headless` for a faster repeatable run. Command components are
`--command VX VY WZ` in the robot frame: forward, leftward, and yaw rate.
When omitted, the runner uses `0.3 0 0` for Flat and `0.5 0 0` for Rough.
Try standing (`0 0 0`), backward (`-0.2 0 0`), sideways (`0 0.1 0`), and yaw
(`0 0 0.2`) separately. Use the same seed, duration, command, and model
parameters when comparing two policies.

To test whether the Rough policy's behavior comes from terrain rather than
the 255-value observation interface, keep its observation layout while using
a plane:

```bash
"$ILIA_MJ_PY" sim2sim/ilia_reduced_sim2sim.py --policy "$ILIA_ROUGH_POLICY" --terrain flat --observation-mode rough --command 0.5 0 0 --duration 10 --headless
```

## What is reproduced

- The canonical 19-joint order comes from the shared robot parameters.
  Flat observations contain body angular velocity, projected gravity, the
  velocity command, gait phase, joint position error, joint velocity, and the
  preceding raw action. Rough adds a yaw-aligned 17 × 11 downward height scan.
- MuJoCo steps at 0.005 s. Policy inference occurs every four steps (50 Hz).
  The position target is `default_q + action_scale * action - encoder_bias`.
  Each joint independently samples a new 1–4 physics-step target delay.
  `--delay none` or `--delay 1` can isolate delay effects.
- Joint torque is clipped after `Kp * (target - q) - Kd * qdot`. Armature,
  damping, friction, gains, effort caps and scales are read from the public
  `parameters.py` file. The default encoder bias is zero; use
  `--encoder-bias-max 0.015` to probe the trained bound.
- Flat uses a MuJoCo plane. Rough uses a deterministic 0–9 cm heightfield
  with a flat central spawn patch. This is a transfer probe, **not** the
  generated Isaac Lab terrain mesh. The runner stops after a tilt above 70°
  or a non-foot ground contact force above 10 N.

The script runs one robot and one explicit command, rather than reproducing
Isaac Lab's parallel training environments, command curriculum, pushes,
sensor-noise randomization, or full terrain distribution. Contact handling
and implicit PhysX drives also differ from MuJoCo. A successful rollout
therefore supports a specific sim-to-sim comparison, not hardware readiness.

## Reading the output

`PASS model` reports the compiled MuJoCo robot shape and mass. `PASS policy`
reports the ONNX input and output sizes. After playback, `RESULT` reports
simulated time, forward and lateral displacement, mean velocity-tracking
errors, and the stop reason. Use the same policy file, command, seed, terrain,
duration, delay, and friction settings for comparisons. A completed rollout
alone does not establish reliable transfer to another simulator or hardware.
