# SPDX-License-Identifier: BSD-3-Clause
"""Standalone Isaac Lab ILIA Reduced policy playback in native MuJoCo.

Dependencies: mujoco, numpy, onnxruntime. No Isaac Sim, mjlab, or ILIA SDK.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = REPO_ROOT / "source/ilia_rl_lab/ilia_rl_lab/assets/robots/ilia_reduced"
URDF_PATH = ASSET_DIR / "urdf/ilia_reduced.urdf"
PARAMETERS_PATH = ASSET_DIR / "parameters.py"
EXPECTED_URDF_SHA256 = "861dd2ed401e57b303bb05ddf505998af17a482319e67c13b4939b6af99e0d1b"
PHYSICS_DT = 0.005
DECIMATION = 4
CONTROL_DT = PHYSICS_DT * DECIMATION
GAIT_PERIOD = 0.8
ROOT_HEIGHT = 0.72
EXPECTED_MASS = 30.18120911
GROUND_GEOM_NAMES = ("floor", "rough_ground")


def load_parameters():
    """Load the plain parameter file without importing the Isaac Lab package."""
    spec = importlib.util.spec_from_file_location(
        "ilia_reduced_sim2sim_parameters", PARAMETERS_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {PARAMETERS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_rough_heightfield(size: int = 129) -> list[float]:
    """Deterministic low-amplitude waves with a flat initial spawn patch.

    This probes rough-terrain transfer. It is not an exact Isaac terrain mesh.
    """
    xy = np.linspace(-5.0, 5.0, size)
    x, y = np.meshgrid(xy, xy, indexing="xy")
    radius = np.sqrt(x * x + y * y)
    envelope = np.clip((radius - 0.9) / 0.6, 0.0, 1.0)
    wave = (
        0.5
        + 0.28 * np.sin(3.1 * x) * np.cos(2.7 * y)
        + 0.15 * np.sin(7.3 * x + 3.7 * y)
    )
    heights = np.clip(envelope * wave, 0.0, 1.0)
    return heights.astype(np.float64).ravel().tolist()


def build_model(params, terrain: str, foot_friction: float) -> mujoco.MjModel:
    digest = hashlib.sha256(URDF_PATH.read_bytes()).hexdigest()
    if digest != EXPECTED_URDF_SHA256:
        raise ValueError(
            f"URDF SHA-256 is {digest}, expected {EXPECTED_URDF_SHA256}. "
            "Review the changed robot model before updating this contract."
        )

    spec = mujoco.MjSpec.from_file(str(URDF_PATH))
    mesh_paths = sorted((URDF_PATH.parent / "meshes_reduced").glob("*.STL"))
    if len(mesh_paths) != 22:
        raise ValueError(f"Expected 22 robot meshes, found {len(mesh_paths)}")
    spec.assets = {path.name: path.read_bytes() for path in mesh_paths}

    base = spec.body("base")
    if base is None:
        raise ValueError("URDF has no base body")
    base.pos = (0.0, 0.0, ROOT_HEIGHT)
    base.add_freejoint(name="floating_base_joint")
    base.add_site(name="imu", pos=(0.0, 0.0, 0.0), size=(0.01, 0.0, 0.0))
    for name, sensor_type in (
        ("imu_gyro", mujoco.mjtSensor.mjSENS_GYRO),
        ("imu_lin_vel", mujoco.mjtSensor.mjSENS_VELOCIMETER),
    ):
        spec.add_sensor(
            name=name,
            type=sensor_type,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            objname="imu",
        )

    # Disable robot self collision; retain robot-ground contact on every link.
    for body in spec.bodies[1:]:
        for geom in body.geoms:
            geom.group = 1  # Ray scans select only group 0 ground geometry.
            geom.contype = 0
            geom.conaffinity = 1
            if body.name in ("left_foot_link", "right_foot_link"):
                geom.condim = 3
                geom.priority = 1
                geom.friction[:] = (foot_friction, 0.005, 0.0001)
            else:
                geom.condim = 1

    if terrain == "flat":
        spec.worldbody.add_geom(
            name="floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=(0.0, 0.0, 0.05),
            contype=1,
            conaffinity=1,
            condim=3,
            friction=(1.0, 0.005, 0.0001),
            rgba=(0.25, 0.28, 0.32, 1.0),
        )
    else:
        spec.add_hfield(
            name="rough_heightfield",
            nrow=129,
            ncol=129,
            size=(5.0, 5.0, 0.09, 0.1),
            userdata=make_rough_heightfield(),
        )
        spec.worldbody.add_geom(
            name="rough_ground",
            type=mujoco.mjtGeom.mjGEOM_HFIELD,
            hfieldname="rough_heightfield",
            contype=1,
            conaffinity=1,
            condim=3,
            friction=(1.0, 0.005, 0.0001),
            rgba=(0.42, 0.48, 0.39, 1.0),
        )

    spec.worldbody.add_light(
        name="light", pos=(0.0, 0.0, 3.0), dir=(0.0, 0.0, -1.0)
    )

    for joint_name in params.ILIA_REDUCED_DOF_NAMES:
        joint = spec.joint(joint_name)
        if joint is None:
            raise ValueError(f"URDF has no joint {joint_name}")
        p = params.ILIA_REDUCED_ACTUATOR_PARAMETERS[joint_name]
        joint.armature = p.armature
        joint.frictionloss = p.dry_friction
        joint.damping = p.viscous_friction

        limit = params.ILIA_REDUCED_EFFORT_LIMITS[joint_name]
        actuator = spec.add_actuator(name=f"motor_{joint_name}", target=joint_name)
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
        actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        actuator.biastype = mujoco.mjtBias.mjBIAS_NONE
        actuator.gear[0] = 1.0
        actuator.forcelimited = True
        actuator.forcerange[:] = (-limit, limit)
        actuator.ctrllimited = True
        actuator.ctrlrange[:] = (-limit, limit)

    spec.option.timestep = PHYSICS_DT
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    spec.option.iterations = 10
    spec.option.ls_iterations = 20
    spec.option.ccd_iterations = 50
    model = spec.compile()

    if (model.nq, model.nv, model.nu) != (26, 25, 19):
        raise ValueError(f"Unexpected nq/nv/nu: {(model.nq, model.nv, model.nu)}")
    if not math.isclose(mujoco.mj_getTotalmass(model), EXPECTED_MASS, abs_tol=1e-4):
        raise ValueError(f"Unexpected robot mass: {mujoco.mj_getTotalmass(model)}")
    return model


class PolicyBridge:
    def __init__(self, model, params, policy_path: Path, observation_mode: str, seed: int,
                 encoder_bias_max: float, delay: str):
        self.model = model
        self.params = params
        self.observation_mode = observation_mode
        self.rng = np.random.default_rng(seed)
        self.delay = delay
        self.names = tuple(params.ILIA_REDUCED_DOF_NAMES)
        self.qpos_ids = []
        self.qvel_ids = []
        for name in self.names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0 or model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(f"Expected hinge joint {name}")
            self.qpos_ids.append(model.jnt_qposadr[jid])
            self.qvel_ids.append(model.jnt_dofadr[jid])
        self.qpos_ids = np.asarray(self.qpos_ids, dtype=np.int64)
        self.qvel_ids = np.asarray(self.qvel_ids, dtype=np.int64)
        self.actuator_ids = np.asarray([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor_{name}")
            for name in self.names
        ], dtype=np.int64)
        if np.any(self.actuator_ids < 0):
            raise ValueError("At least one named motor is missing")

        self.default_q = np.asarray([
            params.ILIA_REDUCED_DEFAULT_JOINT_POS[name] for name in self.names
        ], dtype=np.float64)
        self.scale = np.asarray([
            params.ILIA_REDUCED_ACTION_SCALE[name] for name in self.names
        ], dtype=np.float64)
        self.kp = np.asarray([
            params.ILIA_REDUCED_ACTUATOR_PARAMETERS[name].kp for name in self.names
        ], dtype=np.float64)
        self.kd = np.asarray([
            params.ILIA_REDUCED_ACTUATOR_PARAMETERS[name].kd for name in self.names
        ], dtype=np.float64)
        self.effort = np.asarray([
            params.ILIA_REDUCED_EFFORT_LIMITS[name] for name in self.names
        ], dtype=np.float64)
        self.bias = self.rng.uniform(-encoder_bias_max, encoder_bias_max, 19)
        self.previous_action = np.zeros(19, dtype=np.float64)
        self.history = np.zeros((5, 19), dtype=np.float64)
        self.history_count = 0
        self.cursor = 0
        self.base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.ground_ids = {
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in GROUND_GEOM_NAMES
        }
        self.ground_ids.discard(-1)
        self.foot_body_ids = {
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ("left_foot_link", "right_foot_link")
        }
        if self.base_id < 0 or -1 in self.foot_body_ids:
            raise ValueError("Missing base or foot body")

        policy_path = policy_path.expanduser().resolve()
        if not policy_path.is_file():
            raise FileNotFoundError(policy_path)
        self.policy_sha256 = hashlib.sha256(policy_path.read_bytes()).hexdigest()
        self.session = ort.InferenceSession(
            str(policy_path), providers=["CPUExecutionProvider"]
        )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        expected_obs = 68 if observation_mode == "flat" else 255
        if len(inputs) != 1 or inputs[0].shape != [1, expected_obs]:
            raise ValueError(
                f"Policy expects {[(i.name, i.shape) for i in inputs]}; "
                f"{observation_mode} needs [1, {expected_obs}]"
            )
        if len(outputs) != 1 or outputs[0].shape != [1, 19]:
            raise ValueError(f"Expected [1, 19] policy output, got {outputs}")
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        self.expected_obs = expected_obs

        # Isaac's grid uses x as the inner/fast axis and y as the outer axis.
        xs = np.linspace(-0.8, 0.8, 17)
        ys = np.linspace(-0.5, 0.5, 11)
        self.scan_xy = np.asarray([(x, y) for y in ys for x in xs])
        self.ray_direction = np.array([0.0, 0.0, -1.0])
        self.ray_groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        self.ray_geom_id = np.zeros(1, dtype=np.int32)

    def reset(self, data):
        mujoco.mj_resetData(self.model, data)
        data.qpos[:] = self.model.qpos0
        data.qpos[self.qpos_ids] = self.default_q
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        self.previous_action[:] = 0.0
        self.history[:] = 0.0
        self.history_count = 0
        self.cursor = 0
        mujoco.mj_forward(self.model, data)

    def gravity_body(self, data):
        rotation = data.xmat[self.base_id].reshape(3, 3)
        return rotation.T @ np.array([0.0, 0.0, -1.0])

    def scan(self, data):
        rotation = data.xmat[self.base_id].reshape(3, 3)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        c, s = math.cos(yaw), math.sin(yaw)
        base = data.xpos[self.base_id]
        values = np.empty(187, dtype=np.float64)
        for i, (x, y) in enumerate(self.scan_xy):
            origin = np.array([
                base[0] + c * x - s * y,
                base[1] + s * x + c * y,
                base[2],
            ])
            distance = mujoco.mj_ray(
                self.model, data, origin, self.ray_direction,
                self.ray_groups, 1, -1, self.ray_geom_id,
            )
            values[i] = distance if 0.0 <= distance <= 5.0 else 5.0
        return values * 0.2

    def observation(self, data, command, step):
        phase = (step * CONTROL_DT / GAIT_PERIOD) % 1.0
        clock = np.array([
            math.sin(2.0 * math.pi * phase),
            math.cos(2.0 * math.pi * phase),
        ])
        if np.linalg.norm(command) < 0.1:
            clock[:] = 0.0
        pieces = [
            np.asarray(data.sensor("imu_gyro").data),
            self.gravity_body(data),
            command,
            clock,
            data.qpos[self.qpos_ids] - self.default_q,
            data.qvel[self.qvel_ids],
            self.previous_action,
        ]
        if self.observation_mode == "rough":
            pieces.append(self.scan(data))
        obs = np.concatenate(pieces).astype(np.float32)
        if obs.shape != (self.expected_obs,) or not np.isfinite(obs).all():
            raise ValueError(f"Invalid policy observation: {obs.shape}")
        return obs

    def predict(self, observation):
        result = self.session.run(
            [self.output_name], {self.input_name: observation[None, :]}
        )[0]
        if result.shape != (1, 19) or not np.isfinite(result).all():
            raise ValueError(f"Invalid policy action: {result.shape}")
        return result[0].astype(np.float64)

    def delayed_target(self, target):
        if self.delay == "none":
            return target
        self.history[self.cursor] = target
        self.history_count = min(self.history_count + 1, len(self.history))
        if self.delay == "random":
            lags = self.rng.integers(1, 5, size=19)
        else:
            lags = np.full(19, int(self.delay), dtype=np.int64)
        lags = np.minimum(lags, self.history_count - 1)
        selected = self.history[(self.cursor - lags) % len(self.history), np.arange(19)]
        self.cursor = (self.cursor + 1) % len(self.history)
        return selected

    def apply_control(self, data, raw_action):
        target = self.default_q + self.scale * raw_action - self.bias
        delayed = self.delayed_target(target)
        torque = self.kp * (delayed - data.qpos[self.qpos_ids])
        torque -= self.kd * data.qvel[self.qvel_ids]
        data.ctrl[self.actuator_ids] = np.clip(torque, -self.effort, self.effort)

    def illegal_contact(self, data):
        for i in range(data.ncon):
            contact = data.contact[i]
            if contact.geom1 not in self.ground_ids and contact.geom2 not in self.ground_ids:
                continue
            other = contact.geom2 if contact.geom1 in self.ground_ids else contact.geom1
            if self.model.geom_bodyid[other] in self.foot_body_ids:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, data, i, force)
            if np.linalg.norm(force[:3]) > 10.0:
                return True
        return False


def run(args):
    if args.duration <= 0.0 or not 0.0 < args.foot_friction <= 2.0:
        raise ValueError("duration and foot friction must be positive")
    if not 0.0 <= args.encoder_bias_max <= 0.015:
        raise ValueError("encoder-bias-max must be in [0, 0.015]")
    observation_mode = args.observation_mode or args.terrain
    default_command = (0.5, 0.0, 0.0) if observation_mode == "rough" else (0.3, 0.0, 0.0)
    command = np.asarray(args.command if args.command is not None else default_command, dtype=np.float64)
    if not np.isfinite(command).all() or np.max(np.abs(command)) > 1.0:
        raise ValueError("Command components must be finite and within +/-1")

    params = load_parameters()
    model = build_model(params, args.terrain, args.foot_friction)
    bridge = PolicyBridge(
        model, params, args.policy, observation_mode, args.seed,
        args.encoder_bias_max, args.delay,
    )
    data = mujoco.MjData(model)
    bridge.reset(data)
    print(
        f"PASS model: {model.nq} qpos, {model.nv} qvel, {model.nu} motors; "
        f"mass={mujoco.mj_getTotalmass(model):.6f} kg"
    )
    print(
        f"PASS policy: {bridge.expected_obs} observations -> 19 actions; "
        f"terrain={args.terrain}; observation_mode={bridge.observation_mode}; "
        f"dt={PHYSICS_DT}s; control={CONTROL_DT}s"
    )
    print(f"Policy SHA-256: {bridge.policy_sha256}")
    initial_obs = bridge.observation(data, command, 0)
    initial_action = bridge.predict(initial_obs)
    print(f"PASS first inference: max |action|={np.abs(initial_action).max():.4f}")
    if args.validate_only:
        return

    from mujoco import viewer as mujoco_viewer

    viewer_context = (
        contextlib.nullcontext(None)
        if args.headless else mujoco_viewer.launch_passive(model, data)
    )
    vx_error = []
    yaw_error = []
    end_reason = "duration"
    with viewer_context as viewer:
        for step in range(math.ceil(args.duration / CONTROL_DT)):
            if viewer is not None and not viewer.is_running():
                end_reason = "viewer closed"
                break
            start = time.monotonic()
            obs = bridge.observation(data, command, step)
            action = bridge.predict(obs)
            measured_velocity = np.asarray(data.sensor("imu_lin_vel").data)
            measured_angular = np.asarray(data.sensor("imu_gyro").data)
            vx_error.append(float(np.linalg.norm(command[:2] - measured_velocity[:2])))
            yaw_error.append(float(abs(command[2] - measured_angular[2])))
            bridge.previous_action[:] = action

            for _ in range(DECIMATION):
                bridge.apply_control(data, action)
                mujoco.mj_step(model, data)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise FloatingPointError("MuJoCo produced a non-finite state")

            if viewer is not None:
                viewer.sync()
                sleep_time = CONTROL_DT - (time.monotonic() - start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            upright_cosine = -bridge.gravity_body(data)[2]
            if upright_cosine < math.cos(math.radians(70.0)):
                end_reason = "tilt greater than 70 degrees"
                break
            if bridge.illegal_contact(data):
                end_reason = "non-foot ground contact above 10 N"
                break

    if not vx_error:
        print(f"RESULT: no control steps executed; stop={end_reason}")
        return
    print(
        f"RESULT terrain={args.terrain} command={command.tolist()} "
        f"time={data.time:.2f}s x={data.qpos[0]:.3f}m "
        f"y={data.qpos[1]:.3f}m mean_xy_error={np.mean(vx_error):.3f}m/s "
        f"mean_yaw_error={np.mean(yaw_error):.3f}rad/s stop={end_reason}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True, help="Exported Isaac Lab policy.onnx")
    parser.add_argument("--terrain", choices=("flat", "rough"), required=True)
    parser.add_argument("--observation-mode", choices=("flat", "rough"),
                        help="Policy observation layout; defaults to terrain")
    parser.add_argument("--command", nargs=3, type=float, metavar=("VX", "VY", "WZ"),
                        help="Robot-frame velocity; defaults to 0.3 m/s Flat or 0.5 m/s Rough")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--foot-friction", type=float, default=0.8)
    parser.add_argument("--encoder-bias-max", type=float, default=0.0)
    parser.add_argument("--delay", choices=("random", "none", "1", "2", "3", "4"),
                        default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
