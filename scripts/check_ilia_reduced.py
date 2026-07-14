"""Check reduced ILIA import, actuator properties, and a small joint command."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps < 100:
    parser.error("--steps must be at least 100.")
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Isaac/Omniverse modules must be imported after starting the application.
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.utils.types import ArticulationActions

from ilia_rl_lab.assets.robots.ilia_reduced.actuators import DelayedImplicitActuatorCfg
from ilia_rl_lab.assets.robots.ilia_reduced.parameters import (
    ILIA_REDUCED_ACTUATOR_PARAMETERS,
    ILIA_REDUCED_DOF_NAMES,
    ILIA_REDUCED_EFFORT_LIMITS,
)
from ilia_rl_lab.assets.robots.ilia_reduced.robot_cfg import get_ilia_reduced_robot_cfg


def check_delay():
    """A two-step delay must return old commands and respect partial resets."""
    cfg = DelayedImplicitActuatorCfg(
        joint_names_expr=["test"],
        stiffness=1.0,
        damping=0.0,
        effort_limit_sim=100.0,
        velocity_limit_sim=100.0,
        min_delay=2,
        max_delay=2,
    )
    actuator = cfg.class_type(
        cfg=cfg, joint_names=["test"], joint_ids=slice(None),
        num_envs=2, device="cpu",
    )
    zeros = torch.zeros(2, 1)

    def send(value):
        command = ArticulationActions(
            joint_positions=torch.full((2, 1), float(value)),
            joint_velocities=zeros.clone(),
            joint_efforts=zeros.clone(),
        )
        return actuator.compute(command, zeros, zeros).joint_positions

    for value, expected in ((0, 0), (1, 0), (2, 0), (3, 1)):
        torch.testing.assert_close(send(value), torch.full((2, 1), float(expected)))
    actuator.reset([0])
    torch.testing.assert_close(send(10), torch.tensor([[10.0], [2.0]]))
    print("PASS: delay warmup, steady delay, and partial reset.")


def main():
    check_delay()
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=0.005, render_interval=4, device=args.device)
    )
    sim.set_camera_view([2.0, 2.0, 1.5], [0.0, 0.0, 0.7])
    light = sim_utils.DomeLightCfg(intensity=2000.0)
    light.func("/World/Light", light)

    cfg = get_ilia_reduced_robot_cfg().replace(prim_path="/World/Robot")
    cfg.spawn.force_usd_conversion = True
    # Isolate the articulation/drive check from falling and ground contact.
    cfg.spawn.rigid_props.disable_gravity = True
    robot = Articulation(cfg)
    sim.reset()

    ids, names = robot.find_joints(list(ILIA_REDUCED_DOF_NAMES), preserve_order=True)
    assert names == list(ILIA_REDUCED_DOF_NAMES)
    assert robot.num_joints == 19
    assert not robot.is_fixed_base
    required_bodies = {"base", "torso_link", "left_foot_link", "right_foot_link"}
    assert required_bodies.issubset(robot.body_names)

    view = robot.root_physx_view
    mass = float(view.get_masses()[0].sum())
    assert abs(mass - 30.18120911) < 1.0e-3, f"Unexpected imported mass: {mass}"
    print(f"PASS: floating base, 19 joints, total mass {mass:.8f} kg.")
    print("Native joint order:", robot.joint_names)
    print("Canonical joint indices:", ids)

    params = ILIA_REDUCED_ACTUATOR_PARAMETERS

    def check_values(label, values, expected):
        reference = torch.tensor(
            [expected[name] for name in names],
            device=values.device,
            dtype=values.dtype,
        )
        torch.testing.assert_close(values, reference, rtol=1e-5, atol=1e-6)
        print(f"PASS: {label}")

    for label, values, field in (
        ("Kp", view.get_dof_stiffnesses()[0, ids], "kp"),
        ("Kd", view.get_dof_dampings()[0, ids], "kd"),
        ("armature", view.get_dof_armatures()[0, ids], "armature"),
    ):
        check_values(label, values, {name: getattr(p, field) for name, p in params.items()})
    check_values("effort caps", view.get_dof_max_forces()[0, ids], ILIA_REDUCED_EFFORT_LIMITS)

    friction = view.get_dof_friction_properties()[0, ids]
    for column, field in enumerate(("dry_friction", "dry_friction", "viscous_friction")):
        check_values(
            f"friction column {column}", friction[:, column],
            {name: getattr(p, field) for name, p in params.items()},
        )

    robot.write_root_state_to_sim(robot.data.default_root_state.clone())
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
    )
    robot.reset()
    targets = robot.data.default_joint_pos.clone()
    knee_id = robot.find_joints("left_knee_joint")[0][0]

    for step in range(args.steps):
        if not simulation_app.is_running():
            raise RuntimeError("Simulation closed before the check completed.")
        if step == 40:
            targets[:, knee_id] += 0.05
        robot.set_joint_position_target(targets)
        robot.write_data_to_sim()
        sim.step(render=not args.headless)
        robot.update(0.005)
        assert torch.isfinite(robot.data.joint_pos).all(), "Non-finite joint positions."
        assert torch.isfinite(robot.data.joint_vel).all(), "Non-finite joint velocities."
        assert torch.isfinite(robot.data.root_state_w).all(), "Non-finite root state."

    error = (robot.data.joint_pos - targets).abs().max().item()
    print(f"Maximum final joint tracking error: {error:.6f} rad")
    print(f"PASS: import, parameter readback, and {args.steps} finite physics steps.")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
