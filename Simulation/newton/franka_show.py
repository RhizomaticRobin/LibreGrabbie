# Spawn the Isaac Franka (Panda) arm in the Kit/PhysX viewer and idle, so the
# attachment surface (panda_link8 flange, where panda_hand bolts on) can be
# clicked/inspected. Default PhysX physics (no Newton).
#
#   ./isaaclab.sh -p franka_show.py
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Franka arm viewer (PhysX) for mount inspection.")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli = parser.parse_args()
if args_cli.headless and args_cli.visualizer == ["kit"]:
    args_cli.visualizer = ["none"]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG


def main() -> None:
    # default SimulationCfg -> PhysX
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args_cli.device))
    sim.set_camera_view(eye=[0.9, -0.9, 0.9], target=[0.4, 0.0, 0.5])

    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8)).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8)))

    sim_utils.create_prim("/World/env_0", "Xform")
    robot = Articulation(FRANKA_PANDA_CFG.replace(prim_path="/World/env_0/Robot"))
    sim.reset()
    print("[INFO] Franka loaded. Flange = /World/env_0/Robot/panda_link8; gripper mount = panda_hand.",
          flush=True)
    print("[INFO] joint names:", list(robot.joint_names), flush=True)

    dt = sim.get_physics_dt()
    # implicit actuators already hold the default joint targets set at reset();
    # just step so the window stays interactive for inspecting/clicking the flange.
    while simulation_app.is_running():
        sim.step()
        robot.update(dt)
    simulation_app.close()


if __name__ == "__main__":
    main()
