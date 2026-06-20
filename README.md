<h1 align="center">
  <br>
Robotic Arm Simulation in ROS 2 and Gazebo
  <br>
</h1>

> ## 🚀 Migrated to ROS 2 Jazzy + Gazebo Harmonic — June 2026
>
> This repository has been **fully migrated from ROS 2 Foxy + Gazebo Classic
> (both end-of-life) to ROS 2 Jazzy + Gazebo Harmonic on Ubuntu 24.04.**
>
> The arm, the target sphere, and the full RL environment all run on the new
> stack. Everything below targets Jazzy + Harmonic. If you need the old
> Foxy / Gazebo Classic version, check the git history before this update.

## General Overview

This repository shows, **from scratch**, how to:

1. Simulate a 6-DoF robotic arm (Doosan a0912 / m1013) in **Gazebo** and **ROS 2**.
2. Use a custom **Reinforcement Learning environment** to test the arm with your own RL algorithms.
3. Run a **reacher task**: the arm reaches a visual target (green sphere) that resets to a new random position each episode.

<p align="center">
  <img src="https://github.com/dvalenciar/robotic_arm_environment/blob/main/images/doosan.gif" alt="Doosan arm reacher demo">
</p>

## Prerequisites

You only need the platform installed first:

| Requirement | Version (tested) |
|---|---|
| Ubuntu | 24.04 |
| ROS 2 | Jazzy — [install](https://docs.ros.org/en/jazzy/Installation.html) |

## Dependencies

The `apt` command below installs everything else this repo needs — **Gazebo
Harmonic** (pulled in by `ros-gz`), the `ros_gz` bridge/sim, `ros2_control`,
`gz_ros2_control`, and the controllers. Safe to run on any ROS 2 Jazzy install;
packages you already have are simply skipped.

```bash
sudo apt install ros-jazzy-ros-gz ros-jazzy-gz-ros2-control \
  ros-jazzy-ros2-control ros-jazzy-ros2-controllers \
  ros-jazzy-joint-state-publisher-gui ros-jazzy-xacro \
  ros-jazzy-robot-state-publisher ros-jazzy-rviz2 ros-jazzy-tf2-ros
```

## Installation

Create a colcon workspace and clone this repository into its `src` folder:

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/dvalenciar/robotic_arm_environment.git
cd ~/ros2_ws
```

## Build

```bash
source /opt/ros/jazzy/setup.bash
cd ~/ros2_ws
colcon build
source install/setup.bash
```

## Quick start

```bash
# Full RL environment: arm + target sphere in Gazebo
ros2 launch my_environment_pkg my_environment.launch.py

# In a second terminal — run a few random-action episodes
ros2 run my_environment_pkg run_environment
```

Test pieces individually:

```bash
# Arm only (spawn + joint_trajectory_controller)
ros2 launch my_doosan_pkg my_doosan_gazebo_controller.launch.py

# Arm in RViz with joint sliders
ros2 launch my_doosan_pkg my_doosan_rviz.launch.py

# Target sphere only (standalone)
ros2 launch my_sphere_pkg my_sphere_standalone.launch.py
ros2 run my_sphere_pkg my_client_node   # teleport it to a random pose
```

## Packages

| Package | Role |
|---|---|
| `my_doosan_pkg` | Robot description (xacro), `gz_ros2_control`, worlds, controllers |
| `my_sphere_pkg` | Target sphere: spawn, pose readback, and reset via `ros_gz` |
| `my_environment_pkg` | RL environment node tying the arm + sphere together |

## Citation

If the code helps your work, please star this repo.

## Acknowledgement

Thanks to Doosan Robotics for their repositories and packages:

* https://github.com/doosan-robotics/doosan-robot2
* https://github.com/doosan-robotics/doosan-robot
* https://www.doosanrobotics.com/en/

And to the authors of these repositories and tutorials for the ideas:

* https://github.com/noshluk2/ROS2-Ultimate-learners-Repository/tree/main/bazu
* https://github.com/TomasMerva/ROS_KUKA_env

## Contact

Please open an issue if you have questions or need additional explanations.

###### The released code is only allowed for non-commercial use.