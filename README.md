<h1 align="center">
  <br>
Robotic Arm Simulation in ROS2 and Gazebo
  <br>
</h1>


## General Overview

This repository includes: First, how to simulate a 6DoF Robotic Arm **from scratch** using **GAZEBO** and **ROS2**. Second, it creates a custom Reinforcement Learning Environment for testing the Robotic Arm with various RL algorithms. Finally, we test the simulation and environment with a reacher target task, using RL and the 6DoF Robotic Arm with a visual target point.

![](https://github.com/dvalenciar/robotic_arm_environment/blob/main/images/doosan.gif)

<p align="center">
  < src="https://github.com/dvalenciar/robotic_arm_environment/blob/main/images/doosan.gif">
</p>


## Prerequisites

|Library         | Version (TESTED) |
|----------------------|----|
| Ubuntu | 20.04|
| ROS2| Foxy|
| Python | 3.8|
| ros2_control |[link](https://github.com/ros-controls/ros2_control/tree/foxy) |
| gazebo_ros2_control | [link](https://github.com/ros-simulation/gazebo_ros2_control)|

## How to run this Repository 

In the followings links, you can find a step-by-step instruction section for simulating, controlling, and running this repository with the robotic arm:

* **Simulation in Gazebo and ROS2** --> [Tutorial-link](https://davidvalenciaredro.wixsite.com/my-site/services-7)
  - how to configurate and spawn the robot in Gazebo 
  - add a position controller
   
* **Custom RL Environment** --> [Tutorial-link]
  - how create the RL environment 

* **Reacher task with RL** --> Cooming soon
  - Robot reacher task



## Citation
If you use either the code, data or the step from the tutorial-blog in your paper or project, please kindly star this repo and cite our webpage


## Acknowledgement
I want to thank Doosan Robotics for their tutorials, repositories, and packages where they took some ideas and part of this code.

* https://github.com/doosan-robotics/doosan-robot2
* https://github.com/doosan-robotics/doosan-robot
* https://www.doosanrobotics.com/en/Index

## Contact
Please feel free to contact me or open an issue if you have questions or need additional explanations.

######  The released codes are only allowed for non-commercial use.
