<h1 align="center">
  <br>
Robotic Arm Simulation in ROS2 and Gazebo
  <br>
</h1>


## General Overview

This repository includes: First, how to simulate a 6DoF Robotic Arm **from scratch** using **GAZEBO** and **ROS2**. Second, it provides a custom **Reinforcement Learning Environment** where you can test the Robotic Arm with your RL algorithms. Finally, we test the simulation and environment with a reacher target task, using RL and the 6DoF Robotic Arm with a visual target point.

<p align="center">
  <img src="https://github.com/dvalenciar/robotic_arm_environment/blob/main/images/doosan.gif">
</p>


## Prerequisites

|Library         | Version (TESTED) |
|----------------------|----|
| Ubuntu | 20.04|
| ROS2| Foxy|
| ros2_control |[link](https://github.com/ros-controls/ros2_control/tree/foxy) |
| gazebo_ros2_control | [link](https://github.com/ros-simulation/gazebo_ros2_control/tree/foxy)|

## How to run this Repository 

In the following links you can find a step-by-step instruction section to run this repository and simulate the robotic arm:

* **Simulation in Gazebo and ROS2** --> [Tutorial-link](https://davidvalenciaredro.wixsite.com/my-site/services-7)
  - Configurate and spawn the robotic arm in Gazebo 
  - Move the robot with a simple position controller
   
* **Custom RL Environment** --> [Tutorial-link](https://davidvalenciaredro.wixsite.com/my-site/services-7-1)
  - A complete Reinforcement Learning environment simulation 

* **Reacher task with RL** --> Cooming soon
  - Robot reacher task



## Citation
If you use either the code, data or the step from the tutorial-blog in your paper or project, please kindly star this repo and cite our webpage


## Acknowledgement
I want to thank Doosan Robotics for their repositories, and packages where they took part of this code.

* https://github.com/doosan-robotics/doosan-robot2
* https://github.com/doosan-robotics/doosan-robot
* https://www.doosanrobotics.com/en/Index

Also, thanks to the authors of these repositories and their tutorials where I took some ideas  

* https://github.com/noshluk2/ROS2-Ultimate-learners-Repository/tree/main/bazu
* https://github.com/TomasMerva/ROS_KUKA_env


## Contact
Please feel free to contact me or open an issue if you have questions or need additional explanations.

######  The released codes are only allowed for non-commercial use.
