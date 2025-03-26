# LibreGrabbie

**/ˈliː.brə ˈɡræb.i/** (LEE-bruh GRAB-ee) - *LibreGrabbies* (plural): /ˈliː.brə ˈɡræb.iz/ (LEE-bruh GRAB-eez)

An advanced open-source robotic hand with hybrid soft robotics for superior grasping capabilities

![robot hand](https://github.com/user-attachments/assets/74bdee07-ca2a-47cd-8f38-99ab58790e74)
![robot_hand_back](https://github.com/user-attachments/assets/3834614a-edf5-4b00-b480-64521c76e71a)

## Introduction

"What? Another esoteric 3D printed robot hand project that's basically just a toy/puppet/useless gizmo dohicky with 5 string motors? No! There's too many of those already!"

Well, here's where I might change your mind regarding that!

## Features

- **Flexible TPU palm mechanics**, allowing for hybrid soft robotics that allow for actual grasping around curved objects in comparison to current traditional robotic palm mechanics
- 16 motors allowing for **16 DOF hand dexerity**, 3 DOF for each finger and 4 DOF on the thumb
- Straightforward 3D printing and assembly with **only 5-10 3D printed parts** to be assembled together for more plug-and-play use dynamics
- Mechanics and electronics **entirely constrained** inside the palm, allowing for full end-effector modularity with all current and future robotic manipulators
- To utilize a modified L3-F-TOUCH sensing module for the fingers and palm and a stereo raspberry pi camera system for sensor fusion data to train reliable deep reinforcement learning 
- **Universal Robot Attachment** system that can be 3D printed to interface with any robot arm's end effector attachment system
- **Multi-modal sensor fusion** integrating:
  - Intel® RealSense™ Depth Camera D455
  - L3-F-TOUCH sensing for tactile feedback
  - Arduino Pro Mini for motor encoder data collection

## AI & Simulation

- **Proximal Policy Optimization (PPO)** Deep Reinforcement Learning algorithm for training
- **NVIDIA's Isaac Labs and Isaac Sim** for physics-accurate simulation and training
- **Upcoming trained AI/RL models** to be released allowing for automated grasping capabilities of generalized objects

## Related Resources

- [L3-F-TOUCH shown](https://youtu.be/ASt3WRFcAxU?si=XV7Dn4dw2RqogxgP)
- [General Concept of touch module](https://youtu.be/qtQ4rK66vlE?si=lcGbKkfz59pFsFFY)
- [Simulation environment to be possibly integrated](https://github.com/facebookresearch/tacto)

## Applications

- Primary use as a **robotic end effector** with advanced grasping capabilities
- Secondary application as a **low-weight and cost-accessible hand for prosthetics design**, especially if built with worm drive gearboxes for human-level gripping strength
- Customizable socket system (hopefully also full copyleft) for any given person

## Project Details

- **<$500 cost to build**
- **Fully open source hardware and software** under the GPL 3.0 and CERN-OHL pair license :)

## Coming Soon

- Assembly instructions
- Component lists
- Software setup guide
- Simulation integration details
- Attachment system designs
