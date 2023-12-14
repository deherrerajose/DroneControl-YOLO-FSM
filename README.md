
# Drone Control System

# Project Overview

<p> DroneControl-YOLO-FSM is an advanced Python-based system designed to control a Tello drone using object detection capabilities. It integrates the YOLOv3 model for efficient and accurate object detection with a Finite State Machine (FSM) approach for dynamic and responsive drone control. This system is ideal for hobbyists, researchers, and developers interested in exploring drone technology and computer vision. </p>

# Directory Structure

<h2> DroneControl-YOLO-FSM: </h2> Contains the main scripts for the project.
 <h4> main.py: </h4> Integrates object detection with drone control.
 <h4> fsm.py: </h4> Manages drone states and commands.
 <h4> /models: </h4> Stores machine learning models.
 <h4> yolov3.weights: </h4> Weights for the YOLOv3 model.
 <h4> yolov3.cfg: </h4> Configuration file for YOLOv3.
 <h4> /data: </h4> Additional data files.
 <h4> coco.names: </h4> Object classes for YOLO detection.

# Setup
  # Prerequisites
    Python 3.x
    virtualenv library
    Tello drone with Python SDK

# Python Virtual Environment
Using the .env virtual environment ensures consistent running conditions.

### Install virtualenv if not installed:
    pip install virtualenv

### Create and activate .env:
    Windows: .env\Scripts\activate
    macOS/Linux: source .env/bin/activate

### Installation
  Clone the repository.
  Set up .env virtual environment.
  Add YOLO model files to /models.

# YOLOv3 Object Detection
  YOLOv3 is used for real-time object detection, analyzing the drone's video stream. This model is known for its speed and accuracy.

 <h4> Model Files: </h4> Located in /models.
 <h4> Object Classes: </h4> Defined in coco.names.
 <h4> Integration: </h4>YOLOv3 is loaded in main.py for video analysis.

# Running the Project
  Activate the .env environment.
  Connect to the drone's Wi-Fi.
  Run main.py in /drone_control.

# Usage
  main.py for drone video stream and object detection.
  fsm.py to modify drone responses and states.

# Troubleshooting
  <p> Update drone firmware. </p>
  <p>Check drone battery. </p>
  <p>Verify serial communication settings. </p>

# Contributing
  Contributions are welcome. Fork the project for improvements or fixes.
