# aBAJA2025
Real-time Lane and Object Detection for ADAS using YOLOv8 and classical computer vision, integrated in a ROS 2 workspace with IPG CarMaker simulation.

# 🚗 ADAS Perception Module – Lane & Object Detection

This repository contains the implementation of a real-time **Lane Detection** and **Object Detection** system as part of an Advanced Driver Assistance System (ADAS) perception stack. It integrates classical computer vision techniques with modern deep learning (YOLOv8) inside a ROS 2 workspace, simulated using **IPG CarMaker**.

---

## 📹 Overview

The perception system includes:
- 🛣 **Lane Detection** using OpenCV and NumPy
- 🧠 **Object Detection** using [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- 🔄 ROS 2 communication to handle data streaming between modules
- 🧪 Simulation setup using IPG CarMaker

---

## 🧠 Object Detection – YOLOv8
- Model: YOLOv8m
- Framework: Ultralytics
- Features: Bounding box visualization, class labels, real-time inference

---

## 🛣 Lane Detection – Classical Vision Pipeline
- 🎨 HSV filtering to detect yellow/white lanes
- 🧹 Morphological operations to reduce noise
- 📐 Perspective (bird’s-eye view) transformation
- 🔍 Sliding window and polynomial fitting for lane curves
- 🧠 Curvature & steering angle estimation
- 🖼️ Final overlay visualization on the original frame

---

## 🛠 System Architecture

- 🔧 **Simulation**: IPG CarMaker provides camera feeds from a virtual driving environment.
- 🌐 **ROS 2**: Custom nodes publish and subscribe to image and processed data topics.
- 💻 **Language**: Python (with OpenCV, NumPy, cv_bridge)

---

## 📁 Folder Structure

