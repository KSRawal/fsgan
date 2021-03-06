#!/bin/bash

curl -H "Content-Type: application/json" -d ' {"Source": "/home/kirtikumar/Desktop/CurioVentures/FaceSwap_Project/2_FaceSwapping/FSGAN/projects/fsgan/inference/data/KSR Animated.mp4","Target": "/home/kirtikumar/Desktop/CurioVentures/FaceSwap_Project/2_FaceSwapping/FSGAN/projects/fsgan/inference/data/AB.mp4"} ' -v http://127.0.0.1:5000/predict