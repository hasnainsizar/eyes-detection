# 👁️ Eye Drowsiness Detection

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)](https://opencv.org/)
[![Mediapipe](https://img.shields.io/badge/Mediapipe-0.10+-orange)](https://google.github.io/mediapipe/)

Real-time drowsiness detection using **MediaPipe FaceMesh**.  
Predicts `alert`, `mildly_drowsy`, or `drowsy` from **blink rate, blink duration, and PERCLOS**.

---

## 🚀 Quick Start

```bash
# Clone & enter repo
git clone https://github.com/<your-username>/eyes-detection.git
cd eyes-detection

# Setup environment
conda create -n eyes python=3.11 -y
conda activate eyes
conda install -c conda-forge opencv ffmpeg pyqt -y
pip install mediapipe numpy
