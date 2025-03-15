# Emotion Detector


A Python-based GUI application to detect emotions from facial expressions in images using a pre-trained deep learning model.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

The **Emotion Detector** is a desktop application built using Python, TensorFlow, and OpenCV. It allows users to upload an image, detects faces in the image, and predicts the emotion of each face using a pre-trained deep learning model. The application is built with a user-friendly GUI using the `tkinter` library.

---

## Features

- **Upload Images**: Users can upload images in common formats (JPEG, PNG, etc.).
- **Face Detection**: Detects faces in the uploaded image using Haar Cascades.
- **Emotion Prediction**: Predicts emotions (angry, disgust, fear, happy, neutral, sad, surprise) using a pre-trained deep learning model.
- **User-Friendly Interface**: Simple and intuitive GUI for easy interaction.

---

## Requirements

To run this project, you need the following dependencies:

- Python 3.8 or higher
- TensorFlow 2.x
- OpenCV
- NumPy
- Pillow
- tkinter (included with Python)

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/emotion-detector.git
   cd emotion-detector
