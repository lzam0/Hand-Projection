# Hand Projection

A real-time hand tracking application that detects both hands via webcam and applies visual effects to the region between them.

## Features

- Real-time hand landmark detection using MediaPipe
- Detects and distinguishes left and right hands
- Draws skeletal hand landmarks on the video feed
- Connects finger tips across both hands with boundary lines
- Applies visual effects to the region between the hands:
  - **X-ray effect** — inverted, blue-tinted grayscale (middle finger region)
  - **Risograph effect** — colour channel misregistration, grain, and warm pink/teal tint (index finger region)
- FPS counter in the top-left corner

## Requirements

- Python 3.10+
- Webcam

## Installation

```bash
pip install opencv-python mediapipe numpy
```

## Setup

Download the MediaPipe hand landmark model and place it in the project directory:

```bash
curl -o hand_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
```

## Usage

```bash
python3 main.py
```

- Hold both hands in front of the webcam
- The region between your index fingers and thumbs shows the **risograph** effect
- The region between your middle fingers and index fingers shows the **x-ray** effect
- Press `Esc` to quit

## How It Works

MediaPipe detects 21 landmarks per hand. The app uses landmark indices to find specific fingertip positions:

| Landmark | Finger |
|----------|--------|
| 4 | Thumb tip |
| 8 | Index finger tip |
| 12 | Middle finger tip |

The four corner points of each effect region form a quadrilateral mask. Effects are applied only within that mask and processed on the bounding box of the region (not the full frame) to keep performance high.
