# GUIs for Time-to-Move

We provide two GUIs to generate motion control signals for video generation using **Time-to-Move**:

1. **Cut and Drag GUI** — For object motion control (cut polygons and drag them)
2. **Camera Control GUI** — For camera motion control (3D camera movements using depth)

---

# Cut and Drag GUI

Generate cut-and-drag examples for object motion control.
Given an input frame, you can cut and drag polygons from the initial image, transform their colors, and add external images that can be dragged into the scene.

## ✨ General Guide
- Select an initial image.
- Draw polygons in the image and drag them in several segments.
- During segments you can rotate, scale, and change the polygon colors!
- You can also add an external image into the scene and move it (or polygons cut from it) across segments. Transparency is preserved.
- Write a text prompt that will be used to generate the video afterwards.
- You can preview the motion signal in an in-app demo.
- In the end, all the inputs needed for **Time-to-Move** are saved automatically in a selected output directory.

## 🧰 Requirements
Install dependencies:
```bash
pip install PySide6 opencv-python numpy imageio imageio-ffmpeg
```

## 🚀 Run
Just run the python script:
```bash
python cut_and_drag.py
```

## 🖱️ How to Use
* Select Image — Click 🖼️ Select Image and choose an image.
    * Choose Center Crop / Center Pad at the top of the toolbar if needed.
* Add a Polygon "cutting" the part of the image by clicking Add Polygon.
    * Left-click to add points.
    * After finishing drawing the polygon, press ✅ Finish Polygon Selection.
* Drag to move the polygon
    * During segments you'll see corner circles and a top dot which can be used for scaling and rotating during the segments; in the video the shape is interpolated between the initial frame status and the final segment one.
    * Also, color transformation can be applied (using hue transformation) in the segments to change polygon colors.
    * Click 🎯 End Segment to capture the segment annotated.
    * The movement trajectory can be constructed from multiple segments: repeat move → 🎯 End Segment → move → 🎯 End Segment…
* External Image
    * Another option is to add an external image to the scene.
    * Click 🖼️➕ Add External Image, pick a new image (transparent PNGs are supported).
    * Position/scale/rotate it for its initial pose, then click ✅ Place External Image or right-click on the canvas to lock its starting pose.
    * Now animate it like before: you can move the external image itself, or cut a polygon from it and move it.
* Prompt
    * Type any text prompt you want associated with this example; it will be used later for video generation with our method.
* Preview and Save
    * Preview using ▶️ Play Demo.
    * Click 💾 Save, choose an output folder and then enter a subfolder name.
    * Click 🆕 New to start a new project.

---

# Camera Control GUI

Generate camera motion signals using depth-based 3D reprojection.
Given an input image, depth is estimated automatically using [Depth Pro](https://github.com/apple/ml-depth-pro), and you can control camera pose interactively to create camera motion trajectories.

## ✨ Features
- Automatic depth estimation using Depth Pro
- Interactive 3D camera control (rotation + translation)
- Keyframe-based animation with real-time trajectory visualization
- Preview camera motion in-app

## 🧰 Requirements
Install dependencies:
```bash
pip install PySide6 opencv-python==4.10.0.82 numpy==1.26.4 imageio imageio-ffmpeg torch
pip install git+https://github.com/apple/ml-depth-pro.git
pip install hf_transfer  # Optional: faster model download
```

## 🚀 Run
```bash
python camera_control.py
```

## 🖱️ How to Use
* Select Image — Click 📁 Select Image (depth is estimated automatically)
* Navigate the 3D scene:
    * **Mouse drag**: Rotate view (pitch/yaw)
    * **Mouse wheel**: Zoom in viewing direction
    * **W/S**: Move forward/backward
    * **A/D**: Strafe left/right
    * **Q/E**: Move up/down
    * **Arrow keys**: Rotate view
    * **R**: Reset view
* Add Keyframes — Right-click or click 🎯 Add Keyframe at desired camera poses
* Preview — Click ▶️ Play Demo to preview the interpolated motion
* Export — Click 💾 Save Output, choose folder and subfolder name

---

# Output Files

Both GUIs produce the same output format for Time-to-Move:

| File | Description |
|------|-------------|
| `first_frame.png` | Initial frame for video generation |
| `motion_signal.mp4` | Reference warped video (motion signal) |
| `mask.mp4` | Grayscale mask indicating motion regions |
| `prompt.txt` | Text prompt for video generation |
| `transformations/` | (Camera Control only) Per-frame camera transforms as JSON |

---

## 🧾 License / Credits
Built with PySide6, OpenCV, NumPy, and PyTorch.
Camera Control uses [Depth Pro](https://github.com/apple/ml-depth-pro) for depth estimation.
You own the images and exports you create with these tools.
Motivation for creating an easy-to-use tool from [Go-With-The-Flow](https://github.com/GoWithTheFlowPaper/gowiththeflowpaper.github.io).
