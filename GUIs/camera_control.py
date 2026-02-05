# Copyright 2025 Noam Rotstein
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os, sys

# Enable fast HuggingFace downloads (must be before ANY huggingface imports)
try:
    import hf_transfer
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
except ImportError:
    pass

import cv2, json, time
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from PIL import Image
import depth_pro

from PySide6 import QtCore, QtGui
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap, QKeyEvent
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QLabel, QSpinBox,
    QWidget, QMessageBox, QPushButton, QVBoxLayout, QHBoxLayout,
    QPlainTextEdit, QGroupBox, QComboBox, QInputDialog
)
import imageio

# ------------------------------
# Utility: numpy <-> QPixmap
# ------------------------------

def np_bgr_to_qpixmap(img_bgr: np.ndarray) -> QPixmap:
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    qimg = QImage(img_rgb.data, w, h, img_rgb.strides[0], QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())

# ------------------------------
# Image I/O + fit helpers
# ------------------------------

def load_first_frame(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    low = path.lower()
    if low.endswith((".mp4", ".mov", ".avi", ".mkv")):
        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError("Failed to read first frame from video")
        return frame
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Failed to read image")
    return img

def resize_then_center_crop(img: np.ndarray, target_h: int, target_w: int, interpolation=cv2.INTER_NEAREST) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max(target_w / float(w), target_h / float(h))
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    y0 = (new_h - target_h) // 2
    x0 = (new_w - target_w) // 2
    return resized[y0:y0 + target_h, x0:x0 + target_w]

def fit_center_pad(img: np.ndarray, target_h: int, target_w: int, interpolation=cv2.INTER_NEAREST) -> np.ndarray:
    h, w = img.shape[:2]
    scale_h = target_h / float(h)
    new_w_hfirst = int(round(w * scale_h))
    new_h_hfirst = target_h
    if new_w_hfirst <= target_w:
        resized = cv2.resize(img, (new_w_hfirst, new_h_hfirst), interpolation=interpolation)
        result = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        x0 = (target_w - new_w_hfirst) // 2
        result[:, x0:x0 + new_w_hfirst] = resized
        return result
    scale_w = target_w / float(w)
    new_w_wfirst = target_w
    new_h_wfirst = int(round(h * scale_w))
    resized = cv2.resize(img, (new_w_wfirst, new_h_wfirst), interpolation=interpolation)
    result = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y0 = (target_h - new_h_wfirst) // 2
    result[y0:y0 + new_h_wfirst, :] = resized
    return result

def fill_black_with_nearest(img):
    """Fill black/empty pixels with nearest foreground color using distance transform."""
    mask = np.all(img == [0, 0, 0], axis=-1).astype(np.uint8)
    if mask.sum() == img.shape[0] * img.shape[1]:
        return img, ~mask.astype(bool)
    dist, labels = cv2.distanceTransformWithLabels(mask, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    fg_coords = np.column_stack(np.where(mask == 0))
    nearest_coords = fg_coords[labels.ravel() - 1]
    nearest_coords = nearest_coords.reshape(img.shape[:2] + (2,))
    filled = img.copy()
    black_idx = mask.astype(bool)
    filled[black_idx] = img[nearest_coords[black_idx, 0], nearest_coords[black_idx, 1]]
    return filled, ~black_idx

def save_video_mp4(frames_bgr, path, fps=24):
    if not frames_bgr:
        raise ValueError("No frames to save")
    h, w = frames_bgr[0].shape[:2]
    out_frames = []
    for f in frames_bgr:
        if f is None:
            raise RuntimeError("Encountered None frame")
        if f.ndim == 2:
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        elif f.shape[2] == 4:
            f = cv2.cvtColor(f, cv2.COLOR_BGRA2BGR)
        if f.dtype != np.uint8:
            f = np.clip(f, 0, 255).astype(np.uint8)
        out_frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    hh = h - (h % 2)
    ww = w - (w % 2)
    if (hh != h) or (ww != w):
        out_frames = [frm[:hh, :ww] for frm in out_frames]
    ffmpeg_common = ['-movflags', '+faststart', '-colorspace', 'bt709', '-color_primaries', 'bt709',
                     '-color_trc', 'bt709', '-tag:v', 'avc1']
    try:
        writer = imageio.get_writer(path, format='ffmpeg', fps=float(fps), codec='libx264',
                                    pixelformat='yuv420p', ffmpeg_params=ffmpeg_common)
    except Exception:
        writer = imageio.get_writer(path, format='ffmpeg', fps=float(fps), codec='mpeg4',
                                    pixelformat='yuv420p', ffmpeg_params=['-movflags', '+faststart'])
    try:
        for frm in out_frames:
            writer.append_data(frm)
    finally:
        writer.close()
    return path

# ------------------------------
# 3D Camera and Depth Utilities
# ------------------------------

@dataclass
class CameraPose:
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0

    def to_matrix(self) -> np.ndarray:
        rx_rad, ry_rad, rz_rad = np.radians(self.rx), np.radians(self.ry), np.radians(self.rz)
        Rx = np.array([[1, 0, 0], [0, np.cos(rx_rad), -np.sin(rx_rad)], [0, np.sin(rx_rad), np.cos(rx_rad)]])
        Ry = np.array([[np.cos(ry_rad), 0, np.sin(ry_rad)], [0, 1, 0], [-np.sin(ry_rad), 0, np.cos(ry_rad)]])
        Rz = np.array([[np.cos(rz_rad), -np.sin(rz_rad), 0], [np.sin(rz_rad), np.cos(rz_rad), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [self.tx, self.ty, self.tz]
        return T

    def copy(self) -> 'CameraPose':
        return CameraPose(self.tx, self.ty, self.tz, self.rx, self.ry, self.rz)

    def to_dict(self) -> dict:
        return {'tx': self.tx, 'ty': self.ty, 'tz': self.tz, 'rx': self.rx, 'ry': self.ry, 'rz': self.rz}


class DepthEstimator:
    """Depth estimator using Depth Pro (Apple's metric depth model)"""
    _model = None
    _transform = None
    _device = None
    _status_callback = None

    @classmethod
    def set_status_callback(cls, callback):
        cls._status_callback = callback

    @classmethod
    def _update_status(cls, msg):
        if cls._status_callback:
            cls._status_callback(msg)

    @classmethod
    def _ensure_checkpoint(cls):
        """Download checkpoint from Apple CDN if not present locally"""
        import urllib.request
        
        # Check local checkpoints directory first
        script_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_dir = os.path.join(script_dir, "checkpoints")
        checkpoint_path = os.path.join(checkpoint_dir, "depth_pro.pt")
        
        if os.path.exists(checkpoint_path):
            return checkpoint_path
        
        # Download from Apple CDN
        cls._update_status("Downloading DepthPro model (~500MB)...")
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        url = "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt"
        
        # Download with progress
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                pct = min(100, int(block_num * block_size * 100 / total_size))
                cls._update_status(f"Downloading DepthPro... {pct}%")
                QApplication.processEvents()
        
        urllib.request.urlretrieve(url, checkpoint_path, progress_hook)
        return checkpoint_path

    @classmethod
    def get_model(cls):
        if cls._model is None:
            # Ensure checkpoint exists (downloads if needed)
            checkpoint_path = cls._ensure_checkpoint()
            
            cls._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            device_name = "GPU" if cls._device.type == "cuda" else "CPU"
            cls._update_status(f"Loading depth model on {device_name}...")
            from PySide6.QtWidgets import QApplication
            QApplication.processEvents()
            
            # Create config with our checkpoint path
            config = depth_pro.depth_pro.DepthProConfig(
                patch_encoder_preset="dinov2l16_384",
                image_encoder_preset="dinov2l16_384",
                decoder_features=256,
                checkpoint_uri=checkpoint_path,
                use_fov_head=True,
                fov_encoder_preset="dinov2l16_384",
            )
            cls._model, cls._transform = depth_pro.create_model_and_transforms(
                config=config,
                device=cls._device
            )
            cls._model.eval()
        return cls._model, cls._transform, cls._device

    @classmethod
    def get_device_name(cls):
        if cls._device is None:
            return "CPU"
        return "GPU" if cls._device.type == "cuda" else "CPU"

    @staticmethod
    def estimate_depth(image_bgr: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
        model, transform, device = DepthEstimator.get_model()
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        orig_h, orig_w = image_bgr.shape[:2]
        with torch.no_grad():
            image_tensor = transform(pil_image)
            input_h, input_w = image_tensor.shape[-2:]
            scale_x = orig_w / float(input_w)
            image_tensor = image_tensor.to(device)
            prediction = model.infer(image_tensor)
            depth = prediction["depth"].detach().cpu().numpy()
            focal_length_px = None
            if "focallength_px" in prediction:
                focal_length_px = prediction["focallength_px"].item() * scale_x
        depth = np.squeeze(depth)
        if depth.shape != (orig_h, orig_w):
            depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        depth = np.clip(depth.astype(np.float32), 0.1, 100.0)
        return depth, focal_length_px


class CameraProjector:
    """Handles 3D projection and warping based on camera pose and depth"""
    
    def __init__(self, image_bgr: np.ndarray, depth: np.ndarray, focal_length: float = None):
        self.image = image_bgr
        self.depth = depth
        self.h, self.w = image_bgr.shape[:2]
        
        if focal_length is None or focal_length > self.w * 1.5:
            fov_deg = 55.0
            self.focal = self.w / (2.0 * np.tan(np.radians(fov_deg / 2.0)))
        else:
            self.focal = focal_length
        
        self.cx = self.w / 2.0
        self.cy = self.h / 2.0
        
        u, v = np.meshgrid(np.arange(self.w), np.arange(self.h))
        X = (u - self.cx) * self.depth / self.focal
        Y = (v - self.cy) * self.depth / self.focal
        self.point_cloud = np.stack([X, Y, self.depth], axis=2)

    def warp_to_new_pose(self, pose: CameraPose) -> Tuple[np.ndarray, np.ndarray]:
        T_c2w = pose.to_matrix().astype(np.float32)
        T_w2c = np.linalg.inv(T_c2w)
        
        points_flat = self.point_cloud.reshape(-1, 3)
        points_homo = np.hstack([points_flat, np.ones((points_flat.shape[0], 1), dtype=np.float32)])
        points_new = (T_w2c @ points_homo.T).T[:, :3]
        
        X, Y, Z = points_new[:, 0], points_new[:, 1], points_new[:, 2]
        valid_depth = Z > 0.01
        
        u = (self.focal * X / Z + self.cx).astype(np.float32)
        v = (self.focal * Y / Z + self.cy).astype(np.float32)
        u_int = np.round(u).astype(np.int32)
        v_int = np.round(v).astype(np.int32)
        
        valid_bounds = (u_int >= 0) & (u_int < self.w) & (v_int >= 0) & (v_int < self.h)
        valid = valid_depth & valid_bounds
        
        valid_idx = np.where(valid)[0]
        u_valid = u_int[valid_idx]
        v_valid = v_int[valid_idx]
        z_valid = Z[valid_idx]
        
        colors_flat = self.image.reshape(-1, 3)
        colors_valid = colors_flat[valid_idx]
        
        output = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        mask = np.zeros((self.h, self.w), dtype=np.uint8)
        
        sort_idx = np.argsort(-z_valid)
        output[v_valid[sort_idx], u_valid[sort_idx]] = colors_valid[sort_idx]
        mask[v_valid[sort_idx], u_valid[sort_idx]] = 255
        
        if (mask == 0).any():
            output, _ = fill_black_with_nearest(output)
        
        return output, mask


# ------------------------------
# Interactive Camera View Widget
# ------------------------------

class CameraView(QLabel):
    poseChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("QLabel { background-color: #2b2b2b; border: 2px solid #555; }")
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        
        self.projector: Optional[CameraProjector] = None
        self.current_pose = CameraPose()
        self.original_image: Optional[np.ndarray] = None
        self.last_mouse_pos: Optional[QtCore.QPoint] = None
        self.mouse_active = False
        
        self.translation_speed = 0.05
        self.rotation_speed = 0.3

    def focusInEvent(self, event):
        self.setStyleSheet("QLabel { background-color: #2b2b2b; border: 3px solid #4a9eff; }")
        super().focusInEvent(event)

    def focusOutEvent(self, event):
        self.setStyleSheet("QLabel { background-color: #2b2b2b; border: 2px solid #555; }")
        super().focusOutEvent(event)

    def set_image_and_depth(self, image_bgr: np.ndarray, depth: np.ndarray, focal_length: float = None):
        self.original_image = image_bgr.copy()
        self.projector = CameraProjector(image_bgr, depth, focal_length)
        self.current_pose = CameraPose()
        self.update_view()
        self.setFocus()

    def update_view(self):
        if self.projector is None:
            return
        try:
            warped, _ = self.projector.warp_to_new_pose(self.current_pose)
            pixmap = np_bgr_to_qpixmap(warped)
            self.setPixmap(pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.poseChanged.emit()
        except Exception:
            if self.original_image is not None:
                pixmap = np_bgr_to_qpixmap(self.original_image)
                self.setPixmap(pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def reset_pose(self):
        self.current_pose = CameraPose()
        self.update_view()

    def get_current_pose(self) -> CameraPose:
        return self.current_pose.copy()

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        self.setFocus(Qt.MouseFocusReason)
        if event.button() == Qt.LeftButton:
            self.mouse_active = True
            self.last_mouse_pos = event.pos()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.mouse_active = False
            self.last_mouse_pos = None
        self.setFocus(Qt.MouseFocusReason)
        event.accept()

    def wheelEvent(self, event: QtGui.QWheelEvent):
        if self.projector is None:
            return
        delta = event.angleDelta().y()
        zoom_amount = (delta / 120.0) * self.translation_speed * 2.0
        rx_rad = np.radians(self.current_pose.rx)
        ry_rad = np.radians(self.current_pose.ry)
        Rx = np.array([[1, 0, 0], [0, np.cos(rx_rad), -np.sin(rx_rad)], [0, np.sin(rx_rad), np.cos(rx_rad)]])
        Ry = np.array([[np.cos(ry_rad), 0, np.sin(ry_rad)], [0, 1, 0], [-np.sin(ry_rad), 0, np.cos(ry_rad)]])
        forward = (Ry @ Rx)[:, 2]
        self.current_pose.tx += forward[0] * zoom_amount
        self.current_pose.ty += forward[1] * zoom_amount
        self.current_pose.tz += forward[2] * zoom_amount
        self.update_view()
        self.setFocus(Qt.MouseFocusReason)
        event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        if self.mouse_active and self.last_mouse_pos is not None:
            delta = event.pos() - self.last_mouse_pos
            self.current_pose.ry -= delta.x() * self.rotation_speed * 0.5
            self.current_pose.rx += delta.y() * self.rotation_speed * 0.5
            self.current_pose.rx = np.clip(self.current_pose.rx, -89, 89)
            self.last_mouse_pos = event.pos()
            self.update_view()
        event.accept()

    def keyPressEvent(self, event: QKeyEvent):
        if self.projector is None:
            return
        key = event.key()
        is_repeat = event.isAutoRepeat()
        move_step = self.translation_speed * (0.375 if is_repeat else 0.15)
        rotate_step = 0.8 if is_repeat else 0.3
        moved = False

        if key == Qt.Key_W:
            self.current_pose.tz += move_step; moved = True
        elif key == Qt.Key_S:
            self.current_pose.tz -= move_step; moved = True
        elif key == Qt.Key_A:
            self.current_pose.tx -= move_step; moved = True
        elif key == Qt.Key_D:
            self.current_pose.tx += move_step; moved = True
        elif key == Qt.Key_Q:
            self.current_pose.ty += move_step; moved = True
        elif key == Qt.Key_E:
            self.current_pose.ty -= move_step; moved = True
        elif key == Qt.Key_Space:
            self.current_pose.ty += move_step; moved = True
        elif key == Qt.Key_R:
            self.current_pose = CameraPose(); moved = True
        elif key == Qt.Key_Left:
            self.current_pose.ry -= rotate_step; moved = True
        elif key == Qt.Key_Right:
            self.current_pose.ry += rotate_step; moved = True
        elif key == Qt.Key_Up:
            self.current_pose.rx = min(89, self.current_pose.rx + rotate_step); moved = True
        elif key == Qt.Key_Down:
            self.current_pose.rx = max(-89, self.current_pose.rx - rotate_step); moved = True
        else:
            super().keyPressEvent(event); return

        if moved:
            self.update_view()
        event.accept()


# ------------------------------
# Main Window
# ------------------------------

class CameraControlWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera Control GUI - Time-to-Move")
        self.setGeometry(100, 100, 1200, 800)
        
        self.base_image: Optional[np.ndarray] = None
        self.depth_map: Optional[np.ndarray] = None
        self.focal_length_px: Optional[float] = None
        self.keyframes: List[CameraPose] = []
        
        self.preview_frames: List[QPixmap] = []
        self.preview_index = 0
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self._on_play_timer_tick)
        
        self._init_ui()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        # Left: Camera view
        view_layout = QVBoxLayout()
        self.camera_view = CameraView()
        view_layout.addWidget(self.camera_view)
        self.camera_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.camera_view.customContextMenuRequested.connect(lambda: self.on_add_keyframe())
        self.camera_view.poseChanged.connect(self._update_depth_trajectory)
        
        instructions = QPlainTextEdit()
        instructions.setReadOnly(True)
        instructions.setMaximumHeight(100)
        instructions.setPlainText(
            "CONTROLS:\n"
            "• Mouse drag: Rotate view  • Mouse wheel: Zoom\n"
            "• W/S: Forward/back  • A/D: Left/right  • Q/E: Up/down  • R: Reset\n"
            "• Right-click OR button: Add keyframe"
        )
        view_layout.addWidget(instructions)
        main_layout.addLayout(view_layout, stretch=3)
        
        # Right: Controls
        control_layout = QVBoxLayout()
        
        # 1. Load Image
        img_group = QGroupBox("1. Load Image")
        img_layout = QVBoxLayout()
        self.btn_load = QPushButton("📁 Select Image")
        self.btn_load.clicked.connect(self.on_load_image)
        img_layout.addWidget(self.btn_load)
        self.lbl_image_info = QLabel("No image loaded")
        img_layout.addWidget(self.lbl_image_info)
        img_group.setLayout(img_layout)
        control_layout.addWidget(img_group)
        
        # 2. Depth
        depth_group = QGroupBox("2. Depth")
        depth_layout = QVBoxLayout()
        self.lbl_depth_info = QLabel("No depth")
        depth_layout.addWidget(self.lbl_depth_info)
        self.lbl_depth_preview = QLabel()
        self.lbl_depth_preview.setFixedSize(200, 150)
        self.lbl_depth_preview.setAlignment(Qt.AlignCenter)
        self.lbl_depth_preview.setStyleSheet("QLabel { border: 1px solid #555; background-color: #1a1a1a; }")
        depth_layout.addWidget(self.lbl_depth_preview)
        depth_group.setLayout(depth_layout)
        control_layout.addWidget(depth_group)
        
        # 3. Keyframes
        kf_group = QGroupBox("3. Keyframes")
        kf_layout = QVBoxLayout()
        self.btn_add_keyframe = QPushButton("🎯 Add Keyframe")
        self.btn_add_keyframe.clicked.connect(self.on_add_keyframe)
        self.btn_add_keyframe.setEnabled(False)
        self.btn_add_keyframe.setStyleSheet("QPushButton { background-color: #3a7; color: white; font-weight: bold; }")
        kf_layout.addWidget(self.btn_add_keyframe)
        self.btn_undo_keyframe = QPushButton("↩️ Undo Last")
        self.btn_undo_keyframe.clicked.connect(self.on_undo_keyframe)
        self.btn_undo_keyframe.setEnabled(False)
        kf_layout.addWidget(self.btn_undo_keyframe)
        reset_row = QHBoxLayout()
        self.btn_reset = QPushButton("🔄 Reset")
        self.btn_reset.clicked.connect(self.on_reset_camera)
        self.btn_reset.setEnabled(False)
        reset_row.addWidget(self.btn_reset)
        self.btn_clear = QPushButton("🗑️ Clear")
        self.btn_clear.clicked.connect(self.on_clear_keyframes)
        self.btn_clear.setEnabled(False)
        reset_row.addWidget(self.btn_clear)
        kf_layout.addLayout(reset_row)
        self.lbl_keyframe_info = QLabel("Keyframes: 0")
        kf_layout.addWidget(self.lbl_keyframe_info)
        kf_group.setLayout(kf_layout)
        control_layout.addWidget(kf_group)
        
        # 4. Preview
        play_group = QGroupBox("4. Preview")
        play_layout = QVBoxLayout()
        settings_layout = QHBoxLayout()
        settings_layout.addWidget(QLabel("Frames:"))
        self.spn_total_frames = QSpinBox()
        self.spn_total_frames.setRange(1, 2000)
        self.spn_total_frames.setValue(81)
        settings_layout.addWidget(self.spn_total_frames)
        settings_layout.addWidget(QLabel("FPS:"))
        self.spn_fps = QSpinBox()
        self.spn_fps.setRange(1, 120)
        self.spn_fps.setValue(16)
        settings_layout.addWidget(self.spn_fps)
        settings_layout.addStretch()
        play_layout.addLayout(settings_layout)
        self.btn_play = QPushButton("▶️ Play Demo")
        self.btn_play.clicked.connect(self.on_play_demo)
        self.btn_play.setEnabled(False)
        play_layout.addWidget(self.btn_play)
        play_group.setLayout(play_layout)
        control_layout.addWidget(play_group)
        
        # 5. Export
        export_group = QGroupBox("5. Export")
        export_layout = QVBoxLayout()
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Size:"))
        self.spn_width = QSpinBox()
        self.spn_width.setRange(256, 2048)
        self.spn_width.setValue(720)
        size_layout.addWidget(self.spn_width)
        size_layout.addWidget(QLabel("×"))
        self.spn_height = QSpinBox()
        self.spn_height.setRange(256, 2048)
        self.spn_height.setValue(480)
        size_layout.addWidget(self.spn_height)
        export_layout.addLayout(size_layout)
        fit_layout = QHBoxLayout()
        fit_layout.addWidget(QLabel("Fit:"))
        self.fit_mode_combo = QComboBox()
        self.fit_mode_combo.addItems(["Center Crop", "Fit & Pad"])
        fit_layout.addWidget(self.fit_mode_combo)
        fit_layout.addStretch()
        export_layout.addLayout(fit_layout)
        export_layout.addWidget(QLabel("Prompt:"))
        self.txt_prompt = QPlainTextEdit()
        self.txt_prompt.setPlaceholderText("Enter a description...")
        self.txt_prompt.setMaximumHeight(80)
        export_layout.addWidget(self.txt_prompt)
        self.btn_save = QPushButton("💾 Save Output")
        self.btn_save.clicked.connect(self.on_save_output)
        self.btn_save.setEnabled(False)
        export_layout.addWidget(self.btn_save)
        export_group.setLayout(export_layout)
        control_layout.addWidget(export_group)
        
        control_layout.addStretch()
        main_layout.addLayout(control_layout, stretch=1)

    def _show_overlay_text(self, text: str):
        """Show big overlay text on the image preview"""
        if self.base_image is None:
            return
        h, w = self.base_image.shape[:2]
        img_with_text = self.base_image.copy()
        overlay = img_with_text.copy()
        cv2.rectangle(overlay, (0, h//2 - 40), (w, h//2 + 40), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img_with_text, 0.4, 0, img_with_text)
        font_scale = min(1.2, w / (len(text) * 18 + 1))
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(img_with_text, text, (text_x, h//2 + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
        self.camera_view.setPixmap(np_bgr_to_qpixmap(img_with_text).scaled(
            self.camera_view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        QApplication.processEvents()

    def on_load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All Files (*)")
        if not path:
            return
        try:
            self.base_image = load_first_frame(path)
            self.lbl_image_info.setText("✓ Image loaded")
            self._show_overlay_text("Initializing...")
            
            self.depth_map = None
            self.keyframes = []
            self._update_keyframe_ui()
            self.btn_reset.setEnabled(False)
            self.btn_add_keyframe.setEnabled(False)
            self.btn_play.setEnabled(False)
            self.btn_save.setEnabled(False)
            QApplication.processEvents()
            self._estimate_depth()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image:\n{e}")

    def _estimate_depth(self):
        if self.base_image is None:
            return
        def update_status(msg):
            self.lbl_depth_info.setText(msg)
            if "Downloading" in msg:
                # Extract percentage if present (e.g., "Downloading DepthPro... 45%")
                self._show_overlay_text(msg.replace("DepthPro", "").strip())
            elif "Loading" in msg:
                device = "GPU" if "GPU" in msg else "CPU"
                self._show_overlay_text(f"Loading DepthPro ({device})...")
            QApplication.processEvents()
        DepthEstimator.set_status_callback(update_status)
        self.lbl_depth_info.setText("Initializing depth model...")
        device_name = "GPU" if torch.cuda.is_available() else "CPU"
        self._show_overlay_text(f"Estimating depth ({device_name})...")
        QApplication.processEvents()
        try:
            self.depth_map, self.focal_length_px = DepthEstimator.estimate_depth(self.base_image)
            device_name = DepthEstimator.get_device_name()
            self.lbl_depth_info.setText(f"✓ Depth estimated ({device_name})")
            QApplication.processEvents()
            self._display_depth_and_setup()
        except Exception as e:
            self.lbl_depth_info.setText("Depth failed")
            QMessageBox.critical(self, "Error", f"Depth estimation failed:\n{e}")

    def _display_depth_and_setup(self):
        depth_normalized = ((self.depth_map - self.depth_map.min()) /
                           (self.depth_map.max() - self.depth_map.min()) * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_MAGMA)
        h, w = depth_colored.shape[:2]
        aspect = w / h
        thumb_h = 150
        thumb_w = min(200, int(thumb_h * aspect))
        if thumb_w > 200:
            thumb_w = 200
            thumb_h = int(thumb_w / aspect)
        self.depth_thumb_base = cv2.resize(depth_colored, (thumb_w, thumb_h))
        self.depth_thumb_size = (thumb_w, thumb_h)
        self.lbl_depth_preview.setPixmap(np_bgr_to_qpixmap(self.depth_thumb_base))
        QApplication.processEvents()
        
        self.camera_view.set_image_and_depth(self.base_image, self.depth_map, self.focal_length_px)
        self.lbl_depth_info.setText("✓ Ready")
        self.btn_reset.setEnabled(True)
        self.btn_add_keyframe.setEnabled(True)
        self._add_keyframe_at_current_pose(is_initial=True)

    def on_reset_camera(self):
        self.camera_view.reset_pose()
        self.keyframes = []
        self._add_keyframe_at_current_pose(is_initial=True)

    def _update_depth_trajectory(self):
        if not hasattr(self, 'depth_thumb_base') or self.depth_thumb_base is None:
            return
        depth_viz = self.depth_thumb_base.copy()
        thumb_w, thumb_h = self.depth_thumb_size
        scale = 40.0
        cx, cy = thumb_w // 2, thumb_h // 2

        def pose_to_point(pose):
            px = int(np.clip(cx + pose.tx * scale, 4, thumb_w - 5))
            py = int(np.clip(cy + pose.ty * scale, 4, thumb_h - 5))
            return (px, py)

        kf_points = [pose_to_point(kf) for kf in self.keyframes]
        for i in range(len(kf_points) - 1):
            cv2.line(depth_viz, kf_points[i], kf_points[i + 1], (255, 200, 100), 2, cv2.LINE_AA)
        for i, pt in enumerate(kf_points):
            color = (100, 255, 200) if i == 0 else (255, 255, 255)
            cv2.circle(depth_viz, pt, 4, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(depth_viz, pt, 3, color, -1, cv2.LINE_AA)

        if self.camera_view.projector:
            current_pt = pose_to_point(self.camera_view.get_current_pose())
            if kf_points:
                cv2.line(depth_viz, kf_points[-1], current_pt, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.circle(depth_viz, current_pt, 6, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(depth_viz, current_pt, 5, (100, 180, 255), -1, cv2.LINE_AA)

        self.lbl_depth_preview.setPixmap(np_bgr_to_qpixmap(depth_viz))

    def _add_keyframe_at_current_pose(self, is_initial=False):
        self.keyframes.append(self.camera_view.get_current_pose())
        self._update_keyframe_ui()
        self._update_depth_trajectory()

    def on_add_keyframe(self):
        self._add_keyframe_at_current_pose()

    def on_undo_keyframe(self):
        if len(self.keyframes) > 1:
            self.keyframes.pop()
            self._update_keyframe_ui()
            self._update_depth_trajectory()

    def on_clear_keyframes(self):
        if self.keyframes:
            first_kf = self.keyframes[0]
            self.keyframes = [first_kf]
            self.camera_view.reset_pose()
            self._update_keyframe_ui()
            self._update_depth_trajectory()

    def _update_keyframe_ui(self):
        num_kf = len(self.keyframes)
        self.lbl_keyframe_info.setText(f"Keyframes: {num_kf}")
        self.btn_undo_keyframe.setEnabled(num_kf > 1)
        self.btn_clear.setEnabled(num_kf > 0)
        self.btn_play.setEnabled(num_kf >= 2)
        self.btn_save.setEnabled(num_kf >= 2)

    def on_play_demo(self):
        if len(self.keyframes) < 2 or self.camera_view.projector is None:
            return
        self.play_timer.stop()
        self.btn_play.setEnabled(False)
        total_frames = self.spn_total_frames.value()
        fps = self.spn_fps.value()
        out_w, out_h = self.spn_width.value(), self.spn_height.value()
        poses = self._interpolate_poses(self.keyframes, total_frames)
        self.preview_frames = []
        for pose in poses:
            warped, _ = self.camera_view.projector.warp_to_new_pose(pose)
            warped = cv2.resize(warped, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            pixmap = np_bgr_to_qpixmap(warped).scaled(
                self.camera_view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.preview_frames.append(pixmap)
        self.preview_index = 0
        self.play_timer.start(int(1000 / fps))

    def _on_play_timer_tick(self):
        if self.preview_index >= len(self.preview_frames):
            self.play_timer.stop()
            self.btn_play.setEnabled(True)
            self.camera_view.update_view()
            return
        self.camera_view.setPixmap(self.preview_frames[self.preview_index])
        self.preview_index += 1

    def on_save_output(self):
        if len(self.keyframes) < 2 or self.camera_view.projector is None:
            return
        base_dir = QFileDialog.getExistingDirectory(self, "Select output directory")
        if not base_dir:
            return
        subdir_name, ok = QInputDialog.getText(self, "Subfolder Name", "Create a subfolder:")
        if not ok or not subdir_name.strip():
            return
        final_dir = os.path.join(base_dir, subdir_name.strip())
        if os.path.exists(final_dir):
            if QMessageBox.question(self, "Folder exists", f"'{subdir_name}' exists. Overwrite?",
                                   QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
                return
        else:
            os.makedirs(final_dir, exist_ok=True)

        try:
            trans_dir = os.path.join(final_dir, "transformations")
            os.makedirs(trans_dir, exist_ok=True)
            
            total_frames = self.spn_total_frames.value()
            fps = self.spn_fps.value()
            out_w, out_h = self.spn_width.value(), self.spn_height.value()
            fit_mode = self.fit_mode_combo.currentText()
            
            def apply_fit(img, interpolation=cv2.INTER_LINEAR):
                if fit_mode == "Center Crop":
                    return resize_then_center_crop(img, out_h, out_w, interpolation)
                return fit_center_pad(img, out_h, out_w, interpolation)
            
            poses = self._interpolate_poses(self.keyframes, total_frames)
            frames, masks = [], []
            
            for i, pose in enumerate(poses):
                warped, mask = self.camera_view.projector.warp_to_new_pose(pose)
                warped = apply_fit(warped, cv2.INTER_LINEAR)
                mask = cv2.cvtColor(apply_fit(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), cv2.INTER_NEAREST), cv2.COLOR_BGR2GRAY)
                frames.append(warped)
                masks.append(mask)
                with open(os.path.join(trans_dir, f"transform_{i:04d}.json"), 'w') as f:
                    json.dump({'frame': i, 'pose': pose.to_dict(), 'matrix': pose.to_matrix().tolist()}, f, indent=2)

            cv2.imwrite(os.path.join(final_dir, "first_frame.png"), frames[0])
            save_video_mp4(frames, os.path.join(final_dir, "motion_signal.mp4"), fps=fps)
            
            kernel_connect = np.ones((3, 3), dtype=np.uint8)
            kernel_open = np.ones((5, 5), dtype=np.uint8)
            mask_frames_bgr = []
            for m in masks:
                m_connected = cv2.dilate(m, kernel_connect, iterations=2)
                m_opened = cv2.morphologyEx(m_connected, cv2.MORPH_OPEN, kernel_open)
                mask_frames_bgr.append(cv2.cvtColor(m_opened, cv2.COLOR_GRAY2BGR))
            save_video_mp4(mask_frames_bgr, os.path.join(final_dir, "mask.mp4"), fps=fps)
            
            with open(os.path.join(final_dir, "prompt.txt"), 'w') as f:
                f.write(self.txt_prompt.toPlainText().strip())

            QMessageBox.information(self, "Success", f"Saved to:\n{final_dir}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed:\n{e}")

    def _interpolate_poses(self, poses: List[CameraPose], num_frames: int) -> List[CameraPose]:
        if len(poses) == 0:
            return []
        if len(poses) == 1:
            return [poses[0].copy() for _ in range(num_frames)]
        result = []
        for i in range(num_frames):
            t = i / (num_frames - 1)
            idx_float = t * (len(poses) - 1)
            idx0 = int(np.floor(idx_float))
            idx1 = min(idx0 + 1, len(poses) - 1)
            alpha = idx_float - idx0
            p0, p1 = poses[idx0], poses[idx1]
            result.append(CameraPose(
                tx=p0.tx * (1 - alpha) + p1.tx * alpha,
                ty=p0.ty * (1 - alpha) + p1.ty * alpha,
                tz=p0.tz * (1 - alpha) + p1.tz * alpha,
                rx=p0.rx * (1 - alpha) + p1.rx * alpha,
                ry=p0.ry * (1 - alpha) + p1.ry * alpha,
                rz=p0.rz * (1 - alpha) + p1.rz * alpha
            ))
        return result


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = CameraControlWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
