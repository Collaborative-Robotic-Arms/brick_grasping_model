#!/usr/bin/env python3
import os
import sys
import math
import numpy as np
import cv2
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

# --- IMPORTS FROM YOUR PACKAGE ---
# Make sure your PYTHONPATH is set correctly as discussed previously
# or adjust these imports to match your folder structure
try:
    from brick_grasping_model.models import SwinGraspNoWidth, ResNetUNetGraspNoWidth
except ImportError:
    # Fallback if running from a different root
    from detection_grasping.brick_grasping_model.models import SwinGraspNoWidth, ResNetUNetGraspNoWidth

# =================================================================================
# 1. HELPER FUNCTIONS (Adapted from offline_inference.py)
# =================================================================================

def normalize_rgb(rgb):
    # rgb: float32 [0..1]
    return rgb - rgb.mean()

def normalize_depth(d):
    # d: float32 (meters or similar scale)
    return np.clip(d - float(d.mean()), -1.0, 1.0)

def to_torch(arr):
    # arr: H,W or H,W,C -> Torch Tensor 1,C,H,W
    if arr.ndim == 2:
        return torch.from_numpy(arr[None, ...].astype(np.float32))
    else:
        return torch.from_numpy(arr.transpose(2, 0, 1).astype(np.float32))

def post_process(pos_logits, cos, sin):
    """
    Converts raw model output to Quality (0-1) and Angle (rad).
    """
    pos = torch.sigmoid(pos_logits)
    ang = 0.5 * torch.atan2(sin, cos)
    q = pos.squeeze().detach().cpu().numpy()
    ang = ang.squeeze().detach().cpu().numpy()
    return q, ang

def build_model(arch, in_ch, img_size, pretrained=False):
    arch = arch.lower()
    if arch == "swin_tiny":
        return SwinGraspNoWidth(
            in_channels=in_ch,
            model_name="swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            embed_dim=256,
            img_size=img_size
        )
    elif arch == "resnet_unet":
        return ResNetUNetGraspNoWidth(in_channels=in_ch, pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported arch: {arch}")

def grasp_corners(cx, cy, theta, length=50, width=20):
    hl = length / 2.0
    hw = width / 2.0
    dx = np.cos(theta)
    dy = np.sin(theta)
    px = -np.sin(theta)
    py = np.cos(theta)

    p1 = (cx - hl*dx - hw*px, cy - hl*dy - hw*py)
    p2 = (cx + hl*dx - hw*px, cy + hl*dy - hw*py)
    p3 = (cx + hl*dx + hw*px, cy + hl*dy + hw*py)
    p4 = (cx - hl*dx + hw*px, cy - hl*dy + hw*py)
    return np.array([p1, p2, p3, p4], dtype=np.int32)

# =================================================================================
# 2. ROS NODE
# =================================================================================

class RosGraspNode(Node):
    def __init__(self):
        super().__init__('ros_grasp_node')

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # --- Parameters ---
        self.declare_parameter('ckpt_path', '/home/mohamed/gp_ws/src/detection_grasping/brick_grasping_model/weights/BEST.pth')       # Path to BEST.pth
        self.declare_parameter('arch', 'swin_tiny')   # swin_tiny or resnet_unet
        self.declare_parameter('input_size', 160)     # Model input size (e.g. 160)
        self.declare_parameter('use_depth', True)
        self.declare_parameter('use_rgb', True)
        self.declare_parameter('rgb_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('dets_topic', '/yolo/detections')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('depth_scale', 0.001)  # If depth is mm, convert to meters

        # Read Parameters
        self.ckpt_path = self.get_parameter('ckpt_path').value
        self.arch = self.get_parameter('arch').value
        self.input_size = self.get_parameter('input_size').value
        self.use_depth = self.get_parameter('use_depth').value
        self.use_rgb = self.get_parameter('use_rgb').value
        rgb_topic = self.get_parameter('rgb_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        dets_topic = self.get_parameter('dets_topic').value
        cam_info_topic = self.get_parameter('camera_info_topic').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.depth_scale = self.get_parameter('depth_scale').value

        # --- Load Model ---
        in_ch = (3 if self.use_rgb else 0) + (1 if self.use_depth else 0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.get_logger().info(f"Building model: {self.arch}, Input Ch: {in_ch}, Device: {self.device}")
        self.model = build_model(self.arch, in_ch, self.input_size, pretrained=False).to(self.device)
        self.model.eval()

        if self.ckpt_path:
            self.get_logger().info(f"Loading weights from: {self.ckpt_path}")
            ck = torch.load(self.ckpt_path, map_location=self.device)
            # Handle if checkpoint is a dict or just state_dict
            sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
            self.model.load_state_dict(sd, strict=True)
        else:
            self.get_logger().warn("No checkpoint path provided! Model contains random weights.")

        # --- Sub/Pub ---
        self.bridge = CvBridge()

        self.rgb_sub = self.create_subscription(Image, rgb_topic, self.rgb_callback, qos)
        self.depth_sub = self.create_subscription(Image, depth_topic, self.depth_callback, qos)
        self.dets_sub = self.create_subscription(Detection2DArray, dets_topic, self.dets_callback, 10)
        self.cam_info_sub = self.create_subscription(CameraInfo, cam_info_topic, self.cam_info_callback, 10)

        # Outputs
        self.vis_pub = self.create_publisher(Image, '/grasp/result_image', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/grasp/pose', 10)
        
        # Debug Publishers (Heatmaps)
        self.pub_debug_q = self.create_publisher(Image, 'grasp/debug/quality', 10)

        # Buffers
        self.last_rgb = None
        self.last_depth = None
        self.last_header = None
        self.camera_intrinsics = None # [fx, fy, cx, cy]

        self.get_logger().info("Grasp Node Initialized.")

    # --- Callbacks ---

    def cam_info_callback(self, msg):
        # K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        self.camera_intrinsics = {
            'fx': msg.k[0],
            'fy': msg.k[4],
            'cx': msg.k[2],
            'cy': msg.k[5]
        }

    def rgb_callback(self, msg):
        try:
            self.last_rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.last_header = msg.header
        except Exception as e:
            self.get_logger().error(f"RGB Error: {e}")

    def depth_callback(self, msg):
        try:
            d = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            # Convert to float32 meters immediately
            self.last_depth = d.astype(np.float32) * self.depth_scale
        except Exception as e:
            self.get_logger().error(f"Depth Error: {e}")

    def dets_callback(self, msg):
        """
        Main Trigger: 
        1. Takes YOLO BBox
        2. Crops RGB/Depth
        3. Runs Inference
        4. Publishes Pose
        """
        if self.last_rgb is None or self.last_depth is None or self.camera_intrinsics is None:
            return

        if len(msg.detections) == 0:
            self.get_logger().warn("No detections received.")
            return

        # 1. Pick Best Detection (highest score)
        best_det = max(msg.detections, key=lambda d: d.results[0].hypothesis.score if d.results else 0)
        score = best_det.results[0].hypothesis.score
        
        # 2. Get BBox Coordinates
        cx_det = float(best_det.bbox.center.position.x)
        cy_det = float(best_det.bbox.center.position.y)
        w_det = float(best_det.bbox.size_x)
        h_det = float(best_det.bbox.size_y)

        # 3. Calculate Crop with Margin
        H, W = self.last_rgb.shape[:2]
        margin = 1.2
        side = int(max(w_det, h_det) * margin)
        # Ensure crop is not tiny, nor larger than image
        side = max(32, min(side, min(W, H)))

        x_center = int(cx_det)
        y_center = int(cy_det)
        
        # Calculate boundaries handling image edges
        x1 = max(0, x_center - side // 2)
        y1 = max(0, y_center - side // 2)
        x2 = min(W, x1 + side)
        y2 = min(H, y1 + side)
        
        # Adjust if we hit the right/bottom edge
        if (x2 - x1) < side: x1 = max(0, x2 - side)
        if (y2 - y1) < side: y1 = max(0, y2 - side)

        # Extract Crops
        rgb_crop = self.last_rgb[y1:y2, x1:x2].copy()
        depth_crop = self.last_depth[y1:y2, x1:x2].copy()

        if rgb_crop.size == 0 or depth_crop.size == 0:
            return

        # 4. Preprocessing for Model
        # Resize to model input size (e.g. 160x160)
        rgb_in = cv2.resize(rgb_crop, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        depth_in = cv2.resize(depth_crop, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)

        # Normalize & Convert to Tensor
        # Note: Model expects RGB in [0..1] range before mean-subtraction
        rgb_norm = normalize_rgb(cv2.cvtColor(rgb_in, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
        depth_norm = normalize_depth(depth_in)

        tensors = []
        if self.use_depth:
            tensors.append(to_torch(depth_norm))
        if self.use_rgb:
            tensors.append(to_torch(rgb_norm))
        
        x_tensor = torch.cat(tensors, dim=0)[None, ...].to(self.device) # [1, C, H, W]

        # 5. Inference
        with torch.no_grad():
            pos_logits, cos, sin = self.model(x_tensor)
            q_map, ang_map = post_process(pos_logits, cos, sin)

        # 6. Find Best Grasp Pixel in the 160x160 Map
        # Simple ArgMax
        gy, gx = np.unravel_index(np.argmax(q_map), q_map.shape)
        best_q = q_map[gy, gx]
        best_ang = ang_map[gy, gx]

        # Publish Debug Heatmap
        q_vis = (q_map * 255).astype(np.uint8)
        self.pub_debug_q.publish(self.bridge.cv2_to_imgmsg(cv2.applyColorMap(q_vis, cv2.COLORMAP_JET), "bgr8"))

        if best_q < 0.01: # Threshold
            self.get_logger().warn(f"Skipping: Low Quality Score: {best_q:.3f}")
            return

        # 7. Convert Coordinate Back to Full Image
        # Scale factors from 160 -> Crop Size
        scale_x = (x2 - x1) / float(self.input_size)
        scale_y = (y2 - y1) / float(self.input_size)

        cx_crop = gx * scale_x
        cy_crop = gy * scale_y

        cx_full = int(x1 + cx_crop)
        cy_full = int(y1 + cy_crop)

        # 8. Get 3D Depth (Z)
        # Use a small window median to avoid noisy dead pixels
        d_val = depth_crop[int(cy_crop), int(cx_crop)]
        if d_val <= 0 or np.isnan(d_val):
            # Try 5x5 window around point
            patch = depth_crop[max(0, int(cy_crop)-2):int(cy_crop)+3, 
                               max(0, int(cx_crop)-2):int(cx_crop)+3]
            valid_depths = patch[patch > 0]
            if len(valid_depths) > 0:
                d_val = np.median(valid_depths)
            else:
                d_val = 0.0 # Failed to get depth

        if d_val <= 0:
            self.get_logger().warn("Invalid Depth at grasp point")
            return

        # 9. Compute 3D Pose (Pinhole Model)
        fx, fy = self.camera_intrinsics['fx'], self.camera_intrinsics['fy']
        cx, cy = self.camera_intrinsics['cx'], self.camera_intrinsics['cy']

        X = (cx_full - cx) * d_val / fx
        Y = (cy_full - cy) * d_val / fy
        Z = d_val

        # 10. Publish Pose
        pose_msg = PoseStamped()
        pose_msg.header = self.last_header
        pose_msg.header.frame_id = self.camera_frame
        pose_msg.pose.position.x = float(X)
        pose_msg.pose.position.y = float(Y)
        pose_msg.pose.position.z = float(Z)
        
        # Convert 2D Theta to Quaternion (Rotation around Z-axis)
        # Assuming gripper frame: X-axis is grasp direction? 
        # Usually for top-down: Roll=0, Pitch=Pi, Yaw=Theta
        # Here we just set orientation z/w as 2D rotation for simplicity,
        # but for a real arm, you likely need a full orientation matrix.
        half_theta = best_ang / 2.0
        pose_msg.pose.orientation.z = math.sin(half_theta)
        pose_msg.pose.orientation.w = math.cos(half_theta)

        self.pose_pub.publish(pose_msg)

        # 11. Visualize and Publish Image
        vis_img = self.last_rgb.copy()
        
        # Draw Crop Box
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Draw Grasp
        corners = grasp_corners(cx_full, cy_full, best_ang, length=50, width=20)
        cv2.polylines(vis_img, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.circle(vis_img, (cx_full, cy_full), 4, (0, 0, 255), -1)
        
        cv2.putText(vis_img, f"Q:{best_q:.2f} Z:{Z:.2f}m", (cx_full+10, cy_full), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        self.vis_pub.publish(self.bridge.cv2_to_imgmsg(vis_img, "bgr8"))
        self.get_logger().info(f"Published Grasp at: X={X:.3f}, Y={Y:.3f}, Z={Z:.3f}, Q={best_q:.2f}")


def main(args=None):
    rclpy.init(args=args)
    node = RosGraspNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()