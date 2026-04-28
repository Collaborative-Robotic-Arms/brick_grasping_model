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

from std_msgs.msg import Int32
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

from dual_arms_msgs.msg import GraspPoint  
from dual_arms_msgs.srv import GetGrasp
import threading

try:
    from brick_grasping_model.models import SwinGraspNoWidth, ResNetUNetGraspNoWidth
except ImportError:
    from detection_grasping.brick_grasping_model.models import SwinGraspNoWidth, ResNetUNetGraspNoWidth

# =================================================================================
# 1. HELPER FUNCTIONS
# =================================================================================

def normalize_rgb(rgb):
    return rgb - rgb.mean()

def normalize_depth(d):
    return np.clip(d - float(d.mean()), -1.0, 1.0)

def to_torch(arr):
    if arr.ndim == 2:
        return torch.from_numpy(arr[None, ...].astype(np.float32))
    else:
        return torch.from_numpy(arr.transpose(2, 0, 1).astype(np.float32))

def post_process(pos_logits, cos, sin):
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

        self.declare_parameter('use_sim', False)
        self.declare_parameter('static_z_height', 0.712) 
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.latest_grasp = None
        self.latest_grasp_lock = threading.Lock()
        self.grasp_ready_event = threading.Event()
        self.detections:Detection2DArray = None

        # --- Parameters ---
        default_ckpt_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'weights',
            'resnet_gp2_3_BEST.pth'
        )
        self.declare_parameter('ckpt_path', default_ckpt_path)
        self.declare_parameter('arch', 'resnet_unet')
        self.declare_parameter('input_size', 160)
        self.declare_parameter('use_depth', True)
        self.declare_parameter('use_rgb', True)
        self.use_sim = self.get_parameter('use_sim').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        self.static_z = self.get_parameter('static_z_height').value

        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        # self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info') # No longer used (Hardcoded)

        self.declare_parameter('dets_topic', '/yolo/detections')
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('depth_scale', 1.00)
        self.declare_parameter('detection_debug', '/yolo/annotated_image')
        self.declare_parameter('target_topic', '/grasp/target_index')

        # Read Parameters
        self.ckpt_path = self.get_parameter('ckpt_path').value
        self.arch = self.get_parameter('arch').value
        self.input_size = self.get_parameter('input_size').value
        self.use_depth = self.get_parameter('use_depth').value
        self.use_rgb = self.get_parameter('use_rgb').value
        rgb_topic = self.get_parameter('image_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        dets_topic = self.get_parameter('dets_topic').value
        detection_debug = self.get_parameter('detection_debug').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.depth_scale = self.get_parameter('depth_scale').value
        target_topic = self.get_parameter('target_topic').value

        self.camera_info_received = False
        self.k_matrix = None
        self.dist_coeffs = None
        self.new_camera_mtx = None
        self.map1 = None
        self.map2 = None
        self.optimized_intrinsics = None
        
        # =========================================================================
        # --- HARDCODED CALIBRATION DATA (From Detection Node) ---
        # =========================================================================
        # =========================================================================
        # --- CALIBRATION DATA ---
        # =========================================================================
        # if not self.use_sim:
        #     # Camera Matrix (K)
        #     self.k_matrix = np.array([
        #         [607.6493463464219, 0.0, 330.2045740645484],
        #         [0.0, 605.19606629627, 246.36866587909964],
        #         [0.0, 0.0, 1.0]
        #     ], dtype=np.float64)

        #     # Distortion Coefficients (D)
        #     self.dist_coeffs = np.array([
        #         0.030701467019085278, 0.6603616986413218, 
        #         -0.0030859808687108714, -0.005391372766986892, 
        #         -2.547370818352119
        #     ], dtype=np.float64)

        #     # Optimization: Pre-compute the Map
        #     self.img_w, self.img_h = 640, 480 
            
        #     # Alpha=0 crops the image to remove black edges created by undistortion
        #     self.new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(
        #         self.k_matrix, self.dist_coeffs, (self.img_w, self.img_h), 0, (self.img_w, self.img_h)
        #     )
            
        #     self.map1, self.map2 = cv2.initUndistortRectifyMap(
        #         self.k_matrix, self.dist_coeffs, None, self.new_camera_mtx, 
        #         (self.img_w, self.img_h), cv2.CV_32FC1
        #     )
            
        #     # This will be used for 3D reconstruction instead of the raw K matrix
        #     self.optimized_intrinsics = {
        #         'fx': self.new_camera_mtx[0, 0], 'fy': self.new_camera_mtx[1, 1],
        #         'cx': self.new_camera_mtx[0, 2], 'cy': self.new_camera_mtx[1, 2]
        #     }
            
        #     self.camera_info_received = True 
        #     self.get_logger().info("HARDWARE Calibration Loaded & Undistort Maps Computed.")
        # else:
        #     self.get_logger().info("SIMULATION MODE Active: Waiting for /camera_info...")
        # =========================================================================

        # --- Load Model ---
        in_ch = (3 if self.use_rgb else 0) + (1 if self.use_depth else 0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.get_logger().info(f"Building model: {self.arch}, Input Ch: {in_ch}, Device: {self.device}")
        self.model = build_model(self.arch, in_ch, self.input_size, pretrained=False).to(self.device)
        self.model.eval()

        if self.ckpt_path:
            self.get_logger().info(f"Loading weights from: {self.ckpt_path}")
            ck = torch.load(self.ckpt_path, map_location=self.device)
            sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
            self.model.load_state_dict(sd, strict=True)
        else:
            self.get_logger().warn("No checkpoint path provided! Model contains random weights.")

        # --- Sub/Pub ---
        self.bridge = CvBridge()

        self.rgb_sub = self.create_subscription(Image, rgb_topic, self.rgb_callback, qos)
        # Note: We must also undistort the detection debug image if we want to visualize correctly 
        # aligned with the undistorted RGB/Depth we process.
        self.detection_sub = self.create_subscription(Image, detection_debug, self.detection_callback, qos)
        self.depth_sub = self.create_subscription(Image, depth_topic, self.depth_callback, qos)
        self.dets_sub = self.create_subscription(Detection2DArray, dets_topic, self.dets_callback, 10)
        self.cam_info_sub = self.create_subscription(CameraInfo, camera_info_topic, self.cam_info_callback, 10)
        
        # We NO LONGER need camera_info subscription because we hardcoded it
        # self.cam_info_sub = self.create_subscription(CameraInfo, cam_info_topic, self.cam_info_callback, 10)
        
        self.target_sub = self.create_subscription(Int32, target_topic, self.target_index_callback, 10)

        # Outputs
        self.vis_pub = self.create_publisher(Image, '/grasp/result_image', 10)
        self.pose_pub = self.create_publisher(GraspPoint, '/grasp/result', 10)
        self.pub_debug_q = self.create_publisher(Image, 'grasp/debug/quality', 10)

        self.grasp_srv = self.create_service(
            GetGrasp,
            '/grasp/get_grasp_point',
            self.handle_get_grasp
        )

        # Buffers
        self.last_rgb = None
        self.last_rgb_detection = None
        self.last_depth = None
        self.last_header = None
        
        # Logic State
        self.target_brick_idx = None

        self.get_logger().info("Grasp Node Initialized. Waiting for target index...")

    # --- Callbacks ---
    def handle_get_grasp(self, request, response):
        self.get_logger().info(f"Grasp service requested for brick ID: {request.brick_index}")
        self.target_brick_idx = int(request.brick_index)
        graspPoint = self.process_grasp()

        if graspPoint is None:
            self.get_logger().warn("No valid grasp found.")
            response.success = False
            return response

        response.grasp_point = graspPoint
        response.success = True
        return response

    def target_index_callback(self, msg):
        prev_target = self.target_brick_idx
        self.target_brick_idx = msg.data

        self.process_grasp()
        if prev_target != self.target_brick_idx:
            self.get_logger().info(f"New Target Received: Index {self.target_brick_idx}. Starting grasp tracking.")

    def detection_callback(self, msg):
        try:
            raw_det = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # NOTE: The detection node already publishes an UNDISTORTED image on '/yolo/annotated_image'.
            # If 'detection_debug' topic comes from YoloV8Detector's image_pub, it is ALREADY undistorted.
            # So we DO NOT undistort it again here.
            self.last_rgb_detection = raw_det
        except Exception as e:
            self.get_logger().error(f"Detection Image Error: {e}")


    def cam_info_callback(self, msg: CameraInfo):
        if self.camera_info_received:
            return

        self.img_w = msg.width
        self.img_h = msg.height
        self.k_matrix = np.array(msg.k).reshape((3, 3))
        self.dist_coeffs = np.array(msg.d) if len(msg.d) > 0 else np.zeros(5)

        # If distortion coefficients exist (Hardware)
        if np.any(self.dist_coeffs):
            self.new_camera_mtx, _ = cv2.getOptimalNewCameraMatrix(
                self.k_matrix, self.dist_coeffs, (self.img_w, self.img_h), 0, (self.img_w, self.img_h)
            )
            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                self.k_matrix, self.dist_coeffs, None, self.new_camera_mtx, 
                (self.img_w, self.img_h), cv2.CV_32FC1
            )
            self.optimized_intrinsics = {
                'fx': self.new_camera_mtx[0, 0], 'fy': self.new_camera_mtx[1, 1],
                'cx': self.new_camera_mtx[0, 2], 'cy': self.new_camera_mtx[1, 2]
            }
            self.get_logger().info("Hardware calibration loaded & Undistort maps computed.")
        
        # No distortion (Simulation / pre-rectified)
        else:
            self.optimized_intrinsics = {
                'fx': self.k_matrix[0, 0], 'fy': self.k_matrix[1, 1],
                'cx': self.k_matrix[0, 2], 'cy': self.k_matrix[1, 2]
            }
            self.map1 = None
            self.map2 = None
            self.new_camera_mtx = self.k_matrix
            self.get_logger().info("Simulation/Perfect intrinsics loaded. Undistortion bypassed.")

        self.camera_info_received = True


    def rgb_callback(self, msg):
        if not self.camera_info_received:
            return
        try:
            raw_rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.last_header = msg.header
            
            # --- Check Resolution & Recompute Maps if needed ---
            h, w = raw_rgb.shape[:2]
            if w != self.img_w or h != self.img_h:
                self.get_logger().warn(f"Resolution mismatch (Got {w}x{h}, Expected {self.img_w}x{self.img_h})! Recomputing maps...")
                self.img_w, self.img_h = w, h
                self.new_camera_mtx, _ = cv2.getOptimalNewCameraMatrix(
                    self.k_matrix, self.dist_coeffs, (w, h), 0, (w, h)
                )
                self.map1, self.map2 = cv2.initUndistortRectifyMap(
                    self.k_matrix, self.dist_coeffs, None, self.new_camera_mtx, 
                    (w, h), cv2.CV_32FC1
                )
                self.optimized_intrinsics = {
                    'fx': self.new_camera_mtx[0, 0], 'fy': self.new_camera_mtx[1, 1],
                    'cx': self.new_camera_mtx[0, 2], 'cy': self.new_camera_mtx[1, 2]
                }
            
            # --- APPLY UNDISTORTION ---
            if self.use_sim or self.map1 is None:
                self.last_rgb = raw_rgb.copy()
            else:
                self.last_rgb = cv2.remap(raw_rgb, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)            
        except Exception as e:
            self.get_logger().error(f"RGB Error: {e}")

    def depth_callback(self, msg):
        if not self.camera_info_received:
            return
        try:
            raw_d = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            if self.use_sim:
                # Gazebo outputs R_FLOAT32 directly in meters. No scaling needed!
                raw_d = raw_d.astype(np.float32)
            else:
                # Hardware outputs millimeters. Apply the 0.001 scale.
                raw_d = raw_d.astype(np.float32) * self.depth_scale
            # --- Check Resolution (Assuming depth matches RGB size) ---
            h, w = raw_d.shape[:2]
            if self.use_sim or self.map1 is None:
                self.last_depth = raw_d.copy()
            elif (w == self.img_w) and (h == self.img_h):
                 self.last_depth = cv2.remap(raw_d, self.map1, self.map2, interpolation=cv2.INTER_NEAREST)
            else:
                 self.last_depth = raw_d

        except Exception as e:
            self.get_logger().error(f"Depth Error: {e}")

    def dets_callback(self, msg):
        # 1. Checks
        # We check self.optimized_intrinsics instead of self.camera_intrinsics
        if self.last_rgb is None or self.last_depth is None or self.optimized_intrinsics is None:
            return

        if len(msg.detections) == 0:
            return

        self.detections = msg

    def process_grasp(self):

        # 2. SEARCH for the specific ID
        target_det = None
        if self.detections is None:
            self.get_logger().info("No detections yet.")
            return

        if len(self.detections.detections) == 0:
            self.get_logger().info("No detections available.")
            return
        
        for det in self.detections.detections:
            if not det.results:
                continue
            hypothesis = det.results[0].hypothesis
            
            # Match against class_id or tracking_id
            current_id_str = str(hypothesis.class_id)
            target_id_str = str(self.target_brick_idx)
            
            if current_id_str == target_id_str:
                target_det = det
                break
                
            if hasattr(det, 'id'):
                if str(det.id) == target_id_str:
                    target_det = det
                    break
        
        if target_det is None:
            self.get_logger().info(f"Target brick ID {self.target_brick_idx} not found in detections.")
            return

        # 3. Get Coords
        # IMPORTANT: The Yolo Node sends coordinates based on the UNDISTORTED image.
        # Since we are also undistorting 'self.last_rgb' and 'self.last_depth' here, 
        # these coordinates match perfectly.
        cx_det = float(target_det.bbox.center.position.x)
        cy_det = float(target_det.bbox.center.position.y)
        w_det = float(target_det.bbox.size_x)
        h_det = float(target_det.bbox.size_y)

        # 4. Crop
        H, W = self.last_rgb.shape[:2]
        margin = 1.2
        side = int(max(w_det, h_det) * margin)
        side = max(32, min(side, min(W, H)))

        x_center = int(cx_det)
        y_center = int(cy_det)
        
        x1 = max(0, x_center - side // 2)
        y1 = max(0, y_center - side // 2)
        x2 = min(W, x1 + side)
        y2 = min(H, y1 + side)
        
        if (x2 - x1) < side: x1 = max(0, x2 - side)
        if (y2 - y1) < side: y1 = max(0, y2 - side)

        rgb_crop = self.last_rgb[y1:y2, x1:x2].copy()
        depth_crop = self.last_depth[y1:y2, x1:x2].copy()

        if rgb_crop.size == 0 or depth_crop.size == 0:
            return

        # 5. Preprocess
        rgb_in = cv2.resize(rgb_crop, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        depth_in = cv2.resize(depth_crop, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)

        rgb_norm = normalize_rgb(cv2.cvtColor(rgb_in, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
        depth_norm = normalize_depth(depth_in)

        tensors = []
        if self.use_depth:
            tensors.append(to_torch(depth_norm))
        if self.use_rgb:
            tensors.append(to_torch(rgb_norm))
        
        x_tensor = torch.cat(tensors, dim=0)[None, ...].to(self.device)

        # 6. Inference
        with torch.no_grad():
            pos_logits, cos, sin = self.model(x_tensor)
            q_map, ang_map = post_process(pos_logits, cos, sin)

        h_q, w_q = q_map.shape
        y_grid, x_grid = np.meshgrid(np.arange(h_q), np.arange(w_q), indexing='ij')
        
        center_y, center_x = h_q / 2.0, w_q / 2.0
        sigma = h_q / 4.0  # Adjust this if you want it more/less strict
        
        gaussian_mask = np.exp(-((x_grid - center_x)**2 + (y_grid - center_y)**2) / (2 * sigma**2))
        
        # Apply the mask to the AI's confidence map
        q_map = q_map * gaussian_mask

        # 7. Find Best Grasp
        gy, gx = np.unravel_index(np.argmax(q_map), q_map.shape)
        best_q = q_map[gy, gx]
        best_ang = ang_map[gy, gx]

        q_vis = (q_map * 255).astype(np.uint8)
        self.pub_debug_q.publish(self.bridge.cv2_to_imgmsg(cv2.applyColorMap(q_vis, cv2.COLORMAP_JET), "bgr8"))

        if best_q < 0.01:
            self.get_logger().info("No high-quality grasp found.")
            return

        # 8. Back to Full Image
        scale_x = (x2 - x1) / float(self.input_size)
        scale_y = (y2 - y1) / float(self.input_size)

        cx_crop = gx * scale_x
        cy_crop = gy * scale_y

        cx_full = int(x1 + cx_crop)
        cy_full = int(y1 + cy_crop)

        # 9. Get Z
        d_val = depth_crop[int(cy_crop), int(cx_crop)]
        if d_val <= 0 or np.isnan(d_val):
            patch = depth_crop[max(0, int(cy_crop)-2):int(cy_crop)+3, 
                               max(0, int(cx_crop)-2):int(cx_crop)+3]
            valid_depths = patch[patch > 0]
            if len(valid_depths) > 0:
                d_val = np.median(valid_depths)
            else:
                d_val = 0.0

        if d_val <= 0:
            self.get_logger().info("depth is a negative number.")
            return

        # 10. Compute 3D Pose
        # USE OPTIMIZED INTRINSICS (Because cx_full/cy_full come from the undistorted image)
        fx, fy = self.optimized_intrinsics['fx'], self.optimized_intrinsics['fy']
        cx, cy = self.optimized_intrinsics['cx'], self.optimized_intrinsics['cy']
        
        self.get_logger().info(f"Using Optimized Intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}, depth map={d_val:.3f}")
        
        # UNIFIED LOGIC: Both Sim and Hardware use the static parameter
        fixed_z = float(self.static_z)
        
        X = (cx_full - cx) * fixed_z / fx
        Y = (cy_full - cy) * fixed_z / fy
        Z = fixed_z

        # 11. Publish Custom BrickGrasp Message
        grasp_msg = GraspPoint()
        grasp_msg.header = self.last_header
        grasp_msg.header.frame_id = self.camera_frame
        
        grasp_msg.brick_id = self.target_brick_idx
        grasp_msg.quality = float(best_q)
  

        grasp_msg.pose.position.x = float(X)
        grasp_msg.pose.position.y = float(Y)
        grasp_msg.pose.position.z = float(Z)
        
        # half_theta = -best_ang / 2.0

        # --- FIX: NEGATE ANGLE ---
        # The visual model sees clockwise as positive (Image Space).
        # The robot frame sees counter-clockwise as positive (Right-Hand Rule).
        # We must flip the sign to match them.
        if best_ang > 0:
            corrected_angle = -(best_ang - math.pi)
        else:
            corrected_angle = -(best_ang + math.pi)
        # corrected_angle = best_ang
        log = corrected_angle*180/3.14
        self.get_logger().info(f"Best Angle (Degree): {log}")
        
        # OPTIONAL: Handle the +/- 90 degree gripper symmetry wrap-around
        # This keeps the gripper from spinning 180 unnecessarily
        # corrected_angle = -best_ang + (math.pi / 2)
        # Create Quaternion from corrected_angle
        half_theta = corrected_angle / 2.0
        grasp_msg.pose.orientation.y = 0.0
        grasp_msg.pose.orientation.x = 0.0
        grasp_msg.pose.orientation.z = math.sin(half_theta)
        grasp_msg.pose.orientation.w = math.cos(half_theta)

        self.pose_pub.publish(grasp_msg)
            
        # 12. Visualize
        # Use last_rgb (which is now UNDISTORTED) for visualization to match the calculation
        if self.last_rgb is not None:
            vis_img = self.last_rgb.copy()
            corners = grasp_corners(cx_full, cy_full, best_ang, length=50, width=20)
            cv2.polylines(vis_img, [corners], isClosed=True, color=(255, 255, 0), thickness=2)
            cv2.circle(vis_img, (cx_full, cy_full), 4, (0, 0, 255), -1)

            self.vis_pub.publish(self.bridge.cv2_to_imgmsg(vis_img, "bgr8"))
        
        self.get_logger().info(f"Published BrickGrasp: ID={self.target_brick_idx}, Q={best_q:.2f} at [{X:.2f}, {Y:.2f}, {Z:.2f}, Orientation z={grasp_msg.pose.orientation.z:.2f}, w={grasp_msg.pose.orientation.w:.2f}, angle={best_ang}]")

        return grasp_msg
    
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