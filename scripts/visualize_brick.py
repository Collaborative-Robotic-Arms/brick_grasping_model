from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose
import math

def visualize_block(brick_msg, frame_id="camera_color_optical_frame"):
    """
    Args:
        brick_msg: Your custom Brick message (pose.orientation.z has Yaw in radians)
        frame_id: Frame for RViz (usually camera frame or base_link)
    """
    markers = MarkerArray()
    
    # Extract Type and Pose
    b_type = brick_msg.type
    yaw = brick_msg.pose.orientation.z  # Assuming you stored RAW YAW here
    
    px = brick_msg.pose.position.x
    py = brick_msg.pose.position.y
    pz = brick_msg.pose.position.z

    # Standard "I" Dimensions (meters)
    # You requested 60cm x 30cm x 20cm
    I_len = 0.60
    I_wid = 0.30
    I_hgt = 0.20

    # Helper to create a single cube part
    def make_cube(id_suffix, dx_local, dy_local, dim_x, dim_y, color_rgb):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = brick_msg.header.stamp
        marker.ns = f"brick_{brick_msg.id}"
        marker.id = int(brick_msg.id) * 100 + id_suffix
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        
        # 1. Rotate local offsets (dx, dy) by the Brick's Yaw
        # x_global = x_center + (dx * cos(yaw) - dy * sin(yaw))
        # y_global = y_center + (dx * sin(yaw) + dy * cos(yaw))
        rot_x = dx_local * math.cos(yaw) - dy_local * math.sin(yaw)
        rot_y = dx_local * math.sin(yaw) + dy_local * math.cos(yaw)
        
        marker.pose.position.x = px + rot_x
        marker.pose.position.y = py + rot_y
        marker.pose.position.z = pz
        
        # 2. Orientation (Yaw to Quaternion)
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = math.sin(yaw / 2.0)
        marker.pose.orientation.w = math.cos(yaw / 2.0)
        
        marker.scale.x = dim_x
        marker.scale.y = dim_y
        marker.scale.z = I_hgt
        
        marker.color.r = color_rgb[0]
        marker.color.g = color_rgb[1]
        marker.color.b = color_rgb[2]
        marker.color.a = 0.8  # Transparency
        
        # Lifetime (so they disappear if detection stops)
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = int(5e8) # 0.5 seconds
        
        return marker

    # === LOGIC PER TYPE ===

    if b_type == 0:  # I_BRICK (Blue)
        # Just one block
        m = make_cube(0, 0, 0, I_len, I_wid, (0.0, 0.0, 1.0))
        markers.markers.append(m)

    elif b_type == 1:  # L_BRICK (Red)
        # Two intersecting blocks
        # Vertical leg
        m1 = make_cube(0, 0, 0, I_len, I_wid, (1.0, 0.0, 0.0))
        # Horizontal leg (sticking out side)
        offset = (I_len/2.0 + I_wid/2.0) / 2.0  # Rough offset
        m2 = make_cube(1, I_len/2.0 - I_wid/2.0, I_len/2.0, I_wid, I_len*0.6, (1.0, 0.0, 0.0))
        markers.markers.append(m1)
        markers.markers.append(m2)

    elif b_type == 2:  # T_BRICK (Green)
        # Top Bar (1.5x Length)
        m1 = make_cube(0, 0, 0, I_len * 1.5, I_wid, (0.0, 1.0, 0.0))
        # Stem (0.5x Length, 90 deg)
        stem_len = I_len * 0.5
        offset_y = (I_wid + stem_len) / 2.0
        # To make it perpendicular, we swap Length/Width dimensions in scale
        # But since we only rotate position, we just make a box that is 'wide' along Y local
        m2 = make_cube(1, 0, -offset_y, I_wid, stem_len, (0.0, 1.0, 0.0))
        markers.markers.append(m1)
        markers.markers.append(m2)

    elif b_type == 3:  # Z_BRICK (Yellow, optional)
        m1 = make_cube(0, I_len/4.0, 0, I_len, I_wid, (1.0, 1.0, 0.0))
        m2 = make_cube(1, -I_len/4.0, I_wid, I_len, I_wid, (1.0, 1.0, 0.0))
        markers.markers.append(m1)
        markers.markers.append(m2)
    
    # Fallback (Gray)
    else:
        m = make_cube(0, 0, 0, 0.2, 0.2, (0.5, 0.5, 0.5))
        markers.markers.append(m)

    return markers