#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import threading
from rclpy.duration import Duration
from rclpy.time import Time

# --- Standard ROS Messages ---
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Int32
from rclpy.qos import QoSProfile, DurabilityPolicy

# --- TF2 Imports (Required for Transformation) ---
from tf2_ros import Buffer, TransformListener, TransformException
import tf2_geometry_msgs # Imports necessary tools to transform PoseStamped

# --- Your Custom Message ---
from dual_arms_msgs.msg import GraspPoint 

class GraspRelay(Node):
    def __init__(self):
        super().__init__('grasp_relay_node')

        # --- TF2 SETUP ---
        # 1. Create a buffer to store transform data
        self.tf_buffer = Buffer()
        # 2. Create a listener to fill the buffer from /tf topics
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- 1. Subscriber (Input) ---
        self.sub = self.create_subscription(
            GraspPoint, 
            '/grasp/result', 
            self.listener_callback, 
            10
        )
        
        # --- KEY CONFIGURATION: QoS Setup ---
        qos_profile = QoSProfile(depth=1)
        qos_profile.durability = DurabilityPolicy.TRANSIENT_LOCAL
        
        # --- 2. Publishers (Existing Outputs) ---
        self.pose_pub = self.create_publisher(PoseStamped, '/target_pose_abb', qos_profile)
        self.flag_pub = self.create_publisher(Bool, '/gripper_command', 10)

        # --- 3. New Publisher (User Input) ---
        self.user_int_pub = self.create_publisher(Int32, '/grasp/target_index', 10)
        
        self.get_logger().info('--- GRASP RELAY NODE STARTED ---')
        self.get_logger().info('--- WAITING FOR USER INPUT (Enter an Integer) ---')

        # --- 4. Start Input Thread ---
        self.input_thread = threading.Thread(target=self.get_user_input)
        self.input_thread.daemon = True 
        self.input_thread.start()

    def listener_callback(self, msg):
        """
        Runs in the MAIN thread whenever a message arrives.
        """
        # A. Create the Source PoseStamped (in camera frame)
        source_pose = PoseStamped()
        
        # 1. Set the frame explicitly as requested
        source_pose.header.frame_id = 'camera_color_optical_frame' 
        
        # 2. Use Time(seconds=0) to request the "latest available" transform.
        # If we use "now()", it often fails because TF data is slightly delayed.
        source_pose.header.stamp = Time(seconds=0).to_msg() 
        
        source_pose.pose = msg.pose

        # --- B. TF TRANSFORMATION BLOCK ---
        try:
            # Check if the transform exists before trying (optional, but good for debugging)
            # This will transform source_pose -> target_pose in 'base_link'
            # timeout=Duration(seconds=1.0) waits up to 1s for the transform to arrive
            transformed_pose = self.tf_buffer.transform(
                source_pose, 
                'camera_color_optical_frame', 
                timeout=Duration(seconds=1.0)
            )
            transformed_pose.pose.orientation.z = msg.pose.orientation.z
            transformed_pose.pose.orientation.w = msg.pose.orientation.w
            # Publish the NEW transformed pose
            self.pose_pub.publish(transformed_pose)
            # self.get_logger().info(f"Success: Transformed pose to {transformed_pose.header.frame_id}")

            # Publish Flag
            flag_msg = Bool()
            flag_msg.data = True
            self.flag_pub.publish(flag_msg)

            self.get_logger().info(f"Background: Relayed Grasp ID {msg.brick_id}")

        except TransformException as ex:
            # This block catches errors if the TF tree is broken or frames don't exist
            self.get_logger().error(f"Could not transform camera to base_link: {ex}, publishing anyways...")
             # Publish the NEW transformed pose
            self.pose_pub.publish(transformed_pose)
            # self.get_logger().info(f"Success: Transformed pose to {transformed_pose.header.frame_id}")

            # Publish Flag
            flag_msg = Bool()
            flag_msg.data = True
            self.flag_pub.publish(flag_msg)
            
            return

    def get_user_input(self):
        """
        Runs in a SEPARATE thread. Handles blocking input().
        """
        while rclpy.ok():
            try:
                user_str = input() 
                user_val = int(user_str)
                
                msg = Int32()
                msg.data = user_val
                self.user_int_pub.publish(msg)
                
                print(f"   [User Input] Published integer: {user_val}")
                
            except ValueError:
                print("   [Error] Please enter a valid integer.")
            except EOFError:
                break

def main(args=None):
    rclpy.init(args=args)
    node = GraspRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()