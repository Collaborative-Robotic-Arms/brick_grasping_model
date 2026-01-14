#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import threading  # Required for background input

# --- Standard ROS Messages ---
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Int32  # Added Int32

from rclpy.qos import QoSProfile, DurabilityPolicy
# --- Your Custom Message ---
from dual_arms_msgs.msg import GraspPoint 

class GraspRelay(Node):
    def __init__(self):
        super().__init__('grasp_relay_node')

        # --- 1. Subscriber (Input) ---
        self.sub = self.create_subscription(
            GraspPoint, 
            '/grasp/result', 
            self.listener_callback, 
            10
        )
        # --- KEY CONFIGURATION: QoS Setup ---
        # 1. Create a generic QoS Profile with depth of 1 (keep only latest)
        qos_profile = QoSProfile(depth=1)

        # 2. Set Durability to TRANSIENT_LOCAL
        # This is the Python equivalent of .transient_local() in C++
        # Without this, it defaults to VOLATILE and won't connect to your listener.
        qos_profile.durability = DurabilityPolicy.TRANSIENT_LOCAL
        # --- 2. Publishers (Existing Outputs) ---
        self.pose_pub = self.create_publisher(PoseStamped, '/target_pose_abb',qos_profile)
        self.flag_pub = self.create_publisher(Bool, '/gripper_command', 10)

        # --- 3. New Publisher (User Input) ---
        self.user_int_pub = self.create_publisher(Int32, '/grasp/target_index', 10)
        
        self.get_logger().info('--- GRASP RELAY NODE STARTED ---')
        self.get_logger().info('--- WAITING FOR USER INPUT (Enter an Integer) ---')

        # --- 4. Start Input Thread ---
        # We start a separate thread so input() doesn't block ROS callbacks
        self.input_thread = threading.Thread(target=self.get_user_input)
        self.input_thread.daemon = True # Thread dies when the main node dies
        self.input_thread.start()

    def listener_callback(self, msg):
        """
        Runs in the MAIN thread whenever a message arrives.
        """
        # A. Create the Output PoseStamped
        out_msg = PoseStamped()
        
        out_msg.header.stamp = self.get_clock().now().to_msg()
        out_msg.pose = msg.pose
        out_msg.header.frame_id = 'world'
        # B. Publish Pose and Flag
        self.pose_pub.publish(out_msg)

        flag_msg = Bool()
        flag_msg.data = True
        self.flag_pub.publish(flag_msg)

        self.get_logger().info(f"Background: Relayed Grasp ID {msg.brick_id}")

    def get_user_input(self):
        """
        Runs in a SEPARATE thread. Handles blocking input().
        """
        while rclpy.ok():
            try:
                # This line blocks execution, but only for this thread!
                # The listener_callback above continues to run freely.
                user_str = input() 
                
                # Convert to integer
                user_val = int(user_str)
                
                # Publish
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
        # Spin the node in the main thread to handle ROS callbacks
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()