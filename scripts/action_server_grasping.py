#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient  # Import ActionClient
import threading
from rclpy.duration import Duration
from rclpy.time import Time

# --- Standard ROS Messages ---
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32

# --- TF2 Imports ---
from tf2_ros import Buffer, TransformListener, TransformException
import tf2_geometry_msgs 

# --- Custom Messages & Actions ---
from dual_arms_msgs.msg import GraspPoint 
from dual_arms_msgs.action import ExecuteTask # Import the Action Definition

class GraspRelay(Node):
    def __init__(self):
        super().__init__('grasp_relay_node')

        # --- TF2 SETUP ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- 1. Subscriber (Input) ---
        self.sub = self.create_subscription(
            GraspPoint, 
            '/grasp/result', 
            self.listener_callback, 
            10
        )
        
        # --- 2. Action Client (Replaces old publishers) ---
        # Assuming the action server topic is named 'execute_task'
        self._action_client = ActionClient(self, ExecuteTask, 'abb_control')

        # --- 3. Publisher (User Input) ---
        # Keeps sending the target index as requested
        self.user_int_pub = self.create_publisher(Int32, '/grasp/target_index', 10)
        
        self.get_logger().info('--- GRASP RELAY ACTION CLIENT STARTED ---')
        self.get_logger().info('--- WAITING FOR USER INPUT (Enter an Integer) ---')

        # --- 4. Start Input Thread ---
        self.input_thread = threading.Thread(target=self.get_user_input)
        self.input_thread.daemon = True 
        self.input_thread.start()

    def listener_callback(self, msg):
        """
        Runs whenever a message arrives on /grasp/result.
        Transforms the pose and triggers the Action Client.
        """
        # A. Create the Source PoseStamped (in camera frame)
        source_pose = PoseStamped()
        source_pose.header.frame_id = 'camera_color_optical_frame' 
        source_pose.header.stamp = Time(seconds=0).to_msg() 
        source_pose.pose = msg.pose

        transformed_pose = None

        # --- B. TF TRANSFORMATION BLOCK ---
        try:
            # Transform source_pose -> target_pose in 'base_link'
            transformed_pose = self.tf_buffer.transform(
                source_pose, 
                'base_link', 
                timeout=Duration(seconds=1.0)
            )
            
            # Maintain specific orientation components from the original logic
            transformed_pose.pose.orientation.z = msg.pose.orientation.z
            transformed_pose.pose.orientation.w = msg.pose.orientation.w

            self.get_logger().info(f"Relayed Grasp ID {msg.brick_id}. Sending Action Goal...")
            
            # --- C. SEND ACTION GOAL ---
            self.send_task_goal(transformed_pose.pose)

        except TransformException as ex:
            self.get_logger().error(f"Could not transform camera_color_optical_frame to base_link: {ex}")
            # Do NOT send the goal if transform fails to avoid robot errors

    def send_task_goal(self, target_pose):
        """
        Creates and sends the goal to the ExecuteTask action server.
        """
        # Check if server is ready
        if not self._action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error("Action server 'execute_task' not available!")
            return

        goal_msg = ExecuteTask.Goal()
        goal_msg.task_type = "PICK" # Or "MOVE_TO_POSE" depending on your logic
        goal_msg.target_arm = "ABB"
        QW = target_pose.orientation.w
        QZ = target_pose.orientation.z
        
        goal_msg.target_pose = target_pose
        goal_msg.target_pose.orientation.x = -QZ
        goal_msg.target_pose.orientation.y = -QW
        goal_msg.target_pose.orientation.z = 0.0
        goal_msg.target_pose.orientation.w = 0.0
        goal_msg.target_pose.position.z = 0.26
        self.get_logger().info(f"Target Pose: {target_pose.position.x}, {target_pose.position.y}, {target_pose.position.z}")
        self.get_logger().info(f"Sending goal for Task: {goal_msg.task_type}")

        # Send goal asynchronously
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg, 
            feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """
        Callback when the server accepts or rejects the goal request.
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        """
        Callback when the action execution is finished.
        """
        result_wrapper = future.result()
        result = result_wrapper.result
        
        if result.success:
            self.get_logger().info(f'Action Complete. Success: {result.success}')
        else:
            self.get_logger().warn(f'Action Failed. Error: {result.error_message}')

    def feedback_callback(self, feedback_msg):
        """
        Callback for continuous feedback from the server.
        """
        feedback = feedback_msg.feedback
        # Uncomment below to see live updates in the terminal
        self.get_logger().info(f'Feedback: {feedback.current_status} - {feedback.progress:.2f}')

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