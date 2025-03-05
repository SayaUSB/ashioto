import rclpy
import numpy as np
import gymnasium as gym
from rclpy.node import Node
from openvino.runtime import Core
from geometry_msgs.msg import Point
from std_msgs.msg import String
from gymnasium.envs.registration import register

class FootstepsPlanner(Node):
    def __init__(self):
        super().__init__("footstep_planner")

        # Initialize OpenVINO model
        self.core = Core()
        self.model = self.core.read_model("openvino_model/footsteps_planning_right.xml")
        self.compiled_model = self.core.compile_model(self.model, "CPU")
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

        # Register environment
        register(
            id="footsteps-planning-right-v0",
            entry_point="environment.envs:FootstepsPlanningRightEnv",
            max_episode_steps=1000,
        )
        self.env = gym.make("footsteps-planning-right-v0")

        # ROS2 setup
        self.current_position = None
        self.goal_position = None

        # Subscribe robot position
        self.subscription = self.create_subscription(
            Point,
            '/robot/current_position', # Ndak tau ini nama topik yang disubscribe
            self.position_callback,
            10)
        
        # Publish generated foosteps
        self.publisher = self.create_publisher(
            Point,
            '/ashioto/next_footstep', # Ndak tau ini nama topik yang dipublish 
            10)
        
        # Publish the generated support foot
        self.support_publisher = self.create_publisher(
            String,
            '/ashioto/support_footstep', # Ndak tau ini nama topik yang dipublish
            10)
        
        # Planning timer
        self.timer = self.create_timer(0.1, self.planning_callback)
        self.footstep_generator = None

    def position_callback(self, msg):
        """Store current robot position"""
        self.current_position = [msg.x, msg.y, msg.z]
        self.get_logger().info(f"Received position: {self.current_position}")

    def set_goal(self, goal):
        """Initialize new footstep plan"""
        self.options = {
            # Maximum steps
            "max_dx_forward": 0.08,  # [m]
            "max_dx_backward": 0.03,  # [m]
            "max_dy": 0.04,  # [m]
            "max_dtheta": np.deg2rad(20),  # [rad]
            # Target tolerance
            "tolerance_distance": 0.05,  # [m]
            "tolerance_angle": np.deg2rad(5),  # [rad]
            # Do we include collisions with the ball?
            "has_obstacle": False,
            "obstacle_max_radius": 0.25,  # [m]
            "obstacle_radius": None,  # [m]
            "obstacle_position": np.array([0, 0], dtype=np.float32),  # [m,m]
            # Which foot is targeted (any, left or right)
            "foot": "any",
            # Foot geometry
            "foot_length": 0.14,  # [m]
            "foot_width": 0.08,  # [m]
            "feet_spacing": 0.15,  # [m]
            # Add reward shaping term
            "shaped": True,
            # If True, the goal will be sampled in a 4x4m area, else it will be fixed at (0,0)
            "multi_goal": False,
            "start_foot_pose": np.array(self.current_position),
            "target_foot_pose": np.array(goal),
            "panjang": 8, # [m]
            "lebar": 6, # [m]
        }
        obs, _ = self.env.reset(options=self.options)
        self.footstep_generator = self.generate_footsteps(obs)

    def generate_footsteps(self, initial_obs):
        """Generator for footstep coordinates"""
        obs = initial_obs
        while True:
            obs_input = np.expand_dims(obs, axis=0).astype(np.float32)
            result = self.compiled_model([obs_input])[self.output_layer]
            action = np.squeeze(result, axis=0)
            obs, _, terminated, _, info = self.env.step(action)
            
            if terminated:
                self.env.close()
                break
            
            yield info["Foot Coord"], info["Support Foot"]

    def planning_callback(self):
        """Publish next footstep coordinate"""
        if self.footstep_generator is None:
            return

        try:
            next_coord, support_foot = next(self.footstep_generator)

            # Publish the footsteps Coords
            coord_msg = Point()
            coord_msg.x = float(next_coord[0])
            coord_msg.y = float(next_coord[1])
            coord_msg.z = float(next_coord[2])
            self.publisher.publish(coord_msg)

            # Publish the support foot
            support_msg = String()
            support_msg.data = str(support_foot)
            self.support_publisher.publish(support_msg)
            self.get_logger().info(f"Published footstep: {next_coord}")
            self.get_logger().info(f"Published support_foot: {support_foot}")
        except StopIteration:
            self.get_logger().info("Footstep planning completed")
            self.footstep_generator = None

# Usage example
if __name__ == "__main__":
    rclpy.init(args=None)
    planner = FootstepsPlanner()
    
    # Set initial goal (you can modify this to receive goals dynamically)
    planner.current_position = [0, 0, 0]
    planner.set_goal([8.0, 6.0, 0.0])  # Example goal
    
    rclpy.spin(planner)
    planner.destroy_node()
    rclpy.shutdown()