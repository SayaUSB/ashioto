# Copyright (c) 2025 ICHIRO ITS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import numpy as np
from rclpy.node import Node
from openvino.runtime import Core
from aruku_interfaces.msg import Status as Odometry
from kansei_interfaces.msg import Status
from ashioto_interfaces.msg import Footstep, Footsteps, Goalposition, Obstacle
from ashioto_initialize import initialize
from gymnasium.wrappers import TimeLimit

class FootstepsPlanner(Node):
    def __init__(self):
        super().__init__("footstep_planner")

        # Initialize OpenVINO model
        self.core = Core()
        self.model = self.core.read_model("openvino_model/footsteps_planning_right.xml") # Please give the correct directory for the model
        self.compiled_model = self.core.compile_model(self.model, "CPU")
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

        # Options for the planner
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
            "obstacle_position": np.array([0, 0], dtype=np.float64),  # [m,m]
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
            "start_foot_pose": np.array(self.current_position, dtype=np.float64),
            "target_foot_pose": np.array(self.goal_position, dtype=np.float64),
            "panjang": 8, # [m]
            "lebar": 6, # [m]
        }

        # Initialize Environment
        env = initialize(options=self.options)
        env = TimeLimit(env, max_episode_steps=1000)
        obs, _ = env.reset()

        # ROS2 setup
        self.current_position = [0, 0, 0]
        self.goal_position = [0, 0, 0]

        # Subscribe current robot position
        self.subscription = self.create_subscription(
            Odometry,
            'walking/odometry',
            self.position_callback,
            10)
        
        # Subscribe current robot orientation
        self.orientation_subscription = self.create_subscription(
            Status, 
            'measurement/status',
            self.orientation_subscription_callback,
            10)

        # Subscribe robot goal position
        self.goal_subscription = self.create_subscription(
            Goalposition,
            'walking/set_walking',
            self.goal_position_callback,
            10)

        # Subscribe obstacle position
        self.obstacle_subscription = self.create_subscription(
            Obstacle,
            'obstacle/position', # Dummy topic
            self.obstacle_subcription_callback,
            10)

        # Publish generated foosteps and support foot
        self.footstep_publisher = self.create_publisher(
            Footsteps,
            '/ashioto/next_footstep',
            10)
        
        # Planning timer
        self.timer = self.create_timer(0.1, self.planning_callback)
        self.footstep_generator = None

    def goal_position_callback(self, msg):
        """Store robot goal position"""
        self.goal_position = [msg.goal_position.x, msg.goal_position.y, msg.orientation.yaw]
        self.get_logger().info(f"Received position: {self.goal_position}")

    def orientation_subscription_callback(self, msg):
        """Store current robot orientation"""
        self.current_position[2] = msg.orientation.yaw

    def obstacle_subcription_callback(self, msg):
        """Store obstacle detection coords"""
        self.obstacle_detected = msg.detected
        self.obstacle_position = [msg.position.x, msg.position.y]
        self.obstacle_radius = msg.radius
        self.get_logger().info(f"Received obstacle position: {self.obstacle_position}")
        self.get_logger().info(f"Received obstacle radius: {self.obstacle_radius}")

    def position_callback(self, msg):
        """Store current robot position"""
        self.current_position = [msg.Odometry.x, msg.Odometry.y, self.current_position[2]] # [msg.x, msg.y, msg.z] -> 'z' is for the orientation
        self.get_logger().info(f"Received position: {self.current_position}")

    def set_goal(self, goal, target_foot):
        """Initialize new footstep plan"""
        self.options["target_foot_pose"] = np.array(goal, dtype=np.float64)
        self.options["foot"] = target_foot
        if self.obstacle_detected:
            radius = np.array(self.obstacle_radius, dtype=np.float64)
            pos = np.array(self.obstacle_position, dtype=np.float64)
            self.options["has_obstacle"] = True
            self.options["obstacle_radius"], self.options["obstacle_max_radius"] = radius, radius
            self.options["obstacle_position"] = pos
        obs, _ = self.env.reset(options=self.options)
        self.footstep_generator = self.generate_footsteps(obs)

    def generate_footsteps(self, initial_obs):
        """Generator for footstep coordinates"""
        obs = initial_obs
        while True:
            obs_input = np.expand_dims(obs, axis=0).astype(np.float64)
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

            # Publish the footsteps Coords and support foot
            msg = Footsteps()
            coord_msg = Footstep()
            coord_msg.position = Odometry(float(next_coord[0]), float(next_coord[1])) # x, y
            coord_msg.orientation = float(next_coord[2])
            coord_msg.support_foot = str(support_foot)
            msg.footsteps.append(coord_msg)
            self.footstep_publisher.publish(msg)
            self.get_logger().info(f"Published footstep: {next_coord}")
            self.get_logger().info(f"Published support_foot: {support_foot}")
        except StopIteration:
            self.get_logger().info("Footstep planning completed")
            self.footstep_generator = None
