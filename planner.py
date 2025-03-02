import numpy as np
import gymnasium as gym
from openvino.runtime import Core
from gymnasium.envs.registration import register

class FootstepsPlanner:
    def __init__(self, mode):
        # Initialize OpenVINO model
        self.core = Core()
        self.model = self.core.read_model(f"openvino_model/footsteps_planning_{mode}.xml")
        self.compiled_model = self.core.compile_model(self.model, "CPU")
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        self.mode = mode
        # Register environment
        register(
            id=f"footsteps-planning-{mode}-v0",
            entry_point=f"gym_footsteps_planning.envs:FootstepsPlanning{mode}Env",
            max_episode_steps=1000,
        )
        self.env = gym.make(f"footsteps-planning-{mode}-v0")

    def plan_footsteps(self, start_coords, end_coords):
        """Generator function that yields foot coordinates at each step"""
        options = {
            "start_foot_pose": np.array(start_coords),
            "target_foot_pose": np.array(end_coords),
            "panjang": 8,
            "lebar": 6,
        }
        
        obs, _ = self.env.reset(options=options)
        
        while True:
            obs_input = np.expand_dims(obs, axis=0).astype(np.float32)
            result = self.compiled_model([obs_input])[self.output_layer]
            action = np.squeeze(result, axis=0)
            obs, _, terminated, _, info = self.env.step(action)
            
            yield info["Foot Coord"]
            
            if terminated:
                self.env.close()
                break

# Usage example
if __name__ == "__main__":
    planner = FootstepsPlanner("Right")
    
    # Example coordinates
    start = [0.0, 0.0, 0.0]
    end = [5.0, 5.0, 3.0]
    
    # Subscribe to foot coordinates
    for coord in planner.plan_footsteps(start, end):
        print(f"Foot Coordinate: {coord}")