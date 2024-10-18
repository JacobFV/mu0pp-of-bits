import torch
from muzero.agent import MuZeroAgent
from muzero.config import MuZeroConfig
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from typing import Tuple
from openrl.envs.common import make

import gym
import numpy as np

patch_typeguard()

if __name__ == "__main__":
    # Initialize the base CarRacing-v2 environment
    base_env = gym.make('CarRacing-v2')

    # Define discrete actions
    discrete_actions = [
        np.array([0, 0, 0]),    # No Action
        np.array([0, 0, 0.8]),  # Accelerate
        np.array([0, 0.8, 0]),  # Brake
        np.array([-1, 0, 0]),   # Steer Left
        np.array([1, 0, 0]),    # Steer Right
    ]

    # Wrap the environment to use discrete actions
    class DiscretizedActionEnv(gym.Env):
        def __init__(self, env, discrete_actions):
            self.env = env
            self.discrete_actions = discrete_actions
            self.action_space = gym.spaces.Discrete(len(discrete_actions))
            self.observation_space = env.observation_space

        def step(self, action):
            continuous_action = self.discrete_actions[action]
            obs, reward, done, info = self.env.step(continuous_action)
            return obs, reward, done, info

        def reset(self, **kwargs):
            return self.env.reset(**kwargs)

        def render(self, mode='human'):
            return self.env.render(mode=mode)

        def close(self):
            self.env.close()

    # Use the wrapped environment
    env = DiscretizedActionEnv(base_env, discrete_actions)

    # Now that the environment is created, extract the observation and action space information
    observation_space = env.observation_space
    action_space = env.action_space

    # Update the configuration with environment-specific information
    config = MuZeroConfig(
        action_space_size=action_space.n,  # Dynamically set from environment
        action_embedding_size=6,           # As per your config.py
        observation_shape=observation_space.shape[:2],   # Height and Width of the image
        observation_channels=observation_space.shape[2], # Number of color channels (3 for RGB)
        board_size=observation_space.shape[0],           # Assuming square input, adjust if necessary
        # Include any other necessary configurations
    )

    # Initialize the agent with the environment
    agent = MuZeroAgent(config, environment=env)

    # Start the training process
    agent.train()
