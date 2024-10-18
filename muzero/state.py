import torch
from dataclasses import dataclass
import gym
from typing import Tuple
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()

@dataclass
class State:
    observation: TensorType["channels", "height", "width"]
    hidden_state: TensorType["hidden_size"] | None = None
    environment: gym.Env = None  # Add environment to the state

    @typechecked
    def is_terminal(self) -> bool:
        # Implement logic to check if the state is terminal
        # Example for CartPole-v1:
        return self.environment.env._elapsed_steps >= self.environment.spec.max_episode_steps

    @typechecked
    def next_state(self, action: int) -> Tuple['State', float, bool]:
        # Apply the action to the environment and get the next observation and reward
        observation, reward, done, info = self.environment.step(action)
        return State(observation=observation, hidden_state=None, environment=self.environment), reward, done
