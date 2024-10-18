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

    @typechecked
    def is_terminal(self) -> bool:
        # Implement logic to check if the state is terminal
        # For example, in CartPole, you can check if the pole has fallen
        return False  # Placeholder

    @typechecked
    def next_state(self, action: int, environment: gym.Env) -> Tuple['State', float, bool]:
        # Apply the action to the environment and get the next observation and reward
        observation, reward, done, info = environment.step(action)
        return State(observation), reward, done
