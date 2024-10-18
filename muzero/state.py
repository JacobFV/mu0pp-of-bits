import torch
from dataclasses import dataclass
@dataclass
class State:
    observation: torch.Tensor
    hidden_state: torch.Tensor | None = None

    def is_terminal(self):
        # Implement logic to check if the state is terminal
        # For example, in CartPole, you can check if the pole has fallen
        return False  # Placeholder

    def next_state(self, action, environment):
        # Apply the action to the environment and get the next observation and reward
        observation, reward, done, info = environment.step(action)
        return State(observation), reward, done
