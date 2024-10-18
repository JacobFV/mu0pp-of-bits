import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked


class MuZeroNetwork(nn.Module):
    def __init__(self, config: 'MuZeroConfig'):
        super().__init__()
        self.config = config

        # Representation network h_θ: o_t → s_t
        self.representation_network = nn.Sequential(
            nn.Conv2d(config.observation_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
            # Output will be flattened to a vector
        )

        # Dynamics network g_θ: (s_t, a_t) → (r_t, s_{t+1})
        self.dynamics_state_network = nn.Sequential(
            nn.Conv2d(config.hidden_size + config.action_embedding_size, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, config.hidden_size, kernel_size=3, padding=1)
        )
        self.dynamics_reward_network = nn.Sequential(
            nn.Conv2d(config.hidden_size, 1, kernel_size=1),
            nn.Flatten(),
            nn.Linear(config.board_size**2, 1)
        )

        # Prediction network f_θ: s_t → (p_t, v_t)
        self.prediction_policy_network = nn.Sequential(
            nn.Conv2d(config.hidden_size, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * config.board_size**2, config.action_space_size)
        )
        self.prediction_value_network = nn.Sequential(
            nn.Conv2d(config.hidden_size, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * config.board_size**2, 1),
            nn.Tanh()
        )

        # Action embedding
        self.action_embedding = nn.Embedding(config.action_space_size, config.action_embedding_size)

    @typechecked
    def representation(self, observation: TensorType["batch", "channels", "height", "width"]) -> TensorType["batch", "hidden_size"]:
        """
        Encode the observation into a hidden state.

        Parameters:
            observation (torch.Tensor): The observation tensor of shape (B, C, H, W).

        Returns:
            hidden_state (torch.Tensor): The encoded hidden state of shape (B, hidden_size).
        """
        # Pass the observation through the representation network
        hidden_state = self.representation_network(observation)
        return hidden_state

    @typechecked
    def dynamics(self, state: TensorType["batch", "hidden_size", "height", "width"], action: TensorType["batch"]) -> Tuple[TensorType["batch", "hidden_size", "height", "width"], TensorType["batch", 1]]:
        """
        Dynamics function g_θ: (s_t, a_t) → (r_t, s_{t+1})
        Predicts next hidden state and immediate reward.

        Parameters:
            state (torch.Tensor): The current state tensor of shape (B, hidden_size, H, W).
            action (torch.Tensor): The action tensor of shape (B,).

        Returns:
            next_state (torch.Tensor): The next state tensor of shape (B, hidden_size, H, W).
            reward (torch.Tensor): The reward tensor of shape (B, 1).
        """
        action_embedding = self.action_embedding(action).view(-1, self.config.action_embedding_size, 1, 1)
        action_embedding = action_embedding.expand(-1, -1, state.shape[2], state.shape[3])
        x = torch.cat([state, action_embedding], dim=1)
        next_state = self.dynamics_state_network(x)
        reward = self.dynamics_reward_network(next_state)
        return next_state, reward

    @typechecked
    def prediction(self, state: TensorType["batch", "hidden_size", "height", "width"]) -> Tuple[TensorType["batch", "action_space_size"], TensorType["batch", 1]]:
        """
        Prediction function f_θ: s_t → (p_t, v_t)
        Predicts policy and value from hidden state.

        Parameters:
            state (torch.Tensor): The state tensor of shape (B, hidden_size, H, W).

        Returns:
            policy (torch.Tensor): The policy tensor of shape (B, action_space_size).
            value (torch.Tensor): The value tensor of shape (B, 1).
        """
        policy = self.prediction_policy_network(state)
        value = self.prediction_value_network(state)
        return policy, value

    @typechecked
    def initial_inference(self, observation: TensorType["batch", "channels", "height", "width"]) -> Tuple[TensorType["batch", "hidden_size", "height", "width"], TensorType["batch", "action_space_size"], TensorType["batch", 1]]:
        """
        Initial inference for root state.

        Parameters:
            observation (torch.Tensor): The observation tensor of shape (B, C, H, W).

        Returns:
            state (torch.Tensor): The state tensor of shape (B, hidden_size, H, W).
            policy (torch.Tensor): The policy tensor of shape (B, action_space_size).
            value (torch.Tensor): The value tensor of shape (B, 1).
        """
        state = self.representation(observation)
        policy, value = self.prediction(state)
        return state, policy, value

    @typechecked
    def recurrent_inference(self, hidden_state: TensorType["batch", "hidden_size", "height", "width"], action: TensorType["batch"]) -> Tuple[TensorType["batch", "hidden_size", "height", "width"], TensorType["batch", 1], TensorType["batch", "action_space_size"], TensorType["batch", 1]]:
        """
        Recurrent inference for subsequent states.

        Parameters:
            hidden_state (torch.Tensor): The hidden state tensor of shape (B, hidden_size, H, W).
            action (torch.Tensor): The action tensor of shape (B,).

        Returns:
            next_state (torch.Tensor): The next state tensor of shape (B, hidden_size, H, W).
            reward (torch.Tensor): The reward tensor of shape (B, 1).
            policy (torch.Tensor): The policy tensor of shape (B, action_space_size).
            value (torch.Tensor): The value tensor of shape (B, 1).
        """
        next_state, reward = self.dynamics(hidden_state, action)
        policy, value = self.prediction(next_state)
        return next_state, reward, policy, value
