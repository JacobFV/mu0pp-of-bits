import torch
import torch.nn as nn
import torch.nn.functional as F

class MuZeroNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Representation network h_θ: o_t → s_t
        self.representation = nn.Sequential(
            nn.Conv2d(config.observation_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, config.hidden_size, kernel_size=3, padding=1)
        )

        # Dynamics network g_θ: (s_t, a_t) → (r_t, s_{t+1})
        self.dynamics_state = nn.Sequential(
            nn.Conv2d(config.hidden_size + config.action_embedding_size, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, config.hidden_size, kernel_size=3, padding=1)
        )
        self.dynamics_reward = nn.Sequential(
            nn.Conv2d(config.hidden_size, 1, kernel_size=1),
            nn.Flatten(),
            nn.Linear(config.board_size**2, 1)
        )

        # Prediction network f_θ: s_t → (p_t, v_t)
        self.prediction_policy = nn.Sequential(
            nn.Conv2d(config.hidden_size, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * config.board_size**2, config.action_space_size)
        )
        self.prediction_value = nn.Sequential(
            nn.Conv2d(config.hidden_size, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * config.board_size**2, 1),
            nn.Tanh()
        )

        # Action embedding
        self.action_embedding = nn.Embedding(config.action_space_size, config.action_embedding_size)

    def representation(self, observation):
        """
        Representation function h_θ: o_t → s_t
        Maps raw observation to hidden state.
        """
        return self.representation(observation)

    def dynamics(self, state, action):
        """
        Dynamics function g_θ: (s_t, a_t) → (r_t, s_{t+1})
        Predicts next hidden state and immediate reward.
        """
        action_embedding = self.action_embedding(action).view(-1, self.config.action_embedding_size, 1, 1)
        action_embedding = action_embedding.expand(-1, -1, state.shape[2], state.shape[3])
        x = torch.cat([state, action_embedding], dim=1)
        next_state = self.dynamics_state(x)
        reward = self.dynamics_reward(next_state)
        return next_state, reward

    def prediction(self, state):
        """
        Prediction function f_θ: s_t → (p_t, v_t)
        Predicts policy and value from hidden state.
        """
        policy = self.prediction_policy(state)
        value = self.prediction_value(state)
        return policy, value

    def initial_inference(self, observation):
        """
        Initial inference for root state.
        """
        state = self.representation(observation)
        policy, value = self.prediction(state)
        return state, policy, value

    def recurrent_inference(self, hidden_state, action):
        """
        Recurrent inference for subsequent states.
        """
        next_state, reward = self.dynamics(hidden_state, action)
        policy, value = self.prediction(next_state)
        return next_state, reward, policy, value
