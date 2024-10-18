import torch
import torch.optim as optim
import random
import gym
import numpy as np
from typing import List, Tuple

from muzero.network import MuZeroNetwork
from muzero.mcts import Node, run_mcts
from muzero.state import State
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()

class MuZeroAgent:
    @typechecked
    def __init__(self, config: 'MuZeroConfig', environment: gym.Env):
        """
        Initialize the MuZero agent with the given configuration.
        
        Parameters:
            config (MuZeroConfig): Configuration parameters containing hyperparameters.
            environment (Environment): The environment for the agent to interact with.
        
        Mathematical Components:
            - θ: Neural network parameters encompassing h_θ, g_θ, f_θ.
            - D: Replay buffer storing trajectories τ.
            - α: Learning rate for optimizer.
        """
        self.config = config
        self.environment = environment  # Include the environment
        
        # Initialize the neural network parameterized by θ.
        # θ encompasses parameters for h_θ (representation), g_θ (dynamics), and f_θ (prediction).
        self.network = MuZeroNetwork(config)
        
        # Initialize the optimizer for minimizing the loss function L(θ).
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.config.learning_rate)
        
        # Initialize the replay buffer D for storing game experiences.
        self.replay_buffer: List[List[Tuple[np.ndarray, int]]] = []
        
        # Initialize other necessary components or variables.
        # Example: self.loss_fn = SomeLossFunction()
    
    @typechecked
    def train(self) -> None:
        """
        Execute the main training loop consisting of self-play and network parameter updates.
        
        Objective:
            Maximize the expected return J(π_θ) by optimizing θ to minimize L(θ).
        
        Algorithmic Steps:
            for t = 1 to T:
                1. Generate trajectory τ_t via self-play using MCTS.
                2. Sample a mini-batch {τ_i} from D.
                3. Compute gradients ∇_θ L(θ; {τ_i}).
                4. Update θ ← θ - α ∇_θ L(θ; {τ_i}).
        """
        for iteration in range(self.config.num_iterations):
            # ====================
            # Self-Play Phase
            # ====================
            self.self_play()  # Generate game data through self-play using MCTS.
            # Collect trajectory: τ = {(s_0, a_0), (s_1, a_1), ..., (s_T, a_T)}
            
            # ====================
            # Sampling Phase
            # ====================
            batch = self.sample_batch()  # Sample a mini-batch from the replay buffer D.
            # Sampled data: {τ_i} for i ∈ [1, B]
            
            # ====================
            # Optimization Phase
            # ====================
            self.update_weights(batch)  # Perform a gradient descent step to minimize L(θ).
    
    @typechecked
    def self_play(self) -> None:
        """
        Generate a single game trajectory through self-play using MCTS.
        
        Process:
            1. Initialize the game state s₀.
            2. For each timestep t:
                a. Perform MCTS simulations to select action a_t based on Σ simulations.
                b. Execute action a_t to transition to state s_{t+1}.
                c. Record (s_t, a_t) in the game trajectory.
            3. Store the completed trajectory τ in the replay buffer D.
        
        Mathematical Representation:
            τ = {(s_0, a_0), (s_1, a_1), ..., (s_T, a_T)}
        """
        game: List[Tuple[np.ndarray, int]] = []
        state = self.initial_state()  # Initialize s₀
        
        while not state.is_terminal():
            root = Node(state)
            
            # Execute MCTS to expand the search tree rooted at root.
            run_mcts(root, self.network, self.config)
            
            # Select action a_t based on MCTS visit counts and probabilistic sampling.
            action = self.select_action(root)
            
            # Record the current state observation and selected action.
            game.append((state.observation, action))
            
            # Transition to the next state s_{t+1} based on action a_t.
            state = state.next_state(action)
        
        # Add the completed game trajectory τ to the replay buffer D.
        self.replay_buffer.append(game)
    
    @typechecked
    def select_action(self, root: Node) -> int:
        # Implement your action selection logic
        # Typically, you select the action with the highest visit count
        visit_counts = [(child.visit_count, action) for action, child in root.children.items()]
        _, action = max(visit_counts)
        return action
    
    @typechecked
    def sample_batch(self) -> List[List[Tuple[np.ndarray, int]]]:
        """
        Sample a mini-batch of trajectories from the replay buffer D for training.
        
        Returns:
            batch (list): A list of sampled game trajectories {τ_i}.
        
        Sampling Distribution:
            Uniform random sampling: P(τ_i) = 1 / |D| for all τ_i ∈ D
        """
        # Ensure the replay buffer has enough samples.
        if len(self.replay_buffer) < self.config.batch_size:
            return self.replay_buffer
        
        # Randomly sample B trajectories from D without replacement.
        batch = random.sample(self.replay_buffer, self.config.batch_size)
        return batch
    
    @typechecked
    def update_weights(self, batch: List[List[Tuple[np.ndarray, int]]]) -> None:
        """
        Update the neural network parameters θ by minimizing the loss function L(θ).
        
        Parameters:
            batch (list): A mini-batch of sampled game trajectories {τ_i}.
        
        Loss Function:
            L(θ) = ∑_{τ ∈ batch} ∑_{k=0}^{T} (ℓ^{value}_k + ℓ^{policy}_k + ℓ^{reward}_k)
        
        Gradient Descent Update:
            θ ← θ - α ∇_θ L(θ)
        
        Constraints:
            - Gradient clipping to prevent exploding gradients.
            - Regularization terms (e.g., L2) can be included to prevent overfitting.
        
        Detailed Mathematical Steps:
            1. For each trajectory τ in batch:
                a. For each timestep k in τ:
                    i.   Compute hidden state: s_k = h_θ(o_k)
                    ii.  Obtain predictions: (π̂_k, v̂_k) = f_θ(s_k)
                    iii. Compute target values: π_k, v_k, r_k
                    iv.  Calculate loss components:
                         - ℓ^{value}_k = (v̂_k - v_k)^2
                         - ℓ^{policy}_k = -π_k log π̂_k
                         - ℓ^{reward}_k = (r̂_k - r_k)^2
            2. Aggregate losses and compute gradients.
            3. Update θ using the optimizer.
        """
        # Zero the gradients from the previous optimization step.
        self.optimizer.zero_grad()
        
        total_loss: TensorType["batch_size"] = torch.zeros(len(batch))
        
        for i, trajectory in enumerate(batch):
            for (observation, action) in trajectory:
                # ====================
                # Forward Pass
                # ====================
                # Compute hidden state: s_k = h_θ(o_k)
                hidden_state: TensorType["hidden_size"] = self.network.representation(observation)
                
                # ====================
                # Prediction
                # ====================
                # Obtain policy logits and value estimate: (π̂_k, v̂_k) = f_θ(s_k)
                policy_logits: TensorType["action_space"] = self.network.prediction(hidden_state)[0]
                value_estimate: TensorType[1] = self.network.prediction(hidden_state)[1]
                
                # ====================
                # Compute Target Values
                # ====================
                # Example targets: policy_target = π_k, value_target = v_k
                # These targets are derived from MCTS visit counts and game outcomes.
                policy_target: TensorType["action_space"] = self.compute_policy_target(trajectory)
                value_target: TensorType[1] = self.compute_value_target(trajectory)
                reward_target: TensorType[1] = self.compute_reward_target(trajectory)
                
                # ====================
                # Calculate Loss Components
                # ====================
                # Value Loss: ℓ^{value}_k = (v̂_k - v_k)^2
                value_loss: TensorType[1] = (value_estimate - value_target).pow(2)
                
                # Policy Loss: ℓ^{policy}_k = - π_k log π̂_k
                # Using negative log likelihood for cross-entropy loss.
                policy_loss: TensorType[1] = -torch.sum(policy_target * torch.log_softmax(policy_logits, dim=-1), dim=-1)
                
                # Reward Loss: ℓ^{reward}_k = (r̂_k - r_k)^2
                # Assuming the network has a reward head to predict r̂_k.
                reward_pred: TensorType[1] = self.network.reward_head(hidden_state)
                reward_loss: TensorType[1] = (reward_pred - reward_target).pow(2)
                
                # Aggregate losses
                loss: TensorType[1] = value_loss + policy_loss + reward_loss
                total_loss[i] += loss
        
        # Average the total loss over the batch.
        average_loss: TensorType[1] = total_loss.mean()
        
        # ====================
        # Backward Pass and Optimization
        # ====================
        average_loss.backward()  # Compute gradients: ∇_θ L(θ)
        
        # Apply gradient clipping to stabilize training:
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
        
        # Update network parameters θ: θ ← θ - α ∇_θ L(θ)
        self.optimizer.step()
    
    @typechecked
    def save_model(self, path: str) -> None:
        """
        Persist the model parameters θ to the specified filesystem path.
        
        Parameters:
            path (str): Filesystem path to save the model parameters.
        
        Mathematical Relation:
            θ ↦ saved state_dict
        """
        torch.save(self.network.state_dict(), path)
    
    @typechecked
    def load_model(self, path: str) -> None:
        """
        Load the model parameters θ from the specified filesystem path.
        
        Parameters:
            path (str): Filesystem path from which to load the model parameters.
        
        Mathematical Relation:
            saved state_dict ↦ θ
        """
        self.network.load_state_dict(torch.load(path))
    
    @typechecked
    def initial_state(self) -> State:
        """
        Initialize the starting state s₀ of the environment.
        
        Returns:
            state (State): The initial state object.
        
        Mathematical Representation:
            s₀ = h_θ(o₀)
        """
        # Unpack the observation and info from the environment reset
        initial_observation, info = self.environment.reset()
        print("Initial observation:", initial_observation)
        print("Info:", info)
        
        # Use the initial_observation directly as your observation data
        observation_data = initial_observation

        print("Observation data before conversion:", observation_data)
        
        # Convert observation_data to a NumPy array if it isn't already one
        if isinstance(observation_data, np.ndarray):
            observation_data_np = observation_data
        elif isinstance(observation_data, list):
            observation_data_np = np.array(observation_data)
        else:
            # Handle other data types (e.g., torch tensors, scalars)
            observation_data_np = np.array([observation_data])
        
        print("Observation data shape:", observation_data_np.shape)
        
        # Check if the data has the expected shape
        if observation_data_np.size == 0:
            raise ValueError("Observation data is empty. Cannot proceed.")
        
        # Convert the data into a tensor
        observation_tensor: TensorType["batch", "channels", "height", "width"] = torch.from_numpy(observation_data_np).float()
        
        # Add batch dimension if necessary
        if observation_tensor.dim() == 1:
            observation_tensor = observation_tensor.unsqueeze(0)
        
        print("Observation tensor shape:", observation_tensor.shape)
        
        # Apply the representation function h_θ to get the initial hidden state
        initial_hidden_state: TensorType["hidden_size"] = self.network.representation(observation_tensor)
        
        # Create and return a State object
        return State(observation=initial_observation, hidden_state=initial_hidden_state)
    
    # ====================
    # Auxiliary Functions
    # ====================
    
    @typechecked
    def compute_policy_target(self, trajectory: List[Tuple[np.ndarray, int, dict]]) -> TensorType["action_space"]:
        """
        Compute the target policy π_k(a) derived from MCTS visit counts.
        
        Mathematical Definition:
            π_k(a) = N(s_k, a) / Σ_{b} N(s_k, b)
        
        Where:
            - N(s_k, a): Visit count of action a at state s_k
            - Σ_{b} N(s_k, b): Total visit counts over all actions at state s_k
        
        Parameters:
            trajectory (list): A list of (observation, action, mcts_info) tuples.
        
        Returns:
            policy_target (torch.Tensor): Target policy distribution over actions.
        """
        policy_targets = []
        for _, _, mcts_info in trajectory:
            visit_counts = torch.tensor([mcts_info.child_visits[a] for a in range(self.config.action_space)])
            policy_target = visit_counts / visit_counts.sum()
            policy_targets.append(policy_target)
        return torch.stack(policy_targets)
    
    @typechecked
    def compute_value_target(self, trajectory: List[Tuple[np.ndarray, int, float, bool]]) -> TensorType["trajectory_length"]:
        """
        Compute the target value v_k based on the game outcome.
        
        Mathematical Definition:
            v_k = R(s_k, a_k) + γ R(s_{k+1}, a_{k+1}) + ... + γ^{n-1} R(s_{k+n-1}, a_{k+n-1})} + γ^n V(s_{k+n})
        
        Where:
            - R(s, a): Reward function
            - V(s): Value function estimate from the neural network
        
        Parameters:
            trajectory (list): A list of (observation, action, reward, done) tuples.
        
        Returns:
            value_target (torch.Tensor): Scalar value target.
        """
        value_targets = []
        bootstrap_value = 0
        for observation, action, reward, done in reversed(trajectory):
            if done:
                bootstrap_value = 0
            else:
                bootstrap_value = reward + self.config.discount * bootstrap_value
            value_targets.append(bootstrap_value)
        return torch.tensor(list(reversed(value_targets)))
    
    @typechecked
    def compute_reward_target(self, trajectory: List[Tuple[np.ndarray, int, float]]) -> TensorType["trajectory_length"]:
        """
        Compute the target reward r_k from the trajectory.
        
        Mathematical Definition:
            r_k = R(s_k, a_k)
        
        Parameters:
            trajectory (list): A list of (observation, action, reward) tuples.
        
        Returns:
            reward_target (torch.Tensor): Scalar reward target.
        """
        return torch.tensor([reward for _, _, reward in trajectory])
