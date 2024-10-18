import torch
import torch.optim as optim
import random
import gym

from muzero.network import MuZeroNetwork
from muzero.mcts import Node, run_mcts
from muzero.state import State  # Ensure you have a State class


class MuZeroAgent:
    def __init__(self, config, environment):
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
        self.replay_buffer = []
        
        # Initialize other necessary components or variables.
        # Example: self.loss_fn = SomeLossFunction()
    
    def train(self):
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
    
    def self_play(self):
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
        game = []
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
    
    def select_action(self, root):
        # Implement your action selection logic
        # Typically, you select the action with the highest visit count
        visit_counts = [(child.visit_count, action) for action, child in root.children.items()]
        _, action = max(visit_counts)
        return action
    
    def sample_batch(self):
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
    
    def update_weights(self, batch):
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
        
        total_loss = 0
        
        for trajectory in batch:
            for (observation, action) in trajectory:
                # ====================
                # Forward Pass
                # ====================
                # Compute hidden state: s_k = h_θ(o_k)
                hidden_state = self.network.representation(observation)
                
                # ====================
                # Prediction
                # ====================
                # Obtain policy logits and value estimate: (π̂_k, v̂_k) = f_θ(s_k)
                policy_logits, value_estimate = self.network.prediction(hidden_state)
                
                # ====================
                # Compute Target Values
                # ====================
                # Example targets: policy_target = π_k, value_target = v_k
                # These targets are derived from MCTS visit counts and game outcomes.
                policy_target = self.compute_policy_target(trajectory)
                value_target = self.compute_value_target(trajectory)
                reward_target = self.compute_reward_target(trajectory)
                
                # ====================
                # Calculate Loss Components
                # ====================
                # Value Loss: ℓ^{value}_k = (v̂_k - v_k)^2
                value_loss = (value_estimate - value_target).pow(2)
                
                # Policy Loss: ℓ^{policy}_k = - π_k log π̂_k
                # Using negative log likelihood for cross-entropy loss.
                policy_loss = -torch.sum(policy_target * torch.log_softmax(policy_logits, dim=-1), dim=-1)
                
                # Reward Loss: ℓ^{reward}_k = (r̂_k - r_k)^2
                # Assuming the network has a reward head to predict r̂_k.
                reward_pred = self.network.reward_head(hidden_state)
                reward_loss = (reward_pred - reward_target).pow(2)
                
                # Aggregate losses
                loss = value_loss + policy_loss + reward_loss
                total_loss += loss
        
        # Average the total loss over the batch.
        average_loss = total_loss / len(batch)
        
        # ====================
        # Backward Pass and Optimization
        # ====================
        average_loss.backward()  # Compute gradients: ∇_θ L(θ)
        
        # Apply gradient clipping to stabilize training:
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
        
        # Update network parameters θ: θ ← θ - α ∇_θ L(θ)
        self.optimizer.step()
    
    def save_model(self, path):
        """
        Persist the model parameters θ to the specified filesystem path.
        
        Parameters:
            path (str): Filesystem path to save the model parameters.
        
        Mathematical Relation:
            θ ↦ saved state_dict
        """
        torch.save(self.network.state_dict(), path)
    
    def load_model(self, path):
        """
        Load the model parameters θ from the specified filesystem path.
        
        Parameters:
            path (str): Filesystem path from which to load the model parameters.
        
        Mathematical Relation:
            saved state_dict ↦ θ
        """
        self.network.load_state_dict(torch.load(path))
    
    def initial_state(self):
        """
        Initialize the starting state s₀ of the environment.
        
        Returns:
            state (State): The initial state object.
        
        Mathematical Representation:
            s₀ = h_θ(o₀)
        """
        # Create an initial observation based on the environment
        initial_observation = self.environment.reset()
        
        # Apply the representation function h_θ to get the initial hidden state
        initial_hidden_state = self.network.representation(torch.tensor(initial_observation).float().unsqueeze(0))
        
        # Create and return a State object
        return State(observation=initial_observation, hidden_state=initial_hidden_state)
    
    # ====================
    # Auxiliary Functions
    # ====================
    
    def compute_policy_target(self, trajectory):
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
    
    def compute_value_target(self, trajectory):
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
    
    def compute_reward_target(self, trajectory):
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
