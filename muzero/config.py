from dataclasses import dataclass

@dataclass
class MuZeroConfig:
    # ====================
    # Monte Carlo Tree Search (MCTS) Parameters
    # ====================

    # Number of simulations per MCTS search: K
    # Controls the depth and breadth of the search tree.
    num_simulations = 800  # K ∈ ℕ
    # Reference: Determines the number of iterations in MCTS for policy improvement.

    # Exploration constant in the UCB formula: c_{puct}
    # Balances exploration vs. exploitation in selecting actions during MCTS.
    c_puct = 1.25  # c_{puct} ∈ ℝ^+
    # Reference: Upper Confidence Bound (UCB) formula in MCTS.

    # Discount factor for future rewards: γ
    # Determines the present value of future rewards in the Bellman equation.
    discount = 0.997  # γ ∈ [0, 1)
    # Reference: Used in the recursive calculation of the value function.

    # ====================
    # Training Parameters
    # ====================

    # Learning rate for the optimizer: α
    # Governs the step size during gradient descent optimization.
    learning_rate = 1e-3  # α ∈ ℝ^+
    # Reference: Affects the convergence rate of the optimization algorithm.

    # Number of training iterations: T
    # Specifies how many epochs the training loop will run.
    num_iterations = 1000  # T ∈ ℕ
    # Reference: Total number of times the training loop is executed.

    # Batch size for training: B
    # Number of samples per gradient update.
    batch_size = 64  # B ∈ ℕ
    # Reference: Number of trajectories sampled from the replay buffer per training step.

    # Replay buffer size: |D|
    # Maximum number of game experiences stored for training.
    replay_buffer_size = 10000  # |D| ∈ ℕ
    # Reference: Size of the dataset used for experience replay.

    # Temporal Difference steps: n
    # Number of steps to look ahead in TD learning.
    td_steps = 10  # n ∈ ℕ
    # Reference: Determines the n-step return used for value targets.

    # ====================
    # Neural Network Architecture Parameters
    # ====================

    # Hidden layer size for representation network: |h|
    # Dimensionality of the hidden state vector.
    hidden_size = 256  # |h| ∈ ℕ
    # Reference: Dimension of the latent space used in the networks.

    # Number of actions: |A|
    # Total discrete actions available to the agent.
    self.action_space_size = 10  # |A| ∈ ℕ
    # Reference: Depends on the environment's action space.

    # Action embedding size
    # Size of the action embedding vector used in dynamics network.
    self.action_embedding_size = 6  # ∈ ℕ

    # Observation channels
    # Number of channels in the observation.
    observation_channels = 3  # ∈ ℕ
    # Depends on the environment (e.g., RGB images would have 3 channels).

    # Size of the board or input dimensions
    board_size = 8  # ∈ ℕ
    # Reference: Height and width of the input observations (e.g., an 8x8 board).

    # ====================
    # Optimization Parameters
    # ====================

    # Gradient clipping threshold: θ_{clip}
    # Prevents exploding gradients during training.
    gradient_clip = 0.5  # θ_{clip} ∈ ℝ^+
    # Reference: Stabilizes training by limiting the magnitude of gradients.

    # ====================
    # Miscellaneous Parameters
    # ====================

    # Seed for reproducibility: σ
    # Ensures deterministic behavior across runs.
    seed = 42  # σ ∈ ℕ
    # Reference: For reproducibility of experiments.

    # Checkpoint interval
    # How often to save the model during training.
    checkpoint_interval = 100  # Save model every 100 iterations.
