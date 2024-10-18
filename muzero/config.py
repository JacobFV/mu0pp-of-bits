from dataclasses import dataclass
from torchtyping import TensorType

@dataclass
class MuZeroConfig:
    # ====================
    # Monte Carlo Tree Search (MCTS) Parameters
    # ====================

    # Number of simulations per MCTS search: K
    # Controls the depth and breadth of the search tree.
    num_simulations: int = 800  # K ∈ ℕ
    # Reference: Determines the number of iterations in MCTS for policy improvement.

    # Exploration constant in the UCB formula: c_{puct}
    # Balances exploration vs. exploitation in selecting actions during MCTS.
    c_puct: float = 1.25  # c_{puct} ∈ ℝ^+
    # Reference: Upper Confidence Bound (UCB) formula in MCTS.

    # Discount factor for future rewards: γ
    # Determines the present value of future rewards in the Bellman equation.
    discount: float = 0.997  # γ ∈ [0, 1)
    # Reference: Used in the recursive calculation of the value function.

    # ====================
    # Training Parameters
    # ====================

    # Learning rate for the optimizer: α
    # Governs the step size during gradient descent optimization.
    learning_rate: float = 1e-3  # α ∈ ℝ^+
    # Reference: Affects the convergence rate of the optimization algorithm.

    # Number of training iterations: T
    # Specifies how many epochs the training loop will run.
    num_iterations: int = 1000  # T ∈ ℕ
    # Reference: Total number of times the training loop is executed.

    # Batch size for training: B
    # Number of samples per gradient update.
    batch_size: int = 64  # B ∈ ℕ
    # Reference: Number of trajectories sampled from the replay buffer per training step.

    # Replay buffer size: |D|
    # Maximum number of game experiences stored for training.
    replay_buffer_size: int = 10000  # |D| ∈ ℕ
    # Reference: Size of the dataset used for experience replay.

    # Temporal Difference steps: n
    # Number of steps to look ahead in TD learning.
    td_steps: int = 10  # n ∈ ℕ
    # Reference: Determines the n-step return used for value targets.

    # ====================
    # Neural Network Architecture Parameters
    # ====================

    # Hidden layer size for representation network: |h|
    # Dimensionality of the hidden state vector.
    hidden_size: int = 256  # |h| ∈ ℕ
    # Reference: Dimension of the latent space used in the networks.

    # Number of actions: |A|
    # Total discrete actions available to the agent.
    action_space_size: int = 10  # |A| ∈ ℕ
    # Reference: Depends on the environment's action space.

    # Action embedding size
    # Size of the action embedding vector used in dynamics network.
    action_embedding_size: int = 6  # ∈ ℕ

    # Observation channels
    # Number of channels in the observation.
    observation_channels: int = 3  # ∈ ℕ
    # Depends on the environment (e.g., RGB images would have 3 channels).

    # Size of the board or input dimensions
    board_size: int = 8  # ∈ ℕ
    # Reference: Height and width of the input observations (e.g., an 8x8 board).

    # Environment-specific parameters
    action_space_size: int = None
    observation_shape: tuple = None

    # ====================
    # Optimization Parameters
    # ====================

    # Gradient clipping threshold: θ_{clip}
    # Prevents exploding gradients during training.
    gradient_clip: float = 0.5  # θ_{clip} ∈ ℝ^+
    # Reference: Stabilizes training by limiting the magnitude of gradients.

    # ====================
    # Miscellaneous Parameters
    # ====================

    # Seed for reproducibility: σ
    # Ensures deterministic behavior across runs.
    seed: int = 42  # σ ∈ ℕ
    # Reference: For reproducibility of experiments.

    # Checkpoint interval
    # How often to save the model during training.
    checkpoint_interval: int = 100  # Save model every 100 iterations.

    # Parameters for image observations
    image_channels: int = 3  # Number of channels in the image (e.g., 3 for RGB)
    image_height: int = 84   # Height of the image
    image_width: int = 84    # Width of the image

    # Parameters for vector observations
    vector_observation_size: int = 4  # Size of the vector observation

    # Hidden size for the combined features
    hidden_size: int = 256  # Size of the hidden representation

    # Action space size
    action_space_size: int = None  # To be set based on the environment

    # Action embedding size
    action_embedding_size: int = 6  # Size of the action embedding vector
