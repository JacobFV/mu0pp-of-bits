from muzero.agent import MuZeroAgent
from muzero.config import MuZeroConfig


if __name__ == "__main__":
    # Initialize configuration and agent
    config = MuZeroConfig()
    agent = MuZeroAgent(config)
    # Start the training process
    agent.train()
    # Training involves self-play to generate data and updating the network weights
