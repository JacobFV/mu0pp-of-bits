import gym
from muzero.agent import MuZeroAgent
from muzero.config import MuZeroConfig


if __name__ == "__main__":
    # Initialize the gym environment
    env = gym.make('CartPole-v1')

    # Update the configuration with environment-specific information
    config = MuZeroConfig(
        action_space_size=env.action_space.n,
        observation_shape=env.observation_space.shape,
        # Include any other necessary configurations
    )

    # Initialize the agent with the environment
    agent = MuZeroAgent(config, environment=env)

    # Start the training process
    agent.train()
