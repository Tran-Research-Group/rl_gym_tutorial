import gymnasium as gym
from stable_baselines3 import PPO
import torch

# Define configurations
env_name: str = "CartPole-v1"

# PPO hyperparameters
total_timesteps: int = 10_000
learning_rate: float = 0.0003
n_steps: int = 2048
batch_size: int = 64
n_epochs: int = 10
gamma: float = 0.99

# Paths to save the model, logs, and other files
model_path: str = f"out/models/{env_name}_ppo"
tensorboard_log_dir: str = f"out/log/"
animation_path: str = f"out/animations/{env_name}_ppo"


# Check if the GPU is available
print("CUDA is available: ", torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Create an environment (cartpole-v1)
env = gym.make("CartPole-v1")

model = PPO("MlpPolicy", env, verbose=1)
