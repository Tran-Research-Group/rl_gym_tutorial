import os

import gymnasium as gym
from stable_baselines3 import PPO
import torch
import imageio

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
env = gym.make(env_name, render_mode="rgb_array")

# Create the model

if os.path.exists(model_path):
    model = PPO.load(model_path, env, verbose=1, device=device)
else:
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device=device,
        tensorboard_log=tensorboard_log_dir,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
    )

    # Train the model
    model.learn(total_timesteps=total_timesteps, tb_log_name=env_name)

    # Save the model
    model.save(model_path)

# Create an animation of the trained model
images = []
obs = env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    img = env.render()
    images.append(img)

    if terminated or truncated:
        break

# Save the animation
imageio.mimsave(animation_path, images, fps=30)
