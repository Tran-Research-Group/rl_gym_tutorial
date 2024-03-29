# rl_gym_tutorial

Tutorial for training RL agents using Stable Baseline3 in Farama Foundation's Gymnasium based environments. Gymnasium provides a standardized way of implementing environments for use in RL experiments and has been widely adopted in the community. Stable Baselines3 is a widely used repository containing state of the art implementations of popular RL algorithms. It allows one to quickly pick and run experiments with algorithm of choice on environments of interest.

# Setup

It is recommended to use a dedicated virtual environment for installing packages and running files for this tutorial. 

Please install the Gymnasium package by running `pip install gymnasium`. Visit the [the official GitHub repository](https://github.com/Farama-Foundation/Gymnasium) for more details.
Please install the Stable Baselines3 package by running `pip install stable-baselines3[extra]`. Refer to [the official documentation](https://stable-baselines3.readthedocs.io/en/master/guide/install.html) for more details.

# Plan

1. We will go over the code for base environment class, which is utilized for implementing Gymnasium environments.
2. We will understand the high-level RL training loop.
3. We will learn how to train an RL agent in a natively available Gymansium environments using an algorithm from Stable Baselines3.
4. We will learn how to visualize the results of the experiment.
