# CartPole Reinforcement Learning 
## ML Project - Fosci 1855832 - Martino 

This repository contains implementations of **Tabular Q-Learning** and **Deep Q-Network (DQN)** to solve the CartPole environment from [OpenAI Gym](https://www.gymlibrary.dev/). The goal is to train an agent to balance a pole on a cart by applying reinforcement learning algorithms.

## Introduction

Reinforcement Learning (RL) is a machine learning paradigm where agents learn optimal actions through trial and error by interacting with an environment. This project implements two RL algorithms:

1. **Tabular Q-Learning**: A basic algorithm that uses a Q-table to store state-action values.
2. **Deep Q-Network (DQN)**: A neural network-based approach to approximate Q-values for continuous state spaces.

---

## Algorithms Implemented

### 1. Tabular Q-Learning
- Discretizes the CartPole state space into bins.
- Updates Q-values using the Bellman equation:
- Suitable for small, discrete state-action spaces.

### 2. Deep Q-Network (DQN)
- Uses a neural network to approximate Q-values for a continuous state space.
- Implements key techniques:
  - **Experience Replay**: Stores transitions in a replay buffer for stable learning.
  - **Target Network**: Updates target Q-values using a separate network to improve stability.
- Loss function

---

