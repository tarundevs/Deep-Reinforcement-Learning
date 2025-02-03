# Tic-Tac-Toe Q-Learning Agent

This project implements a reinforcement learning agent that plays the game of Tic-Tac-Toe using Q-learning. The agent is trained to maximize its number of wins and can play against either a hardcoded opponent or another AI.

## Table of Contents
- [Introduction](#introduction)
- [What It Uses](#what-it-uses)
- [Dependencies](#dependencies)

## Introduction

This project simulates a game of Tic-Tac-Toe between two players, one of whom can be controlled by an AI agent using Q-learning. The agent is designed to maximize its performance by learning from the outcomes of the games. It uses both hardcoded strategies (with varying levels of smartness) and a Q-learning model for decision-making. The project also includes functionality for multiprocessing, which allows the agent to play multiple games concurrently during training and testing.

## What It Uses

The project leverages several techniques and technologies to function effectively:

1. **Q-Learning**: The agent uses Q-learning, a type of reinforcement learning, to learn optimal strategies for playing Tic-Tac-Toe. It updates its Q-values based on the reward from each game and iteratively improves its decision-making process.

2. **Hardcoded Strategies**: Along with Q-learning, the agent can use hardcoded strategies for making moves. This allows for different difficulty levels and comparisons between human and AI performance.

3. **Multiprocessing**: The project includes multiprocessing capabilities to allow training and testing across multiple game episodes simultaneously. This reduces the time required to train the agent.

4. **TensorFlow**: The project uses TensorFlow for efficient computation and model training. The reinforcement learning agent's training process is accelerated using TensorFlow's backend.

5. **NumPy**: The agent and environment are built using NumPy for handling arrays, performing matrix operations, and managing game state.

6. **Random and Smart Opponents**: The agent can play against a random player or a more intelligent, hardcoded player to simulate different scenarios and test its performance.

7. **Game Simulation**: The game environment itself is modeled as a class, with features to check for valid moves, determine if the game is over, and print the board's current state.

## Dependencies

The project requires the following Python libraries:
- `numpy`
- `random`
- `os`
- `warnings`
- `matplotlib`
- `tensorflow`
- `tqdm`
