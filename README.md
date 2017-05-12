# Project: Deep Gaming

The aim of this project is to create a deep reinforcement learning model, specifically using a off-policy deep learning algorithm such as Q-learning, coupled with neural networks (and also convolutional neural networks), in order to obtain artificial intelligence to play a game of Gomoku (five-in-a-row).

There are three flavors of the algorithm, each exploring alternative methods of training this AI.
1. game.py uses tradional neural networks and applies a two-dimensional action space (attack and defend). The policy is extensively defined, only letting the neural network to determine whether it should block or attack. 
2. multi-action.py also uses tradional neural networks but applies a multi-dimensional action space (place stones on any row or column that has an adjacent space to the left or right, including the opponents). The policy is hence not defined, other than the exception that moves are typically made next to adjacent pieces.
3. cnn.py (coming soon) uses convolutional neural networks to determine the board game state without feature engineering per se. It will use a set of kernels to visualize the board game state and use this to determine the appropriate action. 

The algorithms also explore various methods of training the reinforcement learning neural network model. The different opposing players that are used to train (sometimes done progressively) include:
1. Random player that places a stone in any random available place
2. Random action player that selects a random action per defined in the policy network
3. Self play where the board game is switched around and the algorithm plays with itself to train

## Feature engineering
The first two algorithms utilize traditonal neural networks and hence feature enginneering was used extensively to summarize the board game states.

Feature engineering items included extracting the columns, rows and diagonals, that house any placement of stones of the current player as well as the opponent player. The maximum number of consecutive placement of stones (both the current player and the opposing player)

## Action state space



## Reward state space
