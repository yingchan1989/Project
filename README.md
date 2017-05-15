# Project: Deep Reinforcement Learning for Gomoku (five-in-a-row)

The aim of this project is to create a deep reinforcement learning model, specifically using a off-policy deep learning algorithm such as Q-learning, coupled with neural networks (and also convolutional neural networks), in order to obtain artificial intelligence to play a game of Gomoku (five-in-a-row).

There are three flavors of the algorithm, each exploring alternative methods of training this AI.
1. game.py uses tradional neural networks and applies a two-dimensional action space (attack and defend). The policy is extensively defined, only letting the neural network to determine whether it should block or attack. 
2. multi-action.py also uses tradional neural networks but applies a multi-dimensional action space (place stones on any row or column that has an adjacent space to the left or right, including the opponents). The policy is hence not defined, other than the exception that moves are typically made next to adjacent pieces.
3. cnn.py (coming soon) uses convolutional neural networks to determine the board game state without feature engineering per se. It will use a set of kernels to visualize the board game state and use this to determine the appropriate action. 

## Training reinforcement learning
The algorithms also explore various methods of training the reinforcement learning neural network model. The different opposing players that are used to train (sometimes done progressively) include:
1. Random player that places a stone in any random available place
2. Random action player that selects a random action per defined in the policy network
3. Self play where the board game is switched around and the algorithm plays with itself to train

In addition to progressive training, an epsilon-greedy algorithm is put in place whereby the epsilon value will decline over time, favoring trained networks for predicted actions. The epsilon declines linearly. 

## Feature engineering
The first two algorithms utilize traditonal neural networks and hence feature enginneering was used extensively to summarize the board game states.

Feature engineering items include:
1. Number of consecutively placed stones horizontally, vertically or diagonally of the current player and the opposing player
2. Left and right available spaces of consecutively placed stones

The game state space (for the first two flavors of the algorithms) are flipped when the game is self-played, allowing for features of the opposing player to remain the same. Feature engineering is currently not used for CNN.

## Action state space
The action state space differs for each of the algorithm flavors and can be tailored. 
1. game.py uses two action state spaces, where the policy network is defined to be attack or defend on the maximum column/row/diagonal linked stones. A random action is placed based on the epsilon-greedy algorithm.
2. multi-action.py does not define the policy network explicitly, and allows full flexibility in filling in any available adjacent spaces on a column/row/diagonal.
3. cnn.py follows the action state space in multi-action.py. 

## Reward state space
A win is typically awarded an arbitary 1000 points, while a loss is awarded -1000. A tweak to the reward was put in place when the player is to play by itself, since if the board game is switched around, and the player plays the winning move, the move prior that caused this winning move is never punished. Hence the punishment occurs when the player fails to block an available space with 4 pieces lined up.

Two reward functions were tested in this project:
1. A win is rewarded with 1000 points, and a loss is -1000. A draw is given 10.
2. A win is rewarded with 1000 points, and a loss is -1000. A draw is however given 0.


## Performance and results
Each model is benchmarked against two standards to determine their performance.
1. Random player which plays on a random space
2. Random action player which plays a random action (where the action is determined by the model). 

The performance test is done on a 5x5 grid, and is performed after training for 1000 games. 100 games is played against each benchmark and the results tallied.

The game.py model produced the following results (Benchmark1/Benchmark2):
Wins: 81 / 21
Loss: 9  / 41
Draw: 10 / 38

Game.py showed exceptional performance when pitched against benchmark 1, but this is expected since a lot of the policy network was coded. However, when done against benchmark 2, it showed that a lot of the performance derived from the hard coded policy network versus the neural network. Furthermore, results showed that when the model is trained against itself, draws are created exceptionally often during the training rounds, hence it appears to hve learnt how to draw versus winning. 

The multi-action.py model produced the following results (Benchmark1/Benchmark2):
Wins: 31 / 36
Loss: 11 / 8
Draw: 58 / 56

When the hardcoding of policy networks were removed in the multi-action.py model, the results improved. It can be demonstrated that the neural network was able to start cpaturing the key strategy of winning even against a random-action player. It was able to manage a better than 4-to-1 win/loss ratio. It is useful to note that draws occur often, and it is expected as this is done on a 5x5 grid, with a winning criteria of 5-in-a-row, meaning that any piece placed on the row (when placed on a row, it will also be placed on a column), that row and column cannot be won.

## Extensibility
The algorithm is extensible to any sized game board. The code is tested using a 5x5 board game with the requirement of five-in-a-row to win. A working prototype for the 19x19 board game currently powers the browser extension to play Gomoku on BoardGameArena.com. A link to download this plugin is currently available here: 

