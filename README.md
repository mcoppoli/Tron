# TRON

This is a modified version of a Tron-41 bot that intelligently chooses the maximizing move based on a
game state. Games are played in the terminal. Code is written in python3. Run simulation with the command
```python3 gamerunner.py```.


## Overview
The bot begins by taking in the current state of the game in the form of an asp (TronProblem), and executes
the following procedure to determine the maximizing action:

1. Initialize free_space and opponent_free_space
2. Determine weights for free space and power ups
3. Determine max depth to use for alpha-beta cutoff
4. Define evaluation function
5. Run alpha-beta cutoff to find the maximizing action

A more detailed writeup of the decision making process can be found in README
