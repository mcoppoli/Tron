# TRON

This is a modified version of a Tron-41 bot that intelligently chooses the maximizing move based on a 
game state. Gamesare played in the terminal. Code is written in python3. Run simulation with the command 
```python3 gamerunner.py```.


## Overview
The bot begins by taking in the current state of the game in the form of an asp (TronProblem), and executes 
the following procedure to determine the maximizing action:

1. Initialize free_space and opponent_free_space
2. Determine weights for free space and power ups
3. Determine max depth to use for alpha-beta cutoff
4. Define evaluation function
5. Run alpha-beta cutoff to find the maximizing action


## Initializing free space variables
The decide method first initializes the free space available to each bot using the method get_free_moves. 
This method takes in the board and a bot’s current position, and uses a modified flood-fill algorithm to 
determine all the free space available to each bot at the current game state.


## Determining weights
The alpha-beta cutoff algorithm uses an evaluation function to determine the reward of each state 
(discussed in greater detail in later section). The function uses of a free space weight and power up 
weights which change based on the relative location to the opponent and power ups. Initially, 
free_space_weighting is .85, power_up_weighting is .05, and armor_weighting is .1. However, if the opponent 
is not reachable (if opponent_loc is not in free_space) or if there are no power ups on the board, I 
redefine these weights so that power_up_weighting becomes 0 and free_space_weighting becomes .9. The way 
in which these weights are used is discussed later.

### Determine max depth used for ab-cut
When implementing the alpha-beta cutoff search algorithm, I face a tradeoff between speed and accuracy. 
By iterating deeper into the search I increase accuracy scores, but lose time and risk a timeout. To 
account for this, I calculate the optimal max depth to use as a function of the total free space avaliable to each player.

Thus with less moves left (closer to a terminal state), I tradeoff speed to find terminal states, or 
states very near terminal, because with so little states left it is unlikely the algo will timeout. 
Otherwise, I do not search as deep, because with so many possible moves, the number of evaluations is 
very large.

The provided asp.evaluate_state only determines the winner in a terminal state. If I had the computing 
power to run every possible move to termination of game, then the eval_func wouldn’t be necessary, but 
this is not the case, so I was required to set a cutoff on the alpha beta search algorithm and run the 
eval_func on the state at the end of the cutoff to approximate the value of that state for the bot. The 
eval_func returns the sum of the following values:

```
free_space_weight: 
  free_space_weight = (free space available to the bot / free space available to both bots)
  return free_space_weight * free_space_weighting

power_up_weight:
  closest_powerup = distance from the bot to closest power up
  opponent_closest_powerup = distance from opponent bot to closest power up
  powerup_weight = 1 - (closest_powerup/ closest_powerup + opponent_closest_powerup)
  return powerup_weight * powerup_weighting

armor_weight:
	if the bot has armor and opponent does not
		armor_weight = 1.0
	if opponent has armor and I do not
		armor_weight = 0.0
	else
		armor_weight = 0.5
	return armor_weight * armor_weighting
```

### Running alpha-beta cutoff to find the maximizing action
The alpha-beta cutoff method contains a private method eval_func which determines the score of a given 
state, passed in to the alpha-beta helper methods get_max_alpha_beta_cutoff and get_min_alpha_beta_cutoff. 
The alpha-beta cutoff function determines the maximizing action.

## Motivations

### Evaluating states based on free space
When playing the game, a bot will lose when it collides with a wall. If I program the bot to only move 
in a valid direction, then it will never move into a wall unless it is out of free space. Thus, it is 
intuitive and required to evaluate states based on the amount of free space available to the bot. Regardless 
of the location of the opponent, distance from powerups, whether or not we have armor, etc., free space is 
the most important factor when deciding the quality of a given state.

### Evaluating states based on distance to power up
We include a small weight for the distance to the nearest power up relative to the opponent, so that if we 
have a choice of moves where all moves result in having close to the same ratio of free space available as 
the opponent, I want my bot to move to a position that is closer to a power up. Thus the bot will still 
prioritize free space, but if it has the choice to move towards a power up with all other things (relatively) 
equal, it will choose to do so.

### Evaluating states based on armor
Armor is a valuable powerup to possess as it prevents loss from colliding into a wall. Therefore, if in 
a given state I have armor and the opponent does not, the probability of my win is more likely and the 
evaluation of that state should be higher, and thus we chose a weight of 1.0. Conversely, if the opponent 
has armor and I do not, I am less likely to win and the evaluation of that state should be lower and I chose 
a weight of 0.0. If both players have armor or do not have armor, neither player is more or less likely to 
win, and the armor weight is 0.5.

### Alpha-beta cutoff and max depth
Alpha beta cutoff is one of the most versatile and efficient algorithms for searching for optimal game states 
in an adversarial search problem. Because efficiency is a high priority, I used alpha-beta pruning to limit 
the number of states we expand, and a cutoff to prevent the algorithm from spending too much time searching 
far into the future. In early stages of the game, the cutoff is set to 5 generations, but as the total amount 
of free space and the total number of possible moves left in the game decreases, I increase this cutoff to 
encounter more terminal states and return a more accurate and informed decision.
