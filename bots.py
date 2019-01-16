import numpy as np
from tronproblem import *
from trontypes import CellType, PowerupType
import random, math


class UserBot:

    def __init__(self):
        self.opposite_moves = {'U':'D', 'D':'U', 'L':'R', 'R':'L', 'NA':'NA'}
        self.opposite_player = {0:1, 1:0, '1':'0', '0':'1'}
        self.free_space = []
        self.free_space_weighting = 0.75
        self.power_up_weighting = 0.1
        self.armor_weighting = 0.15
        self.speed_weighting = 0.0

    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}
        """

        #Initialize game state variables
        state = asp.get_start_state()
        player_locs = state.player_locs
        board = state.board
        self.ptm = state.ptm
        self.initial_power_up_positions = UserBot.get_power_up_positions(board)

        # locations of bots
        player_loc = player_locs[self.ptm]
        opponent_loc = player_locs[self.opposite_player[self.ptm]]

        # free space avaliable to bots
        self.free_space = UserBot.get_free_moves(board, player_loc)
        self.opponent_free_space = UserBot.get_free_moves(board, opponent_loc)

        # is opponent in our free space?
        self.opponent_reachable = opponent_loc in self.free_space

        # power up info for bots
        self.power_up_info = self.get_power_up_info(state, player_loc)
        self.opponent_power_up_info = self.get_power_up_info(state, opponent_loc)

        # determine weights based on game state
        if (not self.opponent_reachable) or (not self.initial_power_up_positions):
            self.free_space_weighting = 0.9
            self.power_up_weighting = 0.0
            self.armor_weighting = 0.1
            self.speed_weighting = 0.0

        # determine max depth of ab-cutoff based on free space avaliable
        if len(self.free_space) + len(self.opponent_free_space) < 100:
            max_depth = 7
        elif len(self.free_space) + len(self.opponent_free_space) < 250:
            max_depth = 6
        else:
            max_depth = 5

        #choose and return an action
        return self.alpha_beta_cutoff(asp, max_depth)

    @staticmethod
    def get_power_up_positions(board):
        """
        This static method takes in a board and returns locations of all powerups
        in the board by checking if each element is a member of powerup_list
        """
        power_up_positions= []
        height = len(board)
        width = len(board[0])
        for r in range(height):
            for c in range(width):
                char = board[r][c]
                if char in CellType.powerup_list:
                    power_up_positions.append((r,c))
        return power_up_positions

    def get_power_up_info(self, state, player_loc):
        """
        This method iterates through initial_power_up_locations and returns
        a list of each remaining powerup's type, location and distance from bot,
        sorted by distance from bot
        """
        # initialize information about the current game state
        board = state.board
        player = int(board[player_loc[0]][player_loc[1]])-1
        player_armor = state.player_has_armor(player)
        power_up_symbols = list(CellType.powerup_list)

        # if we already have a powerup, do not consider that powerup
        if player_armor:
            power_up_symbols.remove(CellType.ARMOR)
        if not self.opponent_reachable:
            power_up_symbols.remove(CellType.SPEED)

        power_up_info = []
        for position in self.initial_power_up_positions:
            r = position[0]
            c = position[1]
            char = board[r][c]
            if char in power_up_symbols: #if powerup still there
                power_up_loc = (r,c)
                distance = UserBot.get_distance(power_up_loc, player_loc) #append info to list
                power_up_info.append({'Type': char, 'Location': power_up_loc, 'Distance': distance})

        power_up_info.sort(key=lambda x: x['Distance']) #sort by distance
        return power_up_info

    def get_power_up_weight(self):
        """
        This method calculates the advantage that our bot has over the opponent
        based on distance to the closest powerup. The higher the weight, the closer
        we are to a powerup than our opponent and the better the state is for us.
        If there are no reachable powerups for either us or our opponent, return
        an equal weight of 0.5
        """
        if self.power_up_info and self.opponent_power_up_info:
            closest_power_up = self.power_up_info[0]['Distance']
            opponent_closest_power_up = self.opponent_power_up_info[0]['Distance']
            power_up_weight =  1 - (closest_power_up/(closest_power_up+opponent_closest_power_up))
            return power_up_weight * self.power_up_weighting
        else:
            return 0.5

    @staticmethod
    def get_distance(loc1, loc2):
        """
        Takes in 2 locations and returns the manhattan distance between them
        """
        x = abs(loc1[1] - loc2[1])
        y = abs(loc1[0] - loc2[0])
        return x + y

    @staticmethod
    def get_empty_neighbors(board, position, max_height, max_width):
        """
        This method takes in a board and a position and finds the adjacent empty
        neighbors of the position, and returns them in a list
        """
        neighbors = []
        height = position[0] #position: (y, x)
        width = position[1]
        barriers = ['x', '#', '1', '2']
        up = position[0] - 1
        down = position[0] + 1
        left = position[1] - 1
        right = position[1] + 1
        if up > 0 and board[up][width] not in barriers:
            neighbors.append((up,width))
        if down < max_height and board[down][width] not in barriers:
            neighbors.append((down,width))
        if left > 0 and board[height][left] not in barriers:
            neighbors.append((height,left))
        if right < max_width and board[height][right] not in barriers:
            neighbors.append((height,right))
        return neighbors

    @staticmethod
    def get_free_moves(board, current_position):
        """
        This method uses flood fill (bfs) to traverse the board and add all
        spaces that can be moved into to the past_moves set, which is returned as
        a list
        """
        max_width = len(board[0])
        max_height = len(board)
        past_moves = set()
        new_moves = set()
        new_moves.add(current_position)
        while len(new_moves)>0:
            position = new_moves.pop()
            past_moves.add(position)
            for new_position in UserBot.get_empty_neighbors(board, position, max_height, max_width):
                if new_position not in past_moves:
                    new_moves.add(new_position)
        return list(past_moves)

    def get_free_space_weight(self):
        """
        Calculates and returns a free space weight based on the ratio of our
        bot's free space to the opponent bot's free space
        """
        player_num_moves = len(self.free_space)
        opponent_num_moves = len(self.opponent_free_space)
        free_space_weight = player_num_moves/(player_num_moves+opponent_num_moves)
        return free_space_weight * self.free_space_weighting

    def get_armor_weight(self, state):
        """
        Calculates and returns a weight based on if we have armor and if our
        opponent has armor
        """
        player_armor = state.player_has_armor(self.ptm)
        opponent_armor = state.player_has_armor(self.opposite_player[self.ptm])
        if player_armor and not opponent_armor:
            armor_weight = 1.0
        elif not player_armor and opponent_armor:
            armor_weight = 0.0
        else:
            armor_weight = 0.5
        return armor_weight * self.armor_weighting

    def alpha_beta_cutoff(self, asp, cutoff_ply):
        """ Alpha beta cutoff algorithm to find the maximizing action """

        def eval_func(asp):
            """
            This method is used to evaluate an asp. It calculates and sums
            free_space_weight, power_up_weight and armor_weight
            """

            #init board and player locations
            state = asp.get_start_state()
            board = state.board
            player_locs = state.player_locs
            player_loc = player_locs[self.ptm]
            opponent_loc = player_locs[self.opposite_player[self.ptm]]

            # store free space for each bot
            self.free_space = UserBot.get_free_moves(board, player_loc)
            self.opponent_free_space = UserBot.get_free_moves(board, opponent_loc)

            #store if opponent is reachable
            self.opponent_reachable = opponent_loc in self.free_space

            #store power up info for both bots
            self.power_up_info = self.get_power_up_info(state, player_loc)
            self.opponent_power_up_info = self.get_power_up_info(state, opponent_loc)

            # store, sum, and return each weight
            free_space_weight = self.get_free_space_weight()
            power_up_weight = self.get_power_up_weight()
            armor_weight = self.get_armor_weight(state)
            return free_space_weight + power_up_weight + armor_weight

        # alpha beta cutoff algorithm
        max_val = -np.inf
        original_problem_state = asp.get_start_state()
        original_ptm = original_problem_state.ptm
        for action in {U, D, L, R}:
            asp.set_start_state(asp.transition(original_problem_state, action))
            state = asp.get_start_state()
            if original_ptm == state.ptm:
                current_val = self.get_max_alpha_beta_cutoff(asp, -np.inf, np.inf, 1, cutoff_ply, eval_func)
            else:
                current_val = self.get_min_alpha_beta_cutoff(asp, -np.inf, np.inf, 1, cutoff_ply, eval_func)
            asp.set_start_state(original_problem_state)
            if current_val >= max_val:
                next_action = action
                max_val = current_val
        return next_action

    #alpha beta helper to find maximizing action
    def get_max_alpha_beta_cutoff(self, asp, alpha, beta, depth, max_depth, eval_func):
        if asp.is_terminal_state(asp.get_start_state()):
            return asp.evaluate_state(asp.get_start_state())[self.ptm]
        elif depth == max_depth:
            return eval_func(asp)
        max_val = -np.inf
        original_problem_state = asp.get_start_state()
        original_ptm = original_problem_state.ptm
        for action in {U, D, L, R}:
            asp.set_start_state(asp.transition(original_problem_state, action))
            state = asp.get_start_state()
            if original_ptm == state.ptm:
                max_val = max(max_val, self.get_max_alpha_beta_cutoff(asp, alpha, beta, depth+1, max_depth, eval_func))
            else:
                max_val = max(max_val, self.get_min_alpha_beta_cutoff(asp, alpha, beta, depth+1, max_depth, eval_func))
            asp.set_start_state(original_problem_state)
            if max_val >= beta: #prune
                return max_val
            alpha = min(alpha, max_val)
        return max_val

    #alpha beta helper to find minimizing action
    def get_min_alpha_beta_cutoff(self, asp, alpha, beta, depth, max_depth, eval_func):
        if asp.is_terminal_state(asp.get_start_state()):
            return asp.evaluate_state(asp.get_start_state())[self.ptm]
        elif depth == max_depth:
            return eval_func(asp)
        min_val = np.inf
        original_problem_state = asp.get_start_state()
        original_ptm = original_problem_state.ptm
        for action in {U, D, L, R}:
            asp.set_start_state(asp.transition(original_problem_state, action))
            state = asp.get_start_state()
            if original_ptm == state.ptm:
                min_val = min(min_val, self.get_min_alpha_beta_cutoff(asp, alpha, beta, depth+1, max_depth, eval_func))
            else:
                min_val = min(min_val, self.get_max_alpha_beta_cutoff(asp, alpha, beta, depth+1, max_depth, eval_func))
            asp.set_start_state(original_problem_state)
            if min_val <= alpha: #prune
                return min_val
            beta = min(beta, min_val)
        return min_val

    def cleanup(self):
        """
        Input: None
        Output: None

        This function resets variables the bot uses during the game,
        called between games.
        """
        self.free_space_weighting = 0.75
        self.power_up_weighting = 0.1
        self.armor_weighting = 0.15
        self.speed_weighting = 0.0

    #end UserBot

class RandBot:
    """
    A trivial bot used for testing which moves in a safe random direction
    """

    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}
        """
        state = asp.get_start_state()
        locs = state.player_locs
        board = state.board
        ptm = state.ptm
        loc = locs[ptm]
        possibilities = list(TronProblem.get_safe_actions(board, loc))
        if possibilities:
            return random.choice(possibilities)
        return "U"

    def cleanup(self):
        pass


class WallBot:
    """
    A trivial bot used for testing which always hugs the wall
    """

    def __init__(self):
        order = ["U", "D", "L", "R"]
        random.shuffle(order)
        self.order = order

    def cleanup(self):
        order = ["U", "D", "L", "R"]
        random.shuffle(order)
        self.order = order

    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}
        """
        state = asp.get_start_state()
        locs = state.player_locs
        board = state.board
        ptm = state.ptm
        loc = locs[ptm]
        possibilities = list(TronProblem.get_safe_actions(board, loc))
        if not possibilities:
            return "U"
        decision = possibilities[0]
        for move in self.order:
            if move not in possibilities:
                continue
            next_loc = TronProblem.move(loc, move)
            if len(TronProblem.get_safe_actions(board, next_loc)) < 3:
                decision = move
                break
        return decision
