# from train_tdl import TDLearning, board, pattern
import random
import numpy as np

import sys
import math
import random
import struct
import typing
import abc
import time
from train_tdl import TDLearning, board, pattern
from Game2048Env import Game2048Env

import copy
import random
import math
import numpy as np

# Note: This MCTS implementation is almost identical to the previous one,
# except for the rollout phase, which now incorporates the approximator.

# Node for TD-MCTS using the TD-trained value approximator
class TD_MCTS_Node:
    def __init__(self, state, score, parent=None, action=None, env=None, type="estimate"):
        """
        Represents a node in the Monte Carlo Tree Search
        
        Parameters:
        - state: current board state (numpy array)
        - score: cumulative game score at this node
        - parent: parent node (None for root)
        - action: action taken from parent to reach this node
        - env: game environment for simulating moves
        - type: "estimate" for player move nodes, "explore" for chance nodes (tile spawns)
        """
        self.type = type        # estimate or explore
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action
        self.children = {}      # Maps actions to child nodes
        self.visits = 0         # Number of times this node has been visited
        self.total_reward = 0.0 # Cumulative rewards from all visits
        self.valid_moves = {}   # Maps valid moves to (resulting_state, resulting_score)
        self.expanded = False   # Flag for explore nodes to track expansion status
        
        # Compute valid moves from this state if environment is provided
        if env is not None:
            for move in range(4):  # Check all 4 possible moves (up, right, down, left)
                sim_env = copy.deepcopy(env)
                sim_env.board = state.copy()
                sim_env.score = score
                next_state, next_score, done, _ = sim_env.step(move, evaluate=True)
                # Only consider moves that change the board state
                if not np.array_equal(self.state, next_state):
                    self.valid_moves[move] = (next_state, next_score)

        self.unexplored_moves = list(self.valid_moves.keys())

    def fully_expanded(self):
        """
        Check if all possible actions from this node have been tried.
        Different criteria based on node type:
        - estimate nodes: all valid moves have been tried
        - explore nodes: single expansion flag (since all tile placements happen at once)
        """
        if self.type == "estimate":
            return self.valid_moves and all(move in self.children for move in self.valid_moves)
        elif self.type == "explore":
            return not self.expanded
    
    def is_leaf(self, empty_tiles):
        """
        Determine if this node is a leaf node in the search tree
        - estimate nodes: not fully expanded
        - explore nodes: all possible tile placements have been considered
        """
        if self.type == "estimate":
            return not self.fully_expanded()
        elif self.type == "explore":
            return len(self.children) == len(empty_tiles) * 2  # 2 possible tile values (2, 4)

class TD_MCTS:
    def __init__(self, env, tdl, iterations=500, exploration_constant=1.41, rollout_depth=10, gamma=0.99):
        """
        Monte Carlo Tree Search algorithm with TD Learning value approximation
        
        Parameters:
        - env: Game environment
        - tdl: Temporal Difference Learning model for state evaluation
        - iterations: Number of simulations to run for each action selection
        - exploration_constant: Controls exploration vs exploitation (UCT formula)
        - rollout_depth: Maximum depth for rollout policy
        - gamma: Discount factor for future rewards
        """
        self.env = env
        self.tdl = tdl
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma

    def state_to_board(self, state):
        """
        Convert numpy 2D array state to the board representation used by TDL
        Maps tile values (2,4,8...) to power-of-2 indices (1,2,3...)
        """
        b = board()
        for i in range(4):
            for j in range(4):
                value = state[i][j]
                if value > 0:
                    b.set(i*4+j, int(np.log2(value)))
                else:
                    b.set(i*4+j, 0)
        return b

    def create_env_from_state(self, state, score):
        """Create a deep copy of the environment with the given state and score."""
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def select_child(self, node):
        """
        Select the best child node using the UCT formula:
        Q(v') + c * sqrt(ln(N(v)) / N(v'))
        
        Where:
        - Q(v') is the average reward of child node
        - N(v) is the visit count of parent node
        - N(v') is the visit count of child node
        - c is the exploration constant
        """
        best_score = float('-inf')
        best_action = None
        best_child = None
        
        for action, child in node.children.items():
            uct_score = 0

            # For unvisited nodes, use TD approximator value directly
            if child.visits == 0:
                uct_score = self.tdl.estimate(self.state_to_board(child.state))
            else:
                # UCT formula balancing exploitation and exploration
                exploitation = child.total_reward / child.visits if child.visits > 0 else 0
                exploration = self.c * math.sqrt(math.log(node.visits) / child.visits) if child.visits > 0 else float('inf')
                uct_score = exploitation + exploration
            
            if uct_score > best_score:
                best_score = uct_score
                best_action = action
                best_child = child
                
        return best_action, best_child

    def evaluate(self, node, sim_env, total_reward):
        """
        Evaluate the value of a node using the TD-Learning approximator.
        
        Different evaluation strategies based on node type:
        - estimate: find the best move using the approximator
        - explore: directly use the approximator value
        
        Returns total accumulated reward plus estimated state value
        """
        state_value = 0

        if (node.type == "estimate"):
            # For player move nodes, evaluate all possible moves
            tmp_node = TD_MCTS_Node(sim_env.board.copy(), sim_env.score, env=sim_env, type="estimate")
            if not tmp_node.valid_moves:
                return 0  # Game over, no valid moves
            
            # Find the best move based on immediate reward + estimated future value
            best_value = float('-inf')
            for action, (state, new_score) in tmp_node.valid_moves.items():
                mv_reward = new_score - sim_env.score  # Immediate reward for this move
                state_value = mv_reward + self.tdl.estimate(self.state_to_board(state))
                best_value = max(best_value, state_value)

            state_value = best_value
        elif (node.type == "explore"):
            # For chance nodes (after tile placement), directly estimate board value
            state_value = self.tdl.estimate(self.state_to_board(node.state))
        else:
            state_value = 0

        return total_reward + state_value

    def backpropagate(self, node, reward):
        """
        Propagate the reward up the tree, updating visit counts and total rewards
        for all nodes from the evaluated leaf up to the root
        """
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def run_simulation(self, root):
        """
        Run one complete MCTS simulation from the root node.
        
        This consists of four phases:
        1. Selection: Navigate down the tree using UCT until reaching a leaf
        2. Expansion: Add a new node to the tree
        3. Evaluation: Estimate the value of the new node using TD-Learning
        4. Backpropagation: Update statistics up the tree
        """
        node = root
        sim_env = self.create_env_from_state(node.state, node.score)

        # Selection: Navigate down the tree until reaching a leaf node
        total_reward = 0
        
        while not node.is_leaf(empty_tiles=[]):
            if node.type == "estimate":
                # For player move nodes, select best action using UCT
                action, _ = self.select_child(node)
                prev_reward = sim_env.score
                _, reward, _, _ = sim_env.step(action, evaluate=True)
                move_reward = reward - prev_reward
                total_reward += move_reward

                # Create child if it doesn't exist (shouldn't happen during selection phase)
                if action not in node.children:
                    node.children[action] = TD_MCTS_Node(sim_env.board.copy(), sim_env.score, parent=node, action=action, type="explore")
                node = node.children[action]

            elif node.type == "explore":
                # For chance nodes, select tile placement (weighted by probability)
                tile_positions = list(node.children.keys())
                # 90% chance for value 2, 10% chance for value 4
                weights = [0.9 if val == 2 else 0.1 for (_, val) in tile_positions]
                chosen_tile = random.choices(tile_positions, weights=weights, k=1)[0]

                node = node.children[chosen_tile]
                sim_env = self.create_env_from_state(node.state, node.score)

        # Expansion: If the node has untried actions, expand
        if not sim_env.is_game_over():
            if node.type == "estimate" and not node.children:
                # For player move nodes, add children for all valid moves
                for move, (state, new_score) in node.valid_moves.items():
                    child_node = TD_MCTS_Node(state.copy(), new_score, parent=node, action=move, type="explore")
                    node.children[move] = child_node

            elif node.type == "explore" and not node.expanded:
                # For chance nodes, add children for all possible tile placements
                empty_positions = list(zip(*np.where(node.state == 0)))

                for pos in empty_positions:
                    for tile_value in [2, 4]:  # New tiles can be 2 or 4
                        new_board = node.state.copy()
                        new_board[pos] = tile_value
                        tile_key = (pos, tile_value)
                        
                        if tile_key not in node.children:
                            child = TD_MCTS_Node(new_board, node.score, parent=node, action=tile_key, env=self.env, type="estimate")
                            node.children[tile_key] = child

                node.expanded = True

        # Evaluation: Use TD-Learning to estimate the node value
        rollout_reward = self.evaluate(node, sim_env, total_reward)
        
        # Backpropagation: Update the tree with the rollout reward
        self.backpropagate(node, rollout_reward)

    def best_action_distribution(self, root):
        """
        Compute the normalized visit count distribution for each move from the root.
        
        Returns:
        - best_action: The move with highest visit count
        - distribution: Probability distribution over all moves
        """
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)  # 4 possible moves in 2048
        best_visits = -1
        best_action = None
        for action, child in root.children.items():
            distribution[action] = child.visits / total_visits if total_visits > 0 else 0
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        return best_action, distribution



board.lookup.init()
tdl = TDLearning()

# print(tdl)

ntuple_patterns = [
    pattern([0, 4, 8]),                # 3-tuple pattern 048
        pattern([1, 5, 9]),                # 3-tuple pattern 159
        pattern([0, 1, 4, 5]),             # 4-tuple pattern 0145
        pattern([1, 2, 5, 6]),             # 4-tuple pattern 1256
        pattern([5, 6, 9, 10]),            # 4-tuple pattern 569a
        pattern([2, 6, 10, 14]),           # 4-tuple pattern 26ae
        pattern([3, 7, 11, 15]),           # 4-tuple pattern 37bf
        pattern([1, 4, 5, 8, 9]),          # 5-tuple pattern 01245
        pattern([2, 5, 6, 9, 10]),          # 5-tuple pattern 12356
        pattern([3, 6, 7, 10, 11]),        # 5-tuple pattern 23467
        pattern([0, 1, 2, 4, 5]),          # 5-tuple pattern 01245
        pattern([1, 2, 3, 5, 6]),          # 5-tuple pattern 12356
        pattern([0, 1, 5, 6, 7]),          # 5-tuple pattern 01567
        pattern([0, 1, 2, 5, 9]),          # 5-tuple pattern 01259
        pattern([0, 1, 2, 6, 10]),          # 5-tuple pattern 0126a
        pattern([0, 1, 5, 9, 10]),
        pattern([0, 4, 8, 9, 12]),
        pattern([1, 5, 9, 10, 13]),
        pattern([2, 6, 10, 11, 14]),
        pattern([0, 1, 5, 9, 13]),
        pattern([1, 2, 6, 10, 14]),
        pattern([2, 3, 7, 11, 15]),
]

for pattern in ntuple_patterns:
    tdl.add_feature(pattern)
# restore the model from file
tdl.load("2048_stage1.bin")

env = Game2048Env()
td_mcts = TD_MCTS(env, tdl, iterations=30, exploration_constant=1.41, rollout_depth=100, gamma=0.99)


def state_to_board(state):
    b = board()
    for i in range(4):
        for j in range(4):
            value = state[i][j]
            if value > 0:
                b.set(i*4+j, int(np.log2(value)))
            else:
                b.set(i*4+j, 0)
    return b


def get_action(state, score):
    """
    Get the best action for the current state
    """
    env.board = state.copy()
    env.score = score

    root = TD_MCTS_Node(state, score, env=env, type="estimate")

    # Run multiple simulations to build the MCTS tree
    for _ in range(td_mcts.iterations):
        td_mcts.run_simulation(root)

    # Select the best action (based on highest visit count)
    best_act, _ = td_mcts.best_action_distribution(root)

    print("Current score: ", env.score)


    return best_act

