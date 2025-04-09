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

board.lookup.init()
tdl = TDLearning()

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
tdl.load("2048.bin")

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

# def get_action(state, score):
#     # return random.choice([0, 1, 2, 3]) # Choose a random action

#     b = state_to_board(state)
    
#     # Use your N-Tuple approximator to select the best move
#     best_move = tdl.select_best_move(b).action()

#     action = best_move

#     return action
    
#     # You can submit this random agent to evaluate the performance of a purely random strategy.

# ... existing code ...
def get_action(state, score):
    # return random.choice([0, 1, 2, 3]) # Choose a random action

    b = state_to_board(state)
    
    # Use MCTS with N-tuple weights to select the best move
    best_move = mcts_search(b, iterations=100)
    
    if best_move == -1:  # If MCTS fails, fall back to N-Tuple approximator
        best_move = tdl.select_best_move(b).action()

    return best_move

def mcts_search(b, iterations=100, exploration_weight=1.0):
    """
    Monte Carlo Tree Search to find the best action, guided by the trained N-tuple network
    
    Args:
        b: Current board state
        iterations: Number of MCTS iterations
        exploration_weight: Controls exploration vs exploitation balance
        
    Returns:
        Best action (0=up, 1=down, 2=left, 3=right)
    """
    class Node:
        def __init__(self, board_state, parent=None, action=None, reward=0):
            self.board_state = board_state  # Board state at this node
            self.parent = parent  # Parent node
            self.action = action  # Action that led to this state
            self.reward = reward  # Immediate reward for the action
            self.children = {}  # Action -> Node mapping
            self.visits = 0  # Number of visits to this node
            self.value = 0  # Total value of this node
            self.untried_actions = [0, 1, 2, 3]  # Possible actions
            
        def ucb_score(self, exploration_weight):
            """Upper Confidence Bound score for node selection"""
            if self.visits == 0:
                return float('inf')
            exploitation = self.value / self.visits
            exploration = exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)
            return exploitation + exploration
            
        def select_child(self, exploration_weight):
            """Select child with highest UCB score"""
            return max(self.children.values(), key=lambda node: node.ucb_score(exploration_weight))
            
        def expand(self):
            """Expand by trying an untried action"""
            if not self.untried_actions:
                return None
                
            action = self.untried_actions.pop()
            new_board = board(self.board_state)
            reward = new_board.move(action)
            
            # If move is valid
            if reward != -1:
                child = Node(new_board, parent=self, action=action, reward=reward)
                self.children[action] = child
                return child
            return None
            
        def is_fully_expanded(self):
            """Check if all actions have been tried"""
            return len(self.untried_actions) == 0
            
        def is_terminal(self):
            """Check if this is a terminal state"""
            # Check if any move is possible
            for action in range(4):
                test_board = board(self.board_state)
                if test_board.move(action) != -1:
                    return False
            return True
    
    # Root node of the search tree
    root = Node(b)
    
    # MCTS iterations
    for _ in range(iterations):
        # Selection phase: select a node to expand
        node = root
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.select_child(exploration_weight)
        
        # Expansion phase: expand the selected node
        if not node.is_terminal():
            while not node.is_fully_expanded():
                child = node.expand()
                if child is not None:
                    node = child
                    break
        
        # Simulation phase: use N-tuple network for evaluation instead of random playout
        sim_board = board(node.board_state)
        
        # Use the trained N-tuple network to evaluate the state
        state_value = tdl.estimate(sim_board)
        
        # Add immediate reward to the state value
        total_value = node.reward + state_value
        
        # Backpropagation phase: update values up the tree
        while node is not None:
            node.visits += 1
            node.value += total_value
            node = node.parent
    
    # Choose the best action based on visit count
    if not root.children:
        return -1  # No valid moves
        
    return max(root.children.items(), key=lambda item: item[1].visits)[0]

