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

def get_action(state, score):
    # return random.choice([0, 1, 2, 3]) # Choose a random action

    b = state_to_board(state)
    
    # Use your N-Tuple approximator to select the best move
    best_move = tdl.select_best_move(b).action()

    action = best_move

    return action
    
    # You can submit this random agent to evaluate the performance of a purely random strategy.


