import sys
import math
import random
import struct
import typing
import abc
import time


def info(*argv):
    """
    default info output
    """
    print(*argv, file=sys.stdout)

def error(*argv):
    """
    default error output
    """
    print(*argv, file=sys.stderr)

def debug(*argv):
    """
    default debug output
    to enable debugging, uncomment the debug output lines below, i.e., debug(...)
    """
    print(*argv, file=sys.stderr)


class board:
    """
    64-bit bitboard implementation for 2048

    index:
     0  1  2  3
     4  5  6  7
     8  9 10 11
    12 13 14 15

    note that the 64-bit raw value is stored in little endian
    i.e., 0x4312752186532731 is displayed as
    +------------------------+
    |     2     8   128     4|
    |     8    32    64   256|
    |     2     4    32   128|
    |     4     2     8    16|
    +------------------------+
    """

    def __init__(self, raw = 0):
        self.raw = int(raw)

    def __int__(self):
        return self.raw

    def fetch(self, i):
        """
        get a 16-bit row
        """
        return (self.raw >> (i << 4)) & 0xffff

    def place(self, i, r):
        """
        set a 16-bit row
        """
        self.raw = (self.raw & ~(0xffff << (i << 4))) | ((r & 0xffff) << (i << 4))

    def at(self, i):
        """
        get a 4-bit tile
        """
        return (self.raw >> (i << 2)) & 0x0f

    def set(self, i, t):
        """
        set a 4-bit tile
        """
        self.raw = (self.raw & ~(0x0f << (i << 2))) | ((t & 0x0f) << (i << 2))

    def __getitem__(self, i):
        return self.at(i)

    def __setitem__(self, i, t):
        self.set(i, t)

    def __eq__(self, other):
        return isinstance(other, board) and self.raw == other.raw

    def __lt__(self, other):
        return isinstance(other, board) and self.raw < other.raw

    def __ne__(self, other):
        return not self == other

    def __gt__(self, other):
        return isinstance(other, board) and other < self

    def __le__(self, other):
        return isinstance(other, board) and not other < self

    def __ge__(self, other):
        return isinstance(other, board) and not self < other

    class lookup:
        """
        the lookup table for sliding board
        """

        find = [None] * 65536

        class entry:
            def __init__(self, row):
                V = [ (row >> 0) & 0x0f, (row >> 4) & 0x0f, (row >> 8) & 0x0f, (row >> 12) & 0x0f ]
                L, score = board.lookup.entry.mvleft(V)
                V.reverse() # mirror
                R, score = board.lookup.entry.mvleft(V)
                R.reverse()
                self.raw = row # base row (16-bit raw)
                self.left = (L[0] << 0) | (L[1] << 4) | (L[2] << 8) | (L[3] << 12) # left operation
                self.right = (R[0] << 0) | (R[1] << 4) | (R[2] << 8) | (R[3] << 12) # right operation
                self.score = score # merge reward

            def move_left(self, raw, sc, i):
                return raw | (self.left << (i << 4)), sc + self.score

            def move_right(self, raw, sc, i):
                return raw | (self.right << (i << 4)), sc + self.score

            @staticmethod
            def mvleft(row):
                buf = [t for t in row if t]
                res, score = [], 0
                while buf:
                    if len(buf) >= 2 and buf[0] is buf[1]:
                        buf = buf[1:]
                        buf[0] += 1
                        score += 1 << buf[0]
                    res += [buf[0]]
                    buf = buf[1:]
                return res + [0] * (4 - len(res)), score

        @classmethod
        def init(cls):
            cls.find = [cls.entry(row) for row in range(65536)]

    def init(self):
        """
        reset to initial state, i.e., witn only 2 random tiles on board
        """
        self.raw = 0
        self.popup()
        self.popup()

    def popup(self):
        """
        add a new random tile on board, or do nothing if the board is full
        2-tile: 90%
        4-tile: 10%
        """
        space = [i for i in range(16) if self.at(i) == 0]
        if space:
            self.set(random.choice(space), 1 if random.random() < 0.9 else 2)

    def move(self, opcode):
        """
        apply an action to the board
        return the reward of the action, or -1 if the action is illegal
        """
        if opcode == 0:
            return self.move_up()
        elif opcode == 1:
            return self.move_down()
        elif opcode == 2:
            return self.move_left()
        elif opcode == 3:
            return self.move_right()
        else:
            return -1

    def move_left(self):
        move = 0
        prev = self.raw
        score = 0
        for i in range(4):
            move, score = self.lookup.find[self.fetch(i)].move_left(move, score, i)
        self.raw = move
        return score if move != prev else -1

    def move_right(self):
        move = 0
        prev = self.raw
        score = 0
        for i in range(4):
            move, score = self.lookup.find[self.fetch(i)].move_right(move, score, i)
        self.raw = move
        return score if move != prev else -1

    def move_up(self):
        self.rotate_clockwise()
        score = self.move_right()
        self.rotate_counterclockwise()
        return score

    def move_down(self):
        self.rotate_clockwise()
        score = self.move_left()
        self.rotate_counterclockwise()
        return score

    def transpose(self):
        """
        swap rows and columns
        +------------------------+       +------------------------+
        |     2     8   128     4|       |     2     8     2     4|
        |     8    32    64   256|       |     8    32     4     2|
        |     2     4    32   128| ----> |   128    64    32     8|
        |     4     2     8    16|       |     4   256   128    16|
        +------------------------+       +------------------------+
        """
        self.raw = (self.raw & 0xf0f00f0ff0f00f0f) | ((self.raw & 0x0000f0f00000f0f0) << 12) | ((self.raw & 0x0f0f00000f0f0000) >> 12)
        self.raw = (self.raw & 0xff00ff0000ff00ff) | ((self.raw & 0x00000000ff00ff00) << 24) | ((self.raw & 0x00ff00ff00000000) >> 24)

    def mirror(self):
        """
        reflect the board horizontally, i.e., exchange columns
        +------------------------+       +------------------------+
        |     2     8   128     4|       |     4   128     8     2|
        |     8    32    64   256|       |   256    64    32     8|
        |     2     4    32   128| ----> |   128    32     4     2|
        |     4     2     8    16|       |    16     8     2     4|
        +------------------------+       +------------------------+
        """
        self.raw = ((self.raw & 0x000f000f000f000f) << 12) | ((self.raw & 0x00f000f000f000f0) << 4) \
                 | ((self.raw & 0x0f000f000f000f00) >> 4) | ((self.raw & 0xf000f000f000f000) >> 12)

    def flip(self):
        """
        reflect the board vertically, i.e., exchange rows
        +------------------------+       +------------------------+
        |     2     8   128     4|       |     4     2     8    16|
        |     8    32    64   256|       |     2     4    32   128|
        |     2     4    32   128| ----> |     8    32    64   256|
        |     4     2     8    16|       |     2     8   128     4|
        +------------------------+       +------------------------+
        """
        self.raw = ((self.raw & 0x000000000000ffff) << 48) | ((self.raw & 0x00000000ffff0000) << 16) \
                 | ((self.raw & 0x0000ffff00000000) >> 16) | ((self.raw & 0xffff000000000000) >> 48)

    def rotate(self, r = 1):
        """
        rotate the board clockwise by given times
        """
        r = ((r % 4) + 4) % 4
        if r == 0:
            pass
        elif r == 1:
            self.rotate_clockwise()
        elif r == 2:
            self.reverse()
        elif r == 3:
            self.rotate_counterclockwise()

    def rotate_clockwise(self):
        self.transpose()
        self.mirror()

    def rotate_counterclockwise(self):
        self.transpose()
        self.flip()

    def reverse(self):
        self.mirror()
        self.flip()

    def __str__(self):
        state = '+' + '-' * 24 + '+\n'
        for i in range(0, 16, 4):
            state += ('|' + ''.join('{0:6d}'.format((1 << self.at(j)) & -2) for j in range(i, i + 4)) + '|\n')
            # use -2 (0xff...fe) to remove the unnecessary 1 for (1 << 0)
        state += '+' + '-' * 24 + '+'
        return state


class feature(abc.ABC):
    """
    feature and weight table for n-tuple networks
    """

    def __init__(self, length):
        self.weight = feature.alloc(length)

    def __getitem__(self, i):
        return self.weight[i]

    def __setitem__(self, i, v):
        self.weight[i] = v

    def __len__(self):
        return len(self.weight)

    def size(self):
        return len(self.weight)

    @abc.abstractmethod
    def estimate(self, b):
        """
        estimate the value of a given board
        """
        pass

    @abc.abstractmethod
    def update(self, b, u):
        """
        update the value of a given board, and return its updated value
        """
        pass

    @abc.abstractmethod
    def name(self):
        """
        get the name of this feature
        """
        pass

    def dump(self, b, out = info):
        """
        dump the detail of weight table of a given board
        """
        out(f"{b}\nestimate = {self.estimate(b)}")

    def write(self, output):
        name = self.name().encode('utf-8')
        output.write(struct.pack('I', len(name)))
        output.write(name)
        size = len(self.weight)
        output.write(struct.pack('Q', size))
        output.write(struct.pack(f'{size}f', *self.weight))

    def read(self, input):
        size = struct.unpack('I', input.read(4))[0]
        name = input.read(size).decode('utf-8')
        if name != self.name():
            error(f'unexpected feature: {name} ({self.name()} is expected)')
            exit(1)
        size = struct.unpack('Q', input.read(8))[0]
        if size != len(self.weight):
            error(f'unexpected feature size {size} for {self.name()} ({self.size()} is expected)')
            exit(1)
        self.weight = list(struct.unpack(f'{size}f', input.read(size * 4)))
        if len(self.weight) != size:
            error('unexpected end of binary')
            exit(1)

    @staticmethod
    def alloc(num):
        if not hasattr(feature.alloc, "total"):
            feature.alloc.total = 0
            feature.alloc.limit = (1 << 30) // 4 # 1G memory
        try:
            feature.alloc.total += num
            if feature.alloc.total > feature.alloc.limit:
                raise MemoryError("memory limit exceeded")
            return [float(320000/num)] * num
        except MemoryError as e:
            error("memory limit exceeded")
            exit(-1)
        return None


class pattern(feature):
    """
    the pattern feature
    including isomorphic (rotate/mirror)

    index:
     0  1  2  3
     4  5  6  7
     8  9 10 11
    12 13 14 15

    isomorphic:
     1: no isomorphic
     4: enable rotation
     8: enable rotation and reflection (default)

    usage:
     pattern([ 0, 1, 2, 3 ])
     pattern([ 0, 1, 2, 3, 4, 5 ])
     pattern([ 0, 1, 2, 3, 4, 5 ], 4)
    """

    def __init__(self, patt, iso = 8):
        super().__init__(1 << (len(patt) * 4))
        if not patt:
            error("no pattern defined")
            exit(1)

        """
        isomorphic patterns can be calculated by board
        take isomorphic patterns [ 0, 1, 2, 3 ] and [ 12, 8, 4, 0 ] as example

        +------------------------+       +------------------------+
        |     2     8   128     4|       |     4     2     8     2|
        |     8    32    64   256|       |     2     4    32     8|
        |     2     4    32   128| ----> |     8    32    64   128|
        |     4     2     8    16|       |    16   128   256     4|
        +------------------------+       +------------------------+
        the left side is an original board and the right side is its clockwise rotation

        apply [ 0, 1, 2, 3 ] to the original board will extract 0x2731
        apply [ 0, 1, 2, 3 ] to the clockwise rotated board will extract 0x1312,
        which is the same as applying [ 12, 8, 4, 0 ] to the original board

        therefore the 8 isomorphic patterns can be calculated by
        using a board whose value is 0xfedcba9876543210 as follows
        """
        self.isom = [None] * iso
        for i in range(iso):
            idx = board(0xfedcba9876543210)
            if i >= 4:
                idx.mirror()
            idx.rotate(i)
            self.isom[i] = [idx.at(t) for t in patt]

    def estimate(self, b):
        """
        estimate the value of a given board
        """
        value = 0
        for iso in self.isom:
            index = self.indexof(iso, b)
            value += self.weight[index]
        return value

    def update(self, b, u):
        """
        update the value of a given board, and return its updated value
        """
        adjust = u / len(self.isom)
        value = 0
        for iso in self.isom:
            index = self.indexof(iso, b)
            self.weight[index] += adjust
            value += self.weight[index]
        return value

    def name(self):
        """
        get the name of this feature
        """
        return f"{len(self.isom[0])}-tuple pattern {self.nameof(self.isom[0])}"

    def dump(self, b, out = info):
        """
        display the weight information of a given board
        """
        for iso in self.isom:
            index = self.indexof(iso, b)
            tiles = [(index >> (4 * i)) & 0x0f for i in range(len(iso))]
            out(f"#{self.nameof(iso)}[{self.nameof(tiles)}] = {self[index]}")

    def indexof(self, patt, b):
        index = 0
        for i, pos in enumerate(patt):
            index |= b.at(pos) << (4 * i)
        return index

    def nameof(self, patt):
        return "".join([f"{p:x}" for p in patt])


class move:
    """
    the data structure for the move
    store state, action, reward, afterstate, and value
    """

    def __init__(self, board = None, opcode = -1):
        self.before = None
        self.after = None
        self.opcode = opcode
        self.score = -1
        self.esti = -float('inf')
        if board is not None:
            self.assign(board)

    def state(self):
        return self.before

    def afterstate(self):
        return self.after

    def value(self):
        return self.esti

    def reward(self):
        return self.score

    def action(self):
        return self.opcode

    def set_state(self, state):
        self.before = state

    def set_afterstate(self, state):
        self.after = state

    def set_value(self, value):
        self.esti = value

    def set_reward(self, reward):
        self.score = reward

    def set_action(self, action):
        self.opcode = action

    def __eq__(self, other):
        return isinstance(other, move) and self.opcode == other.opcode and \
            self.before == other.before and self.after == other.after and \
            self.esti == other.esti and self.score == other.score

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return isinstance(other, move) and self.before == other.before and self.esti < other.esti

    def __le__(self, other):
        return self.__lt__(other) or self.__eq__(other)

    def __gt__(self, other):
        return isinstance(other, move) and other.__lt__(self)

    def __ge__(self, other):
        return self.__gt__(other) or self.__eq__(other)

    def assign(self, b):
        """
        assign a state, then apply the action to generate its afterstate
        return True if the action is valid for the given state
        """
        # debug(f"assign {self.name()}\n{b}")
        self.after = board(b)
        self.before = board(b)
        self.score = self.after.move(self.opcode)
        self.esti = self.score if self.score != -1 else -float('inf')
        return self.score != -1

    def is_valid(self):
        """
        check the move is valid or not

        the move is considered invalid if
         estimated value becomes to NaN (wrong learning rate?)
         invalid action (cause after == before or score == -1)

        call this function after initialization (assign, set_value, etc)
        """
        if math.isnan(self.esti):
            error("numeric exception")
            exit(-1)
        return self.after != self.before and self.opcode != -1 and self.score != -1

    def name(self):
        opname = [ "up", "down", "left", "right" ]
        return opname[self.opcode] if self.opcode >= 0 and self.opcode < 4 else "none"

    def __str__(self):
        move_str = f"moving {self.name()}, reward = {self.score}"
        if self.is_valid():
            move_str += f", value = {self.esti}\n{self.after}"
        else:
            move_str += " (invalid)"
        return move_str
    

class TDLearning:
    def __init__(self):
        self.feats = []
        self.scores = []
        self.maxtile = []
        self.avg_scores = []
        self.stage = 1  # Track which stage we're training (1: up to 8192, 2: 8192 to 16384)
        self.stage_threshold = 13  # 2^13 = 8192

    def add_feature(self, feat):
        """
        add a feature into tuple networks
        """
        self.feats.append(feat)
        sign = f"{feat.name()}, size = {feat.size()}"
        usage = feat.size() * 4
        if usage >= (1 << 30):
            size = f"{(usage >> 30)}GB"
        elif usage >= (1 << 20):
            size = f"{(usage >> 20)}MB"
        elif usage >= (1 << 10):
            size = f"{(usage >> 10)}KB"
        info(f"{sign} ({size})")

    def estimate(self, b):
        """
        estimate the value of the given state
        by accumulating all corresponding feature weights
        """
        # debug(f"estimate {b}")
        return sum(feat.estimate(b) for feat in self.feats)

    def update(self, b, u):
        """
        update the value of the given state and return its new value
        """
        # debug(f"update ({u})\n{b}")
        adjust = u / len(self.feats)
        return sum(feat.update(b, adjust) for feat in self.feats)

    def select_best_move(self, b):
        """
        select the best move of a state b using tile-downgrading when applicable
        and using the appropriate stage model based on the largest tile
        """
        best = move(b)
        
        # Check if tile-downgrading should be applied
        should_downgrade = False
        largest_tile = 0
        tile_counts = [0] * 16  # Count of each tile type (up to 2^31)
        
        # Find the largest tile and count all tiles
        for i in range(16):
            tile_value = b.at(i)
            if tile_value > 0:
                tile_counts[tile_value] += 1
                largest_tile = max(largest_tile, tile_value)
        
        # Find the largest missing tile not exceeding the largest tile
        largest_missing_tile = 0
        for i in range(1, largest_tile):
            if tile_counts[i] == 0:
                largest_missing_tile = i
                break
                
        # Apply downgrading if there's a missing tile smaller than the largest tile
        if largest_missing_tile <= largest_tile and largest_missing_tile > 7:
            should_downgrade = True
            downgrade_threshold = largest_missing_tile
        
        moves = [move(b, opcode) for opcode in range(4)]
        for mv in moves:
            if mv.is_valid():
                afterstate = mv.afterstate()
                
                # Apply tile-downgrading if needed
                if should_downgrade:
                    # Create a downgraded copy of the afterstate
                    downgraded = board(afterstate)
                    
                    # Downgrade tiles (halve values of tiles larger than the threshold)
                    for i in range(16):
                        tile_value = downgraded.at(i)
                        if tile_value > downgrade_threshold:
                            downgraded.set(i, tile_value - 1)
                    
                    # Estimate value using the downgraded state
                    estimated_value = mv.reward() + self.estimate(downgraded)
                else:
                    # Normal estimation
                    estimated_value = mv.reward() + self.estimate(afterstate)
                    
                mv.set_value(estimated_value)
                if mv.value() > best.value():
                    best = mv
                
        return best

    def learn_from_episode(self, path, alpha=0.1):
        """
        learn from the records in an episode, using the appropriate stage model
        """
        target = 0
        path.pop()  # ignore the last record
        while path:
            mv = path.pop()
            error = target - self.estimate(mv.afterstate())
            target = mv.reward() + self.update(mv.afterstate(), alpha * error)

    def make_statistic(self, n, b, score, unit = 1000):
        """
        update the statistic, and show the statistic every 1000 episodes by default

        the statistic contains average, maximum scores, and tile distributions, e.g.,

        100000  avg = 68663.7   max = 177508
                256     100%    (0.2%)
                512     99.8%   (0.9%)
                1024    98.9%   (7.7%)
                2048    91.2%   (22.5%)
                4096    68.7%   (53.9%)
                8192    14.8%   (14.8%)

        is the statistic from the 99001st to the 100000th games (assuming unit = 1000), where
         '100000': current iteration, i.e., number of games trained
         'avg = 68663.7  max = 177508': the average score is 68663.7
                                        the maximum score is 177508
         '2048 91.2% (22.5%)': 91.2% of games reached 2048-tiles, i.e., win rate of 2048-tile
                               22.5% of games terminated with 2048-tiles (the largest tile)
        """
        self.scores.append(score)
        self.maxtile.append(max(b.at(i) for i in range(16)))

        if n % unit == 0: # show the training process
            if len(self.scores) != unit or len(self.maxtile) != unit:
                error("wrong statistic size for show statistics")
                exit(2)

            avg_score = sum(self.scores) / len(self.scores)
            max_score = max(self.scores)
            info(f"{n}\tavg = {avg_score}\tmax = {max_score}")

            stat = [ self.maxtile.count(i) for i in range(16) ]
            t, c, coef = 1, 0, 100 / unit
            while c < unit:
                if stat[t] != 0:
                    accu = sum(stat[t:])
                    tile = (1 << t) & -2
                    winrate = accu * coef
                    share = stat[t] * coef
                    info(f"\t{tile}\t{winrate:.1f}%\t({share:.1f}%)")
                c += stat[t]
                t += 1

            self.avg_scores.append(avg_score)
            self.scores.clear()
            self.maxtile.clear()

            self.save("2048.bin")

    def dump(self, b, out = info):
        """
        display the weight information of a given board
        """
        out(f"{b}\nestimate = {self.estimate(b)}")
        for feat in self.feats:
            feat.dump(b, out=out)

    def load(self, path, stage=1):
        """
        load the weight table from binary file for the specified stage
        """
        try:
            with open(path, 'rb') as input:
                size = struct.unpack('Q', input.read(8))[0]
                if size != len(self.feats):
                    error(f"unexpected feature count: {size} ({len(self.feats)} is expected)")
                for feat in self.feats:
                    feat.read(input)
                    info(f"{feat.name()} is loaded from {path}")
                self.stage = stage
        except FileNotFoundError:
            pass

    def save(self, path):
        """
        save the weight table to binary file with stage information
        """
        try:
            stage_path = f"2048_stage{self.stage}.bin"
            with open(stage_path, 'wb') as output:
                output.write(struct.pack('Q', len(self.feats)))
                for feat in self.feats:
                    feat.write(output)
                    info(f"{feat.name()} is saved to {stage_path}")
        except FileNotFoundError:
            pass

    def train_stage2(self, total=50000, alpha_start=0.1, seed=0):
        """
        Train the second stage model specifically for boards with 8192 tiles
        """
        info(f"Training Stage 2 (8192 to 16384)")
        info(f"total = {total}")
        info(f"alpha_start = {alpha_start}")
        info(f"seed = {seed}")
        random.seed(seed)
        
        self.stage = 2
        self.scores.clear()
        self.maxtile.clear()
        
        for n in range(1, total + 1):
            # Adjust learning rate based on progress
            percentage = n / total
            if percentage > 0.75:
                alpha = alpha_start * 0.1
            elif percentage > 0.5:
                alpha = alpha_start * 0.1
            else:
                alpha = alpha_start
                
            path = []
            state = self.generate_8192_board()  # Start with a board that already has 8192
            score = 0
            
            # Play an episode starting from a board with 8192
            while True:
                best = self.select_best_move(state)
                path.append(best)
                
                if best.is_valid():
                    score += best.reward()
                    state = board(best.afterstate())
                    state.popup()
                else:
                    break
                    
            self.learn_from_episode(path, alpha)
            self.make_statistic(n, state, score)
            
        self.save("2048.bin")
        
    def generate_8192_board(self):
        """
        Generate a board that already has an 8192 tile for stage 2 training
        """
        # Method 1: Start with a predefined board with 8192
        b = board()
        b.set(0, 13)  # Set position 0 to have 8192 (2^13)
        
        # Add some random smaller tiles
        for _ in range(2):
            space = [i for i in range(1, 16) if b.at(i) == 0]
            if space:
                pos = random.choice(space)
                val = random.randint(1, 8)  # Random tile between 2 and 256
                b.set(pos, val)
                
        return b
        
        # Alternative method: Play using stage 1 model until we get 8192
        # This would be more realistic but slower
        # state = board()
        # state.init()
        # while max(state.at(i) for i in range(16)) < 13:  # 13 is 8192
        #     best = self.select_best_move(state)
        #     if best.is_valid():
        #         state = board(best.afterstate())
        #         state.popup()
        #     else:
        #         state = board()
        #         state.init()
        # return state

if __name__ == "__main__":
    board.lookup.init()
    tdl = TDLearning()

    # set the learning parameters
    total = 100000
    alpha = 0.1
    seed = 0
    info(f"total = {total}")
    info(f"alpha = {alpha}")
    info(f"seed = {seed}")
    random.seed(seed)

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

    # Check if we want to train stage 1 or stage 2
    train_stage = 1
    if len(sys.argv) > 1 and sys.argv[1] == "2":
        train_stage = 2
        
    if train_stage == 1:
        # Train stage 1 (up to 8192)
        # train the model
        for n in range(1, total + 1):
            percentage = n / total
            if percentage > 0.75:
                alpha = 0.001
            elif percentage > 0.5:
                alpha = 0.01
            else:
                alpha = 0.1

            path = []
            state = board()
            score = 0

            # play an episode
            state.init()
            while True:
                best = tdl.select_best_move(state)
                path.append(best)

                if best.is_valid():
                    score += best.reward()
                    state = board(best.afterstate())
                    state.popup()
                else:
                    break

            tdl.learn_from_episode(path, alpha)
            tdl.make_statistic(n, state, score)
        
        tdl.save("2048.bin")
    
    elif train_stage == 2:
        # Train stage 2 (8192 to 16384)
        # First load the stage 1 weights as a starting point
        tdl.load("2048_stage1.bin", stage=1)
        # Then train stage 2
        tdl.train_stage2(total=100000, alpha_start=0.1)