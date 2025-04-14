import sys
import numpy as np
import random

class Connect6Game:
    def __init__(self, size=19):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)  # 0: Empty, 1: Black, 2: White
        self.turn = 1  # 1: Black, 2: White
        self.game_over = False

    def reset_board(self):
        """Clears the board and resets the game."""
        self.board.fill(0)
        self.turn = 1
        self.game_over = False
        print("= ", flush=True)

    def set_board_size(self, size):
        """Sets the board size and resets the game."""
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.turn = 1
        self.game_over = False
        print("= ", flush=True)

    def check_win(self):
        """Checks if a player has won.
        Returns:
        0 - No winner yet
        1 - Black wins
        2 - White wins
        """
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] != 0:
                    current_color = self.board[r, c]
                    for dr, dc in directions:
                        prev_r, prev_c = r - dr, c - dc
                        if 0 <= prev_r < self.size and 0 <= prev_c < self.size and self.board[prev_r, prev_c] == current_color:
                            continue
                        count = 0
                        rr, cc = r, c
                        while 0 <= rr < self.size and 0 <= cc < self.size and self.board[rr, cc] == current_color:
                            count += 1
                            rr += dr
                            cc += dc
                        if count >= 6:
                            return current_color
        return 0

    def index_to_label(self, col):
        """Converts column index to letter (skipping 'I')."""
        return chr(ord('A') + col + (1 if col >= 8 else 0))  # Skips 'I'

    def label_to_index(self, col_char):
        """Converts letter to column index (accounting for missing 'I')."""
        col_char = col_char.upper()
        if col_char >= 'J':  # 'I' is skipped
            return ord(col_char) - ord('A') - 1
        else:
            return ord(col_char) - ord('A')

    def play_move(self, color, move):
        """Places stones and checks the game status."""
        if self.game_over:
            print("? Game over")
            return

        stones = move.split(',')
        positions = []

        for stone in stones:
            stone = stone.strip()
            if len(stone) < 2:
                print("? Invalid format")
                return
            col_char = stone[0].upper()
            if not col_char.isalpha():
                print("? Invalid format")
                return
            col = self.label_to_index(col_char)
            try:
                row = int(stone[1:]) - 1
            except ValueError:
                print("? Invalid format")
                return
            if not (0 <= row < self.size and 0 <= col < self.size):
                print("? Move out of board range")
                return
            if self.board[row, col] != 0:
                print("? Position already occupied")
                return
            positions.append((row, col))

        for row, col in positions:
            self.board[row, col] = 1 if color.upper() == 'B' else 2

        self.turn = 3 - self.turn
        print('= ', end='', flush=True)

    def generate_move(self, color):
        """Generates a move using Monte Carlo Tree Search."""
        if self.game_over:
            print("? Game over", flush=True)
            return

        # Convert color to player number (1 for Black, 2 for White)
        player = 1 if color.upper() == 'B' else 2
        best_move = None
        
        # Special case: If playing as Black and this is the first move, play center
        if player == 1 and np.count_nonzero(self.board) == 0:
            center = self.size // 2
            best_move = (center, center)
        else:
            # First check for immediate winning move
            defensing_move = self.find_winning_move(self.board, 3 - player)

            if defensing_move:  
                best_move = defensing_move
            else:
                winning_move = self.find_winning_move(self.board, player)
                if winning_move:
                    best_move = winning_move
                else:
                    # Run MCTS to find the best move
                    best_move = self.mcts_search(player)
                
                # Safety check: If MCTS somehow failed to find a move, find any valid move
                if best_move is None or self.board[best_move[0], best_move[1]] != 0:
                    # Find any empty position
                    empty_positions = [(r, c) for r in range(self.size) 
                                      for c in range(self.size) if self.board[r, c] == 0]
                    if empty_positions:
                        best_move = empty_positions[0]
                    else:
                        print("? No valid moves available", flush=True)
                        return
        
        # Format the move
        move_str = f"{self.index_to_label(best_move[1])}{best_move[0]+1}"
        
        # Apply the move to the board
        self.board[best_move[0], best_move[1]] = player
        
        # Return the move with proper GTP formatting and ensure buffer is flushed
        print(f"= {move_str}", flush=True)
        
        # Debug output to stderr
        print(move_str, file=sys.stderr, flush=True)

    def analyze_board_patterns(self, state, player):
        """
        Analyzes the entire board to find pattern formations and returns a score.
        Includes detection for broken patterns (XX.X, XXX.X, etc.)
        """
        opponent = 3 - player
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        pattern_score = 0
        
        # Pattern counts
        patterns = {
            'open_2': 0,
            'semi_open_2': 0,
            'open_3': 0,
            'semi_open_3': 0,
            'broken_3': 0,  # NEW: Broken Three (XX.X or X.XX)
            'open_4': 0,
            'semi_open_4': 0,
            'broken_4': 0,  # NEW: Broken Four (XXX.X, XX.XX, X.XXX)
            'open_5': 0,
            'semi_open_5': 0,
            'double_3': 0,
            'double_4': 0,
            'broken_double': 0,  # NEW: Double Broken patterns
            'four_plus_gap': 0,
            'three_three': 0,
            'four_three_combo': 0,
            'two_two': 0,
            'two_three_combo': 0
        }
        
        # Calculate broken patterns
        patterns['broken_3'] = self.count_broken_threes(state, player)
        patterns['broken_4'] = self.count_broken_fours(state, player)
        
        # Track all patterns to detect combinations
        twos_positions = []
        threes_positions = []
        fours_positions = []
        fives_positions = []
        
        # Scan the entire board
        for r in range(state.shape[0]):
            for c in range(state.shape[1]):
                if state[r, c] == player:
                    # For each stone, check all directions
                    for dr, dc in directions:
                        # Skip if this is not the start of a pattern
                        prev_r, prev_c = r - dr, c - dc
                        if 0 <= prev_r < state.shape[0] and 0 <= prev_c < state.shape[1] and state[prev_r, prev_c] == player:
                            continue
                        
                        # Count consecutive stones and check open ends
                        count = 0
                        positions = []
                        gaps = []
                        rr, cc = r, c
                        
                        # Count forward with possible gaps
                        consecutive = 0
                        gap_count = 0
                        
                        for i in range(7):  # Check 7 positions to find patterns with gaps
                            if not (0 <= rr < state.shape[0] and 0 <= cc < state.shape[1]):
                                break
                            
                            if state[rr, cc] == player:
                                count += 1
                                consecutive += 1
                                positions.append((rr, cc))
                            elif state[rr, cc] == 0:
                                if gap_count < 2 and consecutive > 0:  # Allow up to 2 gaps
                                    gaps.append((rr, cc))
                                    gap_count += 1
                                else:
                                    break
                            else:
                                break  # Opponent stone
                            
                            rr += dr
                            cc += dc
                        
                        # Check if forward end is open
                        forward_open = (0 <= rr < state.shape[0] and 0 <= cc < state.shape[1] and state[rr, cc] == 0)
                        
                        # Check if backward end is open
                        back_r, back_c = r - dr, c - dc
                        backward_open = (0 <= back_r < state.shape[0] and 0 <= back_c < state.shape[1] and state[back_r, back_c] == 0)
                        
                        # Classify the pattern
                        if count == 2 and gap_count == 0:
                            if forward_open and backward_open:
                                patterns['open_2'] += 1
                                twos_positions.append(positions)
                            elif forward_open or backward_open:
                                patterns['semi_open_2'] += 1
                        elif count == 3 and gap_count == 0:
                            if forward_open and backward_open:
                                patterns['open_3'] += 1
                                threes_positions.append(positions)
                            elif forward_open or backward_open:
                                patterns['semi_open_3'] += 1
                        elif count == 4 and gap_count == 0:
                            if forward_open and backward_open:
                                patterns['open_4'] += 1
                                fours_positions.append(positions)
                            elif forward_open or backward_open:
                                patterns['semi_open_4'] += 1
                        elif count == 5 and gap_count == 0:
                            if forward_open or backward_open:
                                patterns['open_5'] += 1
                                fives_positions.append(positions)
                            else:
                                patterns['semi_open_5'] += 1
                        
                        # Detect "Four Plus Gap" pattern (4 stones with a single gap)
                        if count == 4 and gap_count == 1 and len(gaps) == 1:
                            if forward_open or backward_open:
                                patterns['four_plus_gap'] += 1
    
        # Detect double threats and combinations (non-overlapping patterns)
        # Check for double-2 (two open twos)
        for i in range(len(twos_positions)):
            for j in range(i+1, len(twos_positions)):
                if not any(pos in twos_positions[j] for pos in twos_positions[i]):
                    patterns['two_two'] += 1
        
        # Check for double-3
        for i in range(len(threes_positions)):
            for j in range(i+1, len(threes_positions)):
                if not any(pos in threes_positions[j] for pos in threes_positions[i]):
                    patterns['double_3'] += 1
        
        # Check for double-4
        for i in range(len(fours_positions)):
            for j in range(i+1, len(fours_positions)):
                if not any(pos in fours_positions[j] for pos in fours_positions[i]):
                    patterns['double_4'] += 1
        
        # Check for two-three combination
        for two_pos in twos_positions:
            for three_pos in threes_positions:
                if not any(pos in three_pos for pos in two_pos):
                    patterns['two_three_combo'] += 1
        
        # Check for four-three combination
        for four_pos in fours_positions:
            for three_pos in threes_positions:
                if not any(pos in three_pos for pos in four_pos):
                    patterns['four_three_combo'] += 1
        
        # Calculate score based on patterns
        pattern_score += patterns['open_2'] * 15        # New: Open Two
        pattern_score += patterns['semi_open_2'] * 5     # New: Semi-Open Two
        pattern_score += patterns['open_3'] * 50
        pattern_score += patterns['semi_open_3'] * 10
        pattern_score += patterns['open_4'] * 5000
        pattern_score += patterns['semi_open_4'] * 1000
        pattern_score += patterns['open_5'] * 12000
        pattern_score += patterns['semi_open_5'] * 6000
        pattern_score += patterns['double_3'] * 3000
        pattern_score += patterns['double_4'] * 20000
        pattern_score += patterns['four_plus_gap'] * 7000
        pattern_score += patterns['three_three'] * 400
        pattern_score += patterns['four_three_combo'] * 10000
        pattern_score += patterns['two_two'] * 100       # New: Double Open Two
        pattern_score += patterns['two_three_combo'] * 300 # New: Two-Three combination
        pattern_score += patterns['broken_3'] * 75   # Value between Open Two and Open Three
        pattern_score += patterns['broken_4'] * 800  # Value between Open Three and Open Four
        
        # Now do the same for opponent to calculate defensive score
        opponent_patterns = {
            'open_2': 0,
            'semi_open_2': 0,
            'open_3': 0,
            'semi_open_3': 0,
            'open_4': 0,
            'semi_open_4': 0,
            'open_5': 0,
            'semi_open_5': 0,
            'double_3': 0,
            'double_4': 0,
            'four_plus_gap': 0,
            'three_three': 0,
            'four_three_combo': 0,
            'two_two': 0,
            'two_three_combo': 0,
            'broken_double': 0,
            'broken_3': 0,
            'broken_4': 0
        }
        
        opponent_twos = []
        opponent_threes = []
        opponent_fours = []
        opponent_fives = []

        opponent_patterns['broken_3'] = self.count_broken_threes(state, opponent)
        opponent_patterns['broken_4'] = self.count_broken_fours(state, opponent)
        
        # Scan for opponent patterns (similar logic as above)
        for r in range(state.shape[0]):
            for c in range(state.shape[1]):
                if state[r, c] == opponent:
                    for dr, dc in directions:
                        prev_r, prev_c = r - dr, c - dc
                        if 0 <= prev_r < state.shape[0] and 0 <= prev_c < state.shape[1] and state[prev_r, prev_c] == opponent:
                            continue
                        
                        count = 0
                        positions = []
                        gaps = []
                        rr, cc = r, c
                        
                        # Count forward with possible gaps
                        consecutive = 0
                        gap_count = 0
                        
                        for i in range(7):
                            if not (0 <= rr < state.shape[0] and 0 <= cc < state.shape[1]):
                                break
                            
                            if state[rr, cc] == opponent:
                                count += 1
                                consecutive += 1
                                positions.append((rr, cc))
                            elif state[rr, cc] == 0:
                                if gap_count < 2 and consecutive > 0:
                                    gaps.append((rr, cc))
                                    gap_count += 1
                                else:
                                    break
                            else:
                                break
                            
                            rr += dr
                            cc += dc
                        
                        forward_open = (0 <= rr < state.shape[0] and 0 <= cc < state.shape[1] and state[rr, cc] == 0)
                        
                        back_r, back_c = r - dr, c - dc
                        backward_open = (0 <= back_r < state.shape[0] and 0 <= back_c < state.shape[1] and state[back_r, back_c] == 0)
                        
                        # Classify opponent patterns including Open Two and Semi-Open Two
                        if count == 2 and gap_count == 0:
                            if forward_open and backward_open:
                                opponent_patterns['open_2'] += 1
                                opponent_twos.append(positions)
                            elif forward_open or backward_open:
                                opponent_patterns['semi_open_2'] += 1
                        elif count == 3 and gap_count == 0:
                            if forward_open and backward_open:
                                opponent_patterns['open_3'] += 1
                                opponent_threes.append(positions)
                            elif forward_open or backward_open:
                                opponent_patterns['semi_open_3'] += 1
                        elif count == 4 and gap_count == 0:
                            if forward_open and backward_open:
                                opponent_patterns['open_4'] += 1
                                opponent_fours.append(positions)
                            elif forward_open or backward_open:
                                opponent_patterns['semi_open_4'] += 1
                        elif count == 5 and gap_count == 0:
                            if forward_open or backward_open:
                                opponent_patterns['open_5'] += 1
                                opponent_fives.append(positions)
                            else:
                                opponent_patterns['semi_open_5'] += 1
                        
                        # Detect opponent's "Four Plus Gap" pattern
                        if count == 4 and gap_count == 1 and len(gaps) == 1:
                            if forward_open or backward_open:
                                opponent_patterns['four_plus_gap'] += 1
    
        # Detect opponent's double threats and combinations
        # Check for opponent's double-2
        for i in range(len(opponent_twos)):
            for j in range(i+1, len(opponent_twos)):
                if not any(pos in opponent_twos[j] for pos in opponent_twos[i]):
                    opponent_patterns['two_two'] += 1
        
        # Check for opponent's double-3
        for i in range(len(opponent_threes)):
            for j in range(i+1, len(opponent_threes)):
                if not any(pos in opponent_threes[j] for pos in opponent_threes[i]):
                    opponent_patterns['double_3'] += 1
        
        # Check for opponent's double-4
        for i in range(len(opponent_fours)):
            for j in range(i+1, len(opponent_fours)):
                if not any(pos in opponent_fours[j] for pos in opponent_fours[i]):
                    opponent_patterns['double_4'] += 1
        
        # Check for opponent's two-three combination
        for two_pos in opponent_twos:
            for three_pos in opponent_threes:
                if not any(pos in three_pos for pos in two_pos):
                    opponent_patterns['two_three_combo'] += 1
        
        # Check for opponent's four-three combination
        for four_pos in opponent_fours:
            for three_pos in opponent_threes:
                if not any(pos in three_pos for pos in four_pos):
                    opponent_patterns['four_three_combo'] += 1
        
        # Calculate defensive score (higher weights to prioritize defense)
        defensive_score = 0
        defensive_score += opponent_patterns['open_2'] * 20       # New: Block Open Two
        defensive_score += opponent_patterns['semi_open_2'] * 7    # New: Block Semi-Open Two
        defensive_score += opponent_patterns['open_3'] * 60
        defensive_score += opponent_patterns['semi_open_3'] * 15
        defensive_score += opponent_patterns['open_4'] * 4000
        defensive_score += opponent_patterns['semi_open_4'] * 1200
        defensive_score += opponent_patterns['open_5'] * 12000
        defensive_score += opponent_patterns['semi_open_5'] * 6000
        defensive_score += opponent_patterns['double_3'] * 3500
        defensive_score += opponent_patterns['double_4'] * 18000
        defensive_score += opponent_patterns['four_plus_gap'] * 7000
        defensive_score += opponent_patterns['three_three'] * 450
        defensive_score += opponent_patterns['four_three_combo'] * 9000
        defensive_score += opponent_patterns['two_two'] * 120      # New: Block Double Open Two
        defensive_score += opponent_patterns['two_three_combo'] * 350 # New: Block Two-Three combination
        defensive_score += opponent_patterns['broken_3'] * 85    # Slightly higher value to prioritize defense
        defensive_score += opponent_patterns['broken_4'] * 850   # Slightly higher value to prioritize defense
        
        # Return the combined score
        return pattern_score - defensive_score

    def mcts_search(self, player, iterations=80, exploration_weight=1.41):
        """Monte Carlo Tree Search implementation for Connect6 with pattern recognition."""
        import math
        from copy import deepcopy
        
        # First check for immediate critical moves before starting MCTS
        opponent = 3 - player
        
        # Check for immediate winning move for us
        winning_move = self.find_winning_move(self.board, player)
        if winning_move:
            return winning_move
            
        # Check for immediate blocking move (opponent's winning move)
        blocking_move = self.find_winning_move(self.board, opponent)
        if blocking_move:
            return blocking_move
            
        # Check for opponent's critical threat (4+ in a row)
        critical_block = self.find_critical_threat(self.board, opponent)
        if critical_block:
            return critical_block
        
        # Now proceed with MCTS with pattern recognition
        class Node:
            def __init__(self, state, parent=None, move=None, player=None, game=None):
                self.state = state  # Board state
                self.parent = parent  # Parent node
                self.move = move  # Move that led to this state
                self.children = []  # Child nodes
                self.wins = 0  # Number of wins from this node
                self.visits = 0  # Number of visits to this node
                self.player = player  # Player who made the move to reach this state
                self.game = game  # Reference to the game object for pattern analysis
                self.untried_moves = self.get_restricted_moves(state, player)  # Restricted set of moves
                
            def get_restricted_moves(self, state, player):
                """Get a restricted set of valid moves with opening book pattern bias."""
                # Get all empty positions
                empty_positions = [(r, c) for r in range(state.shape[0]) 
                                  for c in range(state.shape[1]) if state[r, c] == 0]
                
                # If few moves remain, consider all
                if len(empty_positions) <= 10:
                    return empty_positions
                
                # Check if we're in early game (opening book applies)
                stone_count = np.count_nonzero(state)
                is_early_game = stone_count < 12
                
                strategic_moves = []
                center = state.shape[0] // 2
                opponent = 3 - player
                
                # Apply opening book patterns if in early game
                if is_early_game:
                    # Black's opening strategies
                    if player == 1:
                        # First move - play center
                        if stone_count == 0:
                            return [(center, center)]
                        
                        # Second move - create star pattern
                        elif stone_count == 1:
                            star_points = [
                                (center-3, center), (center+3, center),  # Horizontal points
                                (center, center-3), (center, center+3),  # Vertical points
                                (center-2, center-2), (center+2, center+2),  # Diagonal points
                                (center-2, center+2), (center+2, center-2)   # Diagonal points
                            ]
                            valid_points = [(r, c) for r, c in star_points 
                                           if 0 <= r < state.shape[0] and 0 <= c < state.shape[1] and state[r, c] == 0]
                            if valid_points:
                                # Assign high scores to star pattern moves
                                for r, c in valid_points:
                                    strategic_moves.append((r, c, 1000))  # High score to prioritize these moves
                        
                        # Third move - complete the star pattern
                        elif stone_count == 3:
                            # Look for existing star points and try to place opposite to them
                            for i in range(1, 4):
                                # Check horizontal
                                if state[center-i, center] == player and state[center+i, center] == 0:
                                    strategic_moves.append((center+i, center, 1000))
                                if state[center+i, center] == player and state[center-i, center] == 0:
                                    strategic_moves.append((center-i, center, 1000))
                                
                                # Check vertical
                                if state[center, center-i] == player and state[center, center+i] == 0:
                                    strategic_moves.append((center, center+i, 1000))
                                if state[center, center+i] == player and state[center, center-i] == 0:
                                    strategic_moves.append((center, center-i, 1000))
                                
                                # Check diagonals
                                if state[center-i, center-i] == player and state[center+i, center+i] == 0:
                                    strategic_moves.append((center+i, center+i, 1000))
                                if state[center+i, center+i] == player and state[center-i, center-i] == 0:
                                    strategic_moves.append((center-i, center-i, 1000))
                                
                                if state[center-i, center+i] == player and state[center+i, center-i] == 0:
                                    strategic_moves.append((center+i, center-i, 1000))
                                if state[center+i, center-i] == player and state[center-i, center+i] == 0:
                                    strategic_moves.append((center-i, center+i, 1000))
                    
                    # White's opening strategies
                    elif player == 2:
                        # First move as white - respond to center play with "net" strategy
                        if stone_count == 2:
                            net_points = [
                                (center-2, center-2), (center-2, center+2),
                                (center+2, center-2), (center+2, center+2)
                            ]
                            valid_points = [(r, c) for r, c in net_points 
                                           if 0 <= r < state.shape[0] and 0 <= c < state.shape[1] and state[r, c] == 0]
                            for r, c in valid_points:
                                strategic_moves.append((r, c, 1000))
                        
                        # Second move as white - extend the net pattern
                        elif stone_count == 4:
                            # Find clusters to connect
                            white_stones = [(r, c) for r in range(state.shape[0]) 
                                           for c in range(state.shape[1]) if state[r, c] == player]
                            
                            # If we have a stone at any corner of the center, try to connect to another corner
                            for r1, c1 in white_stones:
                                for r2, c2 in white_stones:
                                    if (r1, c1) != (r2, c2):
                                        # Try to find a connecting point between these stones
                                        mid_r, mid_c = (r1 + r2) // 2, (c1 + c2) // 2
                                        if 0 <= mid_r < state.shape[0] and 0 <= mid_c < state.shape[1] and state[mid_r, mid_c] == 0:
                                            strategic_moves.append((mid_r, mid_c, 900))
                    
                    # General early game strategies for both players
                    
                    # Block opponent's developing patterns
                    opponent_stones = [(r, c) for r in range(state.shape[0]) 
                                      for c in range(state.shape[1]) if state[r, c] == opponent]
                    
                    for r1, c1 in opponent_stones:
                        for r2, c2 in opponent_stones:
                            if (r1, c1) != (r2, c2):
                                # If these stones form a line with 1 gap
                                dr, dc = r2 - r1, c2 - c1
                                if abs(dr) == abs(dc) or dr == 0 or dc == 0:  # Diagonal or straight line
                                    # Check if the position to extend this line is empty
                                    extended_r, extended_c = r2 + dr, c2 + dc
                                    if 0 <= extended_r < state.shape[0] and 0 <= extended_c < state.shape[1] and state[extended_r, extended_c] == 0:
                                        # Block this extension point
                                        strategic_moves.append((extended_r, extended_c, 800))
                
                # If we have strategic moves from opening book, prioritize them
                if strategic_moves:
                    # Sort by score (highest first)
                    strategic_moves.sort(key=lambda x: x[2], reverse=True)
                    # Return top moves (up to 5) plus some other empty positions for exploration
                    top_strategic = [(r, c) for r, c, _ in strategic_moves[:5]]
                    return top_strategic + self.get_other_promising_moves(state, player, top_strategic)[:10]
                    
                # Otherwise, fall back to the original logic for mid/late game
                # Find positions with existing stones nearby (within 2 spaces)
                stones = [(r, c) for r in range(state.shape[0]) 
                         for c in range(state.shape[1]) if state[r, c] > 0]
                
                nearby_moves = []
                for r, c in empty_positions:
                    # Check if this position is near any existing stone
                    min_distance = float('inf')
                    for sr, sc in stones:
                        distance = max(abs(r - sr), abs(c - sc))  # Chebyshev distance
                        min_distance = min(min_distance, distance)
                        
                        if distance <= 2:  # Only consider positions very close to existing stones
                            # Score this position
                            score = self.evaluate_position(state, r, c, player)
                            # Add proximity bonus - closer stones get higher scores
                            proximity_bonus = (3 - distance) * 50  # 100 for distance 1, 50 for distance 2
                            score += proximity_bonus
                            nearby_moves.append((r, c, score))
                            break
                
                # If we found strategic moves, return the top moves by score
                if nearby_moves:
                    nearby_moves.sort(key=lambda x: x[2], reverse=True)
                    return [(r, c) for r, c, _ in nearby_moves[:min(15, len(nearby_moves))]]
                
                # Fallback: return center and positions near center
                center = state.shape[0] // 2
                return [(r, c) for r, c in empty_positions 
                       if abs(r - center) + abs(c - center) <= 8][:15]  # Manhattan distance to center
            
            def get_other_promising_moves(self, state, player, already_selected):
                """Get other promising moves besides those already selected from opening book."""
                empty_positions = [(r, c) for r in range(state.shape[0]) 
                                  for c in range(state.shape[1]) if state[r, c] == 0 and (r, c) not in already_selected]
                
                scored_moves = []
                for r, c in empty_positions:
                    score = self.evaluate_position(state, r, c, player)
                    # Add distance from center as a small factor
                    center = state.shape[0] // 2
                    center_dist = max(abs(r - center), abs(c - center))
                    score += (10 - center_dist) * 5  # Higher score for moves closer to center
                    scored_moves.append((r, c, score))
                
                scored_moves.sort(key=lambda x: x[2], reverse=True)
                return [(r, c) for r, c, _ in scored_moves]
            
            def evaluate_position(self, state, r, c, color):
                """Evaluate the strategic value of a position."""
                directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
                score = 0
                opponent = 3 - color
                
                # Temporarily place a stone to evaluate
                state[r, c] = color
                
                # Use the new pattern analysis function if game reference is available
                if self.game:
                    pattern_score = self.game.analyze_board_patterns(state, color)
                    score += pattern_score
                
                # Check for connectivity with existing stones of same color
                connectivity_score = 0
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        if dr == 0 and dc == 0:
                            continue  # Skip the position itself
                        
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < state.shape[0] and 0 <= nc < state.shape[1]:
                            if state[nr, nc] == color:
                                # Closer stones provide higher connectivity
                                distance = max(abs(dr), abs(dc))
                                connectivity_score += (6 - distance) * 50
                
                score += connectivity_score
                
                for dr, dc in directions:
                    # Count consecutive stones in both directions
                    count = 1  # Start with 1 for the current position
                    open_ends = 0
                    
                    # Check forward direction
                    rr, cc = r + dr, c + dc
                    while 0 <= rr < state.shape[0] and 0 <= cc < state.shape[1]:
                        if state[rr, cc] == color:
                            count += 1
                        elif state[rr, cc] == 0:
                            open_ends += 1
                            break
                        else:
                            break  # Blocked by opponent
                        rr += dr
                        cc += dc
                    
                    # Check backward direction
                    rr, cc = r - dr, c - dc
                    while 0 <= rr < state.shape[0] and 0 <= cc < state.shape[1]:
                        if state[rr, cc] == color:
                            count += 1
                        elif state[rr, cc] == 0:
                            open_ends += 1
                            break
                        else:
                            break  # Blocked by opponent
                        rr -= dr
                        cc -= dc
                    
                    # Score based on pattern
                    if count >= 5:
                        score += 10000  # Almost winning
                    elif count == 4 and open_ends == 2:
                        score += 5000   # Strong threat
                    elif count == 4 and open_ends == 1:
                        score += 1000   # Potential threat
                    elif count == 3 and open_ends == 2:
                        score += 500    # Developing threat
                    elif count == 3 and open_ends == 1:
                        score += 100    # Potential development
                    elif count == 2 and open_ends == 2:
                        score += 50     # Early development
                
                # Also check defensive value (blocking opponent)
                state[r, c] = opponent
                for dr, dc in directions:
                    count = 1
                    open_ends = 0
                    
                    # Forward direction
                    rr, cc = r + dr, c + dc
                    while 0 <= rr < state.shape[0] and 0 <= cc < state.shape[1]:
                        if state[rr, cc] == opponent:
                            count += 1
                        elif state[rr, cc] == 0:
                            open_ends += 1
                            break
                        else:
                            break
                        rr += dr
                        cc += dc
                    
                    # Backward direction
                    rr, cc = r - dr, c - dc
                    while 0 <= rr < state.shape[0] and 0 <= cc < state.shape[1]:
                        if state[rr, cc] == opponent:
                            count += 1
                        elif state[rr, cc] == 0:
                            open_ends += 1
                            break
                        else:
                            break
                        rr -= dr
                        cc -= dc
                    
                    # Score defensive value
                    if count >= 5:
                        score -= 8000  # Block opponent's potential win
                    elif count == 4 and open_ends == 2:
                        score -= 4000  # Block strong threat
                    elif count == 4 and open_ends == 1:
                        score -= 800   # Block potential threat
                    elif count == 3 and open_ends == 1:
                        score -= 400   # Block developing threat
                
                # Reset the position
                state[r, c] = 0
                
                return score
                
            def select_child(self, exploration_weight):
                """Select a child node using UCB1 formula with opening book bias."""
                log_parent_visits = math.log(self.visits)
                stone_count = np.count_nonzero(self.state)
                is_early_game = stone_count < 12
                
                def ucb(child):
                    exploitation = child.wins / child.visits if child.visits > 0 else 0
                    exploration = exploration_weight * math.sqrt(log_parent_visits / child.visits) if child.visits > 0 else float('inf')
                    
                    # Add opening book bias in early game
                    opening_bias = 0
                    if is_early_game and child.move:
                        r, c = child.move
                        center = self.state.shape[0] // 2
                        
                        # Bias toward center and star pattern for Black
                        if self.player == 1:
                            # Center bias
                            if r == center and c == center:
                                opening_bias += 0.5
                            
                            # Star pattern bias
                            if stone_count <= 3:
                                # Points at distance 2-3 from center
                                dist = max(abs(r - center), abs(c - center))
                                if dist in [2, 3]:
                                    opening_bias += 0.3
                        
                        # Net pattern bias for White
                        elif self.player == 2 and stone_count <= 4:
                            # Corner points around center
                            if abs(r - center) == 2 and abs(c - center) == 2:
                                opening_bias += 0.4
                    
                    return exploitation + exploration + opening_bias
                
                return max(self.children, key=ucb)
                
            def expand(self, state, move, next_player):
                """Expand the tree by adding a new child node."""
                child_state = deepcopy(state)
                child_state[move[0], move[1]] = next_player
                
                child = Node(child_state, parent=self, move=move, player=next_player, game=self.game)
                self.children.append(child)
                
                if move in self.untried_moves:
                    self.untried_moves.remove(move)
                
                return child
                
            def is_fully_expanded(self):
                """Check if all possible moves from this node have been tried."""
                return len(self.untried_moves) == 0
                
            def is_terminal(self):
                """Check if this node represents a terminal state (game over)."""
                # Check for a winner
                winner = self.check_win_state(self.state)
                if winner > 0:
                    return True
                
                # Check if the board is full
                if np.count_nonzero(self.state) == self.state.shape[0] * self.state.shape[1]:
                    return True
                
                return False
                
            def check_win_state(self, board):
                """Check if a player has won on the given board state."""
                directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
                for r in range(board.shape[0]):
                    for c in range(board.shape[1]):
                        if board[r, c] != 0:
                            current_color = board[r, c]
                            for dr, dc in directions:
                                prev_r, prev_c = r - dr, c - dc
                                if 0 <= prev_r < board.shape[0] and 0 <= prev_c < board.shape[1] and board[prev_r, prev_c] == current_color:
                                    continue
                                count = 0
                                rr, cc = r, c
                                while 0 <= rr < board.shape[0] and 0 <= cc < board.shape[1] and board[rr, cc] == current_color:
                                    count += 1
                                    rr += dr
                                    cc += dc
                                if count >= 6:
                                    return current_color
                return 0
                
            def rollout(self, current_player):
                """Simulate a game from this node using a semi-random policy with opening book patterns."""
                rollout_state = deepcopy(self.state)
                rollout_player = current_player
                
                # Use a limited depth for rollout (faster simulation)
                max_depth = 5  # Reduced for efficiency
                depth = 0
                stone_count = np.count_nonzero(rollout_state)
                is_early_game = stone_count < 12
                
                while depth < max_depth:
                    # Check for a winner
                    winner = self.check_win_state(rollout_state)
                    if winner > 0:
                        return winner
                    
                    # Check if the board is full
                    if np.count_nonzero(rollout_state) == rollout_state.shape[0] * rollout_state.shape[1]:
                        return 0  # Draw
                    
                    # Get valid moves
                    empty_positions = [(r, c) for r in range(rollout_state.shape[0]) 
                                      for c in range(rollout_state.shape[1]) if rollout_state[r, c] == 0]
                    
                    if not empty_positions:
                        return 0  # Draw (shouldn't happen)
                    
                    # In early game, apply opening book patterns with higher probability
                    if is_early_game and random.random() < 0.9:
                        center = rollout_state.shape[0] // 2
                        made_move = False
                        
                        # Apply opening patterns based on player and stone count
                        if rollout_player == 1:  # Black
                            if stone_count == 0:
                                # First move is center
                                rollout_state[center, center] = rollout_player
                                made_move = True
                            elif stone_count <= 3:
                                # Create star pattern
                                star_points = [
                                    (center-3, center), (center+3, center),
                                    (center, center-3), (center, center+3),
                                    (center-2, center-2), (center+2, center+2),
                                    (center-2, center+2), (center+2, center-2)
                                ]
                                valid_points = [(r, c) for r, c in star_points 
                                               if 0 <= r < rollout_state.shape[0] and 0 <= c < rollout_state.shape[1] 
                                               and rollout_state[r, c] == 0]
                                if valid_points:
                                    r, c = random.choice(valid_points)
                                    rollout_state[r, c] = rollout_player
                                    made_move = True
                        
                        elif rollout_player == 2:  # White
                            if stone_count <= 4:
                                # Create net pattern
                                net_points = [
                                    (center-2, center-2), (center-2, center+2),
                                    (center+2, center-2), (center+2, center+2)
                                ]
                                valid_points = [(r, c) for r, c in net_points 
                                               if 0 <= r < rollout_state.shape[0] and 0 <= c < rollout_state.shape[1] 
                                               and rollout_state[r, c] == 0]
                                if valid_points:
                                    r, c = random.choice(valid_points)
                                    rollout_state[r, c] = rollout_player
                                    made_move = True
                        
                        # If we already made a move based on opening patterns, continue
                        if made_move:
                            rollout_player = 3 - rollout_player
                            depth += 1
                            stone_count += 1
                            continue
                    
                    # If we reach here, use the existing simulation logic
                    if random.random() < 0.8:  # 80% chance to use heuristic
                        # Find winning move
                        for r, c in random.sample(empty_positions, min(10, len(empty_positions))):
                            rollout_state[r, c] = rollout_player
                            if self.check_win_state(rollout_state) == rollout_player:
                                return rollout_player  # Found winning move
                            rollout_state[r, c] = 0
                        
                        # Find blocking move
                        opponent = 3 - rollout_player
                        for r, c in random.sample(empty_positions, min(10, len(empty_positions))):
                            rollout_state[r, c] = opponent
                            if self.check_win_state(rollout_state) == opponent:
                                rollout_state[r, c] = rollout_player  # Block opponent
                                break
                            rollout_state[r, c] = 0
                        else:
                            # Find stones of current player
                            player_stones = [(r, c) for r in range(rollout_state.shape[0]) 
                                            for c in range(rollout_state.shape[1]) if rollout_state[r, c] == rollout_player]
                            
                            if player_stones:
                                # Prioritize moves near existing stones for better connectivity
                                move_scores = []
                                # Sample a subset of moves to evaluate (for speed)
                                sample_size = min(12, len(empty_positions))
                                sample_moves = random.sample(empty_positions, sample_size)
                                
                                for r, c in sample_moves:
                                    # Calculate connectivity score
                                    connectivity = 0
                                    for pr, pc in player_stones:
                                        dist = max(abs(r - pr), abs(c - pc))
                                        if dist <= 2:  # Only consider nearby stones
                                            connectivity += (3 - dist) * 10
                                    
                                    # Simple pattern evaluation
                                    pattern_score = 0
                                    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
                                    
                                    rollout_state[r, c] = rollout_player  # Temporarily place stone
                                    
                                    for dr, dc in directions:
                                        count = 1
                                        # Forward
                                        rr, cc = r + dr, c + dc
                                        while 0 <= rr < rollout_state.shape[0] and 0 <= cc < rollout_state.shape[1] and rollout_state[rr, cc] == rollout_player:
                                            count += 1
                                            rr += dr
                                            cc += dc
                                        # Backward
                                        rr, cc = r - dr, c - dc
                                        while 0 <= rr < rollout_state.shape[0] and 0 <= cc < rollout_state.shape[1] and rollout_state[rr, cc] == rollout_player:
                                            count += 1
                                            rr -= dr
                                            cc -= dc
                                        
                                        pattern_score += count * count  # Square for emphasis
                                    
                                    rollout_state[r, c] = 0  # Reset
                                    
                                    total_score = connectivity + pattern_score
                                    move_scores.append((r, c, total_score))
                                
                                # Choose one of the top moves
                                if move_scores:
                                    move_scores.sort(key=lambda x: x[2], reverse=True)
                                    top_moves = move_scores[:min(3, len(move_scores))]
                                    r, c, _ = random.choice(top_moves)
                                    rollout_state[r, c] = rollout_player
                                else:
                                    # Fallback to random
                                    r, c = random.choice(empty_positions)
                                    rollout_state[r, c] = rollout_player
                            else:
                                # No player stones yet, play near center
                                center = rollout_state.shape[0] // 2
                                center_moves = [(r, c) for r, c in empty_positions 
                                               if abs(r - center) + abs(c - center) < 5]
                                if center_moves:
                                    r, c = random.choice(center_moves)
                                else:
                                    r, c = random.choice(empty_positions)
                                rollout_state[r, c] = rollout_player
                    else:
                        # Choose a random move
                        r, c = random.choice(empty_positions)
                        rollout_state[r, c] = rollout_player
                    
                    # Switch player
                    rollout_player = 3 - rollout_player
                    depth += 1
                    stone_count += 1
                
                # If we reach max depth without conclusion, evaluate the position
                return self.evaluate_rollout_result(rollout_state, player)
            
            def evaluate_rollout_result(self, state, original_player):
                """Evaluate the final state if max depth is reached."""
                # Count stones and patterns for each player
                player_score = 0
                opponent_score = 0
                opponent = 3 - original_player
                
                # Check for patterns in all directions
                directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
                
                for r in range(state.shape[0]):
                    for c in range(state.shape[1]):
                        if state[r, c] == original_player:
                            player_score += 1
                            
                            # Check for patterns
                            for dr, dc in directions:
                                count = 1
                                rr, cc = r + dr, c + dc
                                while 0 <= rr < state.shape[0] and 0 <= cc < state.shape[1] and state[rr, cc] == original_player:
                                    count += 1
                                    rr += dr
                                    cc += dc
                                
                                if count >= 5:
                                    player_score += 10
                                elif count >= 4:
                                    player_score += 5
                                elif count >= 3:
                                    player_score += 2
                        
                            # Check for connectivity (clusters of stones)
                            connectivity = 0
                            for dr in range(-1, 2):
                                for dc in range(-1, 2):
                                    if dr == 0 and dc == 0:
                                        continue
                                    nr, nc = r + dr, c + dc
                                    if 0 <= nr < state.shape[0] and 0 <= nc < state.shape[1] and state[nr, nc] == original_player:
                                        connectivity += 1
                        
                            player_score += connectivity * 0.5  # Bonus for connected stones
                        
                        elif state[r, c] == opponent:
                            opponent_score += 1
                            
                            # Check for patterns
                            for dr, dc in directions:
                                count = 1
                                rr, cc = r + dr, c + dc
                                while 0 <= rr < state.shape[0] and 0 <= cc < state.shape[1] and state[rr, cc] == opponent:
                                    count += 1
                                    rr += dr
                                    cc += dc
                                
                                if count >= 5:
                                    opponent_score += 10
                                elif count >= 4:
                                    opponent_score += 5
                                elif count >= 3:
                                    opponent_score += 2
                        
                            # Check for connectivity
                            connectivity = 0
                            for dr in range(-1, 2):
                                for dc in range(-1, 2):
                                    if dr == 0 and dc == 0:
                                        continue
                                    nr, nc = r + dr, c + dc
                                    if 0 <= nr < state.shape[0] and 0 <= nc < state.shape[1] and state[nr, nc] == opponent:
                                        connectivity += 1
                        
                            opponent_score += connectivity * 0.5  # Bonus for connected stones
                
                # Return winner based on score comparison
                if player_score > opponent_score:
                    return original_player
                elif opponent_score > player_score:
                    return opponent
                else:
                    return 0  # Draw
            
            def backpropagate(self, result):
                """Update statistics for this node and all parent nodes."""
                self.visits += 1
                
                # Update wins based on the result
                if result == self.player:
                    self.wins += 1
                elif result == 0:  # Draw
                    self.wins += 0.5
                
                # Backpropagate to parent
                if self.parent:
                    self.parent.backpropagate(result)
        
        # Create the root node with a reference to the game object for pattern analysis
        root = Node(deepcopy(self.board), player=opponent, game=self)  # Parent node has the opponent's perspective
        
        # Run MCTS for the specified number of iterations
        for _ in range(iterations):
            # Selection: Select a node to expand
            node = root
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.select_child(exploration_weight)
            
            # Expansion: If the node is not terminal and not fully expanded, expand it
            if not node.is_terminal() and not node.is_fully_expanded():
                # Choose a random untried move
                if node.untried_moves:
                    move = random.choice(node.untried_moves)
                    
                    # Expand the node
                    next_player = 3 - node.player if node.player else player
                    node = node.expand(node.state, move, next_player)
            
            # Simulation: Simulate a random game from this node
            result = node.rollout(3 - node.player)  # Next player after the one who made the move
            
            # Backpropagation: Update statistics for all nodes in the path
            node.backpropagate(result)
        
        # Choose the best move based on the most visited child
        if not root.children:
            # If no children (shouldn't happen), choose a random valid move
            valid_moves = [(r, c) for r in range(self.board.shape[0]) 
                          for c in range(self.board.shape[1]) if self.board[r, c] == 0]
            return random.choice(valid_moves) if valid_moves else None
        
        # Choose the child with the highest number of visits
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move


    def check_win_state(self, board):
        """Checks if a player has won on the given board state.
        Returns:
        0 - No winner yet
        1 - Black wins
        2 - White wins
        """
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for r in range(self.size):
            for c in range(self.size):
                if board[r, c] != 0:
                    current_color = board[r, c]
                    for dr, dc in directions:
                        prev_r, prev_c = r - dr, c - dc
                        if 0 <= prev_r < self.size and 0 <= prev_c < self.size and board[prev_r, prev_c] == current_color:
                            continue
                        count = 0
                        rr, cc = r, c
                        while 0 <= rr < self.size and 0 <= cc < self.size and board[rr, cc] == current_color:
                            count += 1
                            rr += dr
                            cc += dc
                        if count >= 6:
                            return current_color
        return 0

    def show_board(self):
        """Displays the board as text."""
        print("= ")
        for row in range(self.size - 1, -1, -1):
            line = f"{row+1:2} " + " ".join("X" if self.board[row, col] == 1 else "O" if self.board[row, col] == 2 else "." for col in range(self.size))
            print(line, file=sys.stderr)
        col_labels = "   " + " ".join(self.index_to_label(i) for i in range(self.size))
        print(col_labels, file=sys.stderr)
        print(flush=True)

    def list_commands(self):
        """Lists all available commands."""
        print("= ", flush=True)  

    def process_command(self, command):
        """Parses and executes GTP commands."""
        command = command.strip()
        if command == "get_conf_str env_board_size:":
            print("= 19", flush=True)
            return

        if not command:
            return
        
        parts = command.split()
        cmd = parts[0].lower()

        if cmd == "boardsize":
            try:
                size = int(parts[1])
                self.set_board_size(size)
            except ValueError:
                print("? Invalid board size", flush=True)
        elif cmd == "clear_board":
            self.reset_board()
        elif cmd == "play":
            if len(parts) < 3:
                print("? Invalid play command format", flush=True)
            else:
                self.play_move(parts[1], parts[2])
        elif cmd == "genmove":
            if len(parts) < 2:
                print("? Invalid genmove command format", flush=True)
            else:
                self.generate_move(parts[1])
        elif cmd == "showboard":
            self.show_board()
        elif cmd == "list_commands":
            self.list_commands()
        elif cmd == "quit":
            print("= ", flush=True)
            sys.exit(0)
        else:
            print("? Unsupported command", flush=True)

    def run(self):
        """Main loop that reads GTP commands from standard input."""
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                self.process_command(line)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"? Error: {str(e)}")

    def find_winning_move(self, state, player):
        """Find a move that would create a winning position (6 in a row) with one or two stones."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        # First check for immediate wins with a single stone (5 in a row already)
        single_stone_win = self.find_single_stone_win(state, player)
        if single_stone_win:
            return single_stone_win
        
        # Check for opponent's 4-in-a-row with open ends (critical to block)
        opponent = 3 - player
        opponent_threat = self.find_opponent_threat(state, opponent)
        if opponent_threat:
            return opponent_threat
        
        # Find all player's stones
        player_stones = [(r, c) for r in range(state.shape[0]) 
                        for c in range(state.shape[1]) if state[r, c] == player]
        
        # For each stone, check for potential winning patterns with two stones
        for r, c in player_stones:
            for dr, dc in directions:
                # Check if this stone is at the start of a pattern
                prev_r, prev_c = r - dr, c - dc
                if 0 <= prev_r < state.shape[0] and 0 <= prev_c < state.shape[1] and state[prev_r, prev_c] == player:
                    continue  # Not at the start
                
                # Count consecutive stones and track empty positions
                count = 0
                empty_positions = []
                
                for i in range(7):  # Check 7 positions (to find patterns that could become 6)
                    rr, cc = r + i*dr, c + i*dc
                    if 0 <= rr < state.shape[0] and 0 <= cc < state.shape[1]:
                        if state[rr, cc] == player:
                            count += 1
                        elif state[rr, cc] == 0:
                            empty_positions.append((rr, cc))
                        else:
                            break  # Opponent's stone
                    else:
                        break  # Out of bounds
                
                # If we have 4 stones and two empty positions, filling them would create a win
                if count == 4 and len(empty_positions) == 2:
                    # Check if the empty positions are consecutive and would create 6 in a row
                    if self.would_create_six(state, player, empty_positions, dr, dc):
                        return empty_positions[0]  # Return the first empty position
        
        return None  # No winning move found

    def find_opponent_threat(self, state, opponent):
        """Find and block opponent's threats (4+ consecutive stones with open ends or one-side blocked)."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        # Find all opponent's stones
        opponent_stones = [(r, c) for r in range(state.shape[0]) 
                          for c in range(state.shape[1]) if state[r, c] == opponent]
        
        # For each opponent stone, check for dangerous patterns
        for r, c in opponent_stones:
            for dr, dc in directions:
                # Check if this stone is at the start of a pattern
                prev_r, prev_c = r - dr, c - dc
                if 0 <= prev_r < state.shape[0] and 0 <= prev_c < state.shape[1] and state[prev_r, prev_c] == opponent:
                    continue  # Not at the start
                
                # Count consecutive stones and track empty positions
                count = 0
                empty_positions = []
                
                for i in range(7):  # Check 7 positions
                    rr, cc = r + i*dr, c + i*dc
                    if 0 <= rr < state.shape[0] and 0 <= cc < state.shape[1]:
                        if state[rr, cc] == opponent:
                            count += 1
                        elif state[rr, cc] == 0:
                            empty_positions.append((rr, cc))
                        else:
                            break  # Our stone
                    else:
                        break  # Out of bounds
                
                # Check for critical threats:
                
                # 1. Opponent has 4+ consecutive stones with open ends or one-side blocked
                if count >= 4 and empty_positions:
                    # Check if the empty positions are at the ends or within the sequence
                    # and would allow opponent to create 6 in a row
                    if self.is_critical_threat(state, opponent, count, empty_positions, r, c, dr, dc):
                        return empty_positions[0]  # Block the first empty position
                
                # 2. Opponent has 5 consecutive stones with one empty position
                if count == 5 and len(empty_positions) == 1:
                    return empty_positions[0]  # Must block this immediately
                
                # 3. Opponent has 3 consecutive stones with potential to create a critical threat
                if count == 3 and len(empty_positions) >= 3:
                    if self.is_critical_threat(state, opponent, count, empty_positions, r, c, dr, dc):
                        return empty_positions[0]  # Block the first empty position
        
        return None

    def is_critical_threat(self, state, opponent, count, empty_positions, start_r, start_c, dr, dc):
        """Determine if a pattern is a critical threat that must be blocked."""
        # If opponent has 5 in a row with an empty position, it's definitely critical
        if count == 5:
            return True
        
        # For 4 in a row, check if the empty positions would allow a win
        if count == 4:
            # Check if there are empty positions at both ends
            # This would allow the opponent to place two stones and win
            
            # Check before the start
            before_start = (start_r - dr, start_c - dc)
            before_start_empty = (0 <= before_start[0] < state.shape[0] and 
                                 0 <= before_start[1] < state.shape[1] and 
                                 state[before_start] == 0)
            
            # Check after the end
            end_r, end_c = start_r + count*dr, start_c + count*dc
            after_end = (end_r, end_c)
            after_end_empty = (0 <= after_end[0] < state.shape[0] and 
                              0 <= after_end[1] < state.shape[1] and 
                              state[after_end] == 0)
            
            # If both ends are empty, this is a critical threat
            if before_start_empty and after_end_empty:
                return True
            
            # Check for one side blocked, but two consecutive empty spaces on the other side
            # This is also critical because opponent can place two stones to win
            if before_start_empty and not after_end_empty:
                # Check for two empty spaces before the start
                second_before = (start_r - 2*dr, start_c - 2*dc)
                second_before_empty = (0 <= second_before[0] < state.shape[0] and 
                                      0 <= second_before[1] < state.shape[1] and 
                                      state[second_before] == 0)
                if second_before_empty:
                    return True
                
            if after_end_empty and not before_start_empty:
                # Check for two empty spaces after the end
                second_after = (end_r + dr, end_c + dc)
                second_after_empty = (0 <= second_after[0] < state.shape[0] and 
                                     0 <= second_after[1] < state.shape[1] and 
                                     state[second_after] == 0)
                if second_after_empty:
                    return True
            
            # Check for gaps within the sequence that could be filled
            # For example: O O _ O O _ (where O is opponent, _ is empty)
            temp_state = state.copy()
            
            # Try placing opponent stones in the empty positions
            for pos in empty_positions[:2]:  # Consider at most 2 empty positions
                temp_state[pos] = opponent
                
            # Check if this would create a win for the opponent
            if self.check_win_state(temp_state) == opponent:
                return True
        
        # Also check for 3 in a row with 3 empty spaces in a row
        # This could allow opponent to place 2 stones and then win on next turn
        if count == 3:
            # Check for three consecutive empty spaces at either end
            
            # Check before the start
            empty_before = []
            for i in range(1, 4):  # Check 3 positions before
                pos = (start_r - i*dr, start_c - i*dc)
                if (0 <= pos[0] < state.shape[0] and 
                    0 <= pos[1] < state.shape[1] and 
                    state[pos] == 0):
                    empty_before.append(pos)
                else:
                    break
                
            # Check after the end
            empty_after = []
            end_r, end_c = start_r + count*dr, start_c + count*dc
            for i in range(0, 3):  # Check 3 positions after
                pos = (end_r + i*dr, end_c + i*dc)
                if (0 <= pos[0] < state.shape[0] and 
                    0 <= pos[1] < state.shape[1] and 
                    state[pos] == 0):
                    empty_after.append(pos)
                else:
                    break
            
            # If we have 3 empty spaces on either side, this is potentially critical
            if len(empty_before) >= 3 or len(empty_after) >= 3:
                # Create a temporary board to test if placing 2 stones would create a critical threat
                temp_state = state.copy()
                
                # Try placing opponent stones in the first 2 empty positions
                if len(empty_before) >= 2:
                    for i in range(2):
                        temp_state[empty_before[i]] = opponent
                    
                    # Check if this creates a critical 5-in-a-row threat
                    for r, c in empty_before[:2]:
                        temp_count = 0
                        for i in range(6):
                            rr, cc = r + i*dr, c + i*dc
                            if (0 <= rr < state.shape[0] and 
                                0 <= cc < state.shape[1] and 
                                temp_state[rr, cc] == opponent):
                                temp_count += 1
                            else:
                                break
                        if temp_count >= 5:
                            return True
                
                # Reset and try the other end
                temp_state = state.copy()
                if len(empty_after) >= 2:
                    for i in range(2):
                        temp_state[empty_after[i]] = opponent
                    
                    # Check if this creates a critical 5-in-a-row threat
                    for r, c in empty_after[:2]:
                        temp_count = 0
                        for i in range(6):
                            rr, cc = r - i*dr, c - i*dc  # Reverse direction
                            if (0 <= rr < state.shape[0] and 
                                0 <= cc < state.shape[1] and 
                                temp_state[rr, cc] == opponent):
                                temp_count += 1
                            else:
                                break
                        if temp_count >= 5:
                            return True
        
        return False

    def find_single_stone_win(self, state, player):
        """Find a move that would create a winning position with a single stone (5 in a row already)."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        # Find all player's stones
        player_stones = [(r, c) for r in range(state.shape[0]) 
                        for c in range(state.shape[1]) if state[r, c] == player]
        
        # For each stone, check for potential winning patterns
        for r, c in player_stones:
            for dr, dc in directions:
                # Check if this stone is at the start of a pattern
                prev_r, prev_c = r - dr, c - dc
                if 0 <= prev_r < state.shape[0] and 0 <= prev_c < state.shape[1] and state[prev_r, prev_c] == player:
                    continue  # Not at the start
                
                # Count consecutive stones and track empty positions
                count = 0
                empty_positions = []
                
                for i in range(6):  # Check 6 positions (Connect6)
                    rr, cc = r + i*dr, c + i*dc
                    if 0 <= rr < state.shape[0] and 0 <= cc < state.shape[1]:
                        if state[rr, cc] == player:
                            count += 1
                        elif state[rr, cc] == 0:
                            empty_positions.append((rr, cc))
                        else:
                            break  # Opponent's stone
                    else:
                        break  # Out of bounds
                
                # If we have 5 stones and one empty position, filling it would create a win
                if count == 5 and len(empty_positions) == 1:
                    return empty_positions[0]
        
        return None

    def would_create_six(self, state, player, empty_positions, dr, dc):
        """Check if filling the empty positions would create 6 in a row."""
        # Create a temporary board to test the move
        temp_state = state.copy()
        
        # Place stones at the empty positions
        for r, c in empty_positions:
            temp_state[r, c] = player
        
        # Check if this creates a win
        return self.check_win_state(temp_state) == player

    def find_critical_threat(self, state, player):
        """Find critical threats that must be blocked immediately."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        # First check for "Five Plus One" pattern (5 in a row with an empty space)
        five_plus_one = self.find_single_stone_win(state, player)
        if five_plus_one:
            return five_plus_one
        
        # Check for "Four Plus Two" pattern (4 in a row with empty spaces on both sides)
        for r in range(state.shape[0]):
            for c in range(state.shape[1]):
                if state[r, c] == player:
                    for dr, dc in directions:
                        # Skip if not the start of a pattern
                        prev_r, prev_c = r - dr, c - dc
                        if 0 <= prev_r < state.shape[0] and 0 <= prev_c < state.shape[1] and state[prev_r, prev_c] == player:
                            continue
                        
                        # Count consecutive stones
                        count = 0
                        rr, cc = r, c
                        while 0 <= rr < state.shape[0] and 0 <= cc < state.shape[1] and state[rr, cc] == player:
                            count += 1
                            rr += dr
                            cc += dc
                        
                        # Check if we have 4 in a row
                        if count == 4:
                            # Check if both ends are open
                            forward_pos = (rr, cc)
                            backward_pos = (r - dr, c - dc)
                            
                            forward_open = (0 <= forward_pos[0] < state.shape[0] and 
                                           0 <= forward_pos[1] < state.shape[1] and 
                                           state[forward_pos] == 0)
                            
                            backward_open = (0 <= backward_pos[0] < state.shape[0] and 
                                            0 <= backward_pos[1] < state.shape[1] and 
                                            state[backward_pos] == 0)
                            
                            if forward_open and backward_open:
                                return forward_pos  # Return one of the open ends to block
        
        # Check for "Four Plus Gap" pattern (4 stones with a single gap)
        for r in range(state.shape[0]):
            for c in range(state.shape[1]):
                if state[r, c] == player:
                    for dr, dc in directions:
                        # Skip if not the start of a pattern
                        prev_r, prev_c = r - dr, c - dc
                        if 0 <= prev_r < state.shape[0] and 0 <= prev_c < state.shape[1] and state[prev_r, prev_c] == player:
                            continue
                        
                        # Look for pattern with a gap
                        positions = []
                        gap_position = None
                        
                        rr, cc = r, c
                        for i in range(6):  # Check 6 positions
                            if not (0 <= rr < state.shape[0] and 0 <= cc < state.shape[1]):
                                break
                            
                            if state[rr, cc] == player:
                                positions.append((rr, cc))
                            elif state[rr, cc] == 0:
                                if gap_position is None:  # First gap
                                    gap_position = (rr, cc)
                                else:
                                    break  # Second gap, pattern doesn't match
                            else:
                                break  # Opponent stone
                            
                            rr += dr
                            cc += dc
                        
                        # Check if we have 4 stones with a single gap
                        if len(positions) == 4 and gap_position is not None:
                            # Check if filling the gap would create a winning position
                            temp_state = state.copy()
                            temp_state[gap_position] = player
                            
                            # Check if this creates a win
                            if self.check_win_state(temp_state) == player:
                                return gap_position
        
        # Check for "Double Three" pattern
        threes_positions = []
        
        for r in range(state.shape[0]):
            for c in range(state.shape[1]):
                if state[r, c] == player:
                    for dr, dc in directions:
                        prev_r, prev_c = r - dr, c - dc
                        if 0 <= prev_r < state.shape[0] and 0 <= prev_c < state.shape[1] and state[prev_r, prev_c] == player:
                            continue
                        
                        count = 0
                        positions = []
                        rr, cc = r, c
                        
                        while 0 <= rr < state.shape[0] and 0 <= cc < state.shape[1] and state[rr, cc] == player:
                            count += 1
                            positions.append((rr, cc))
                            rr += dr
                            cc += dc
                        
                        # Check if forward end is open
                        forward_open = (0 <= rr < state.shape[0] and 0 <= cc < state.shape[1] and state[rr, cc] == 0)
                        
                        # Check if backward end is open
                        back_r, back_c = r - dr, c - dc
                        backward_open = (0 <= back_r < state.shape[0] and 0 <= back_c < state.shape[1] and state[back_r, back_c] == 0)
                        
                        # If we have 3 in a row with both ends open, add to threes_positions
                        if count == 3 and forward_open and backward_open:
                            threes_positions.append((positions, (rr, cc), (back_r, back_c)))
        
        # Check for double three (two separate open threes)
        if len(threes_positions) >= 2:
            for i in range(len(threes_positions)):
                for j in range(i+1, len(threes_positions)):
                    positions_i, forward_i, backward_i = threes_positions[i]
                    positions_j, forward_j, backward_j = threes_positions[j]
                    
                    # Check if the two patterns don't share any stones
                    if not any(pos in positions_j for pos in positions_i):
                        # Return one of the open ends to block
                        return forward_i
        
        # Check for "Three-Three" pattern (3 stones with 3+ empty spaces on either side)
        for r in range(state.shape[0]):
            for c in range(state.shape[1]):
                if state[r, c] == player:
                    for dr, dc in directions:
                        prev_r, prev_c = r - dr, c - dc
                        if 0 <= prev_r < state.shape[0] and 0 <= prev_c < state.shape[1] and state[prev_r, prev_c] == player:
                            continue
                        
                        count = 0
                        rr, cc = r, c
                        
                        while 0 <= rr < state.shape[0] and 0 <= cc < state.shape[1] and state[rr, cc] == player:
                            count += 1
                            rr += dr
                            cc += dc
                        
                        if count == 3:
                            # Check for 3+ empty spaces on either side
                            forward_spaces = 0
                            forward_positions = []
                            
                            while 0 <= rr < state.shape[0] and 0 <= cc < state.shape[1] and state[rr, cc] == 0:
                                forward_spaces += 1
                                forward_positions.append((rr, cc))
                                rr += dr
                                cc += dc
                                if forward_spaces >= 3:
                                    break
                            
                            backward_spaces = 0
                            backward_positions = []
                            rr, cc = r - dr, c - dc
                            
                            while 0 <= rr < state.shape[0] and 0 <= cc < state.shape[1] and state[rr, cc] == 0:
                                backward_spaces += 1
                                backward_positions.append((rr, cc))
                                rr -= dr
                                cc -= dc
                                if backward_spaces >= 3:
                                    break
                            
                            if forward_spaces >= 3:
                                return forward_positions[0]  # Block the first empty space
                            elif backward_spaces >= 3:
                                return backward_positions[0]  # Block the first empty space
        
        # Check for "Four-Three Combination"
        fours = []
        threes = []
        
        # Find all open fours and threes
        for r in range(state.shape[0]):
            for c in range(state.shape[1]):
                if state[r, c] == player:
                    for dr, dc in directions:
                        prev_r, prev_c = r - dr, c - dc
                        if 0 <= prev_r < state.shape[0] and 0 <= prev_c < state.shape[1] and state[prev_r, prev_c] == player:
                            continue
                        
                        count = 0
                        positions = []
                        rr, cc = r, c
                        
                        while 0 <= rr < state.shape[0] and 0 <= cc < state.shape[1] and state[rr, cc] == player:
                            count += 1
                            positions.append((rr, cc))
                            rr += dr
                            cc += dc
                        
                        forward_open = (0 <= rr < state.shape[0] and 0 <= cc < state.shape[1] and state[rr, cc] == 0)
                        
                        back_r, back_c = r - dr, c - dc
                        backward_open = (0 <= back_r < state.shape[0] and 0 <= back_c < state.shape[1] and state[back_r, back_c] == 0)
                        
                        if count == 4 and (forward_open or backward_open):
                            fours.append((positions, (rr, cc) if forward_open else (back_r, back_c)))
                        elif count == 3 and forward_open and backward_open:
                            threes.append((positions, (rr, cc), (back_r, back_c)))
        
        # Check for four-three combination
        if fours and threes:
            for four_positions, four_open in fours:
                for three_positions, three_open1, three_open2 in threes:
                    # Check if the patterns don't share any stones
                    if not any(pos in three_positions for pos in four_positions):
                        # Block the four first
                        return four_open
        
        return None  # No critical threats found

    def count_broken_threes(self, state, player):
        """Count the number of Broken Three patterns (XX.X or X.XX) for a player."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        broken_three_count = 0
        
        for r in range(state.shape[0]):
            for c in range(state.shape[1]):
                if state[r, c] == player:
                    for dr, dc in directions:
                        # Skip if not the start of a pattern
                        prev_r, prev_c = r - dr, c - dc
                        if 0 <= prev_r < state.shape[0] and 0 <= prev_c < state.shape[1] and state[prev_r, prev_c] == player:
                            continue
                        
                        # Look for Broken Three patterns (XX.X or X.XX)
                        found_pattern = False
                        positions = []
                        gap_position = None
                        
                        # Check for XX.X pattern
                        if r + 3*dr < state.shape[0] and c + 3*dc < state.shape[1] and r + 3*dr >= 0 and c + 3*dc >= 0:
                            if (state[r, c] == player and 
                                state[r + dr, c + dc] == player and 
                                state[r + 2*dr, c + 2*dc] == 0 and 
                                state[r + 3*dr, c + 3*dc] == player):
                                found_pattern = True
                                positions = [(r, c), (r + dr, c + dc), (r + 3*dr, c + 3*dc)]
                                gap_position = (r + 2*dr, c + 2*dc)
                        
                        # Check for X.XX pattern
                        if not found_pattern and r + 3*dr < state.shape[0] and c + 3*dc < state.shape[1] and r + 3*dr >= 0 and c + 3*dc >= 0:
                            if (state[r, c] == player and 
                                state[r + dr, c + dc] == 0 and 
                                state[r + 2*dr, c + 2*dc] == player and 
                                state[r + 3*dr, c + 3*dc] == player):
                                found_pattern = True
                                positions = [(r, c), (r + 2*dr, c + 2*dc), (r + 3*dr, c + 3*dc)]
                                gap_position = (r + dr, c + dc)
                        
                        # If found, check if ends are open
                        if found_pattern:
                            # Check if forward end is open
                            forward_r, forward_c = r + 4*dr, c + 4*dc
                            forward_open = (0 <= forward_r < state.shape[0] and 
                                           0 <= forward_c < state.shape[1] and 
                                           state[forward_r, forward_c] == 0)
                            
                            # Check if backward end is open
                            backward_r, backward_c = r - dr, c - dc
                            backward_open = (0 <= backward_r < state.shape[0] and 
                                            0 <= backward_c < state.shape[1] and 
                                            state[backward_r, backward_c] == 0)
                            
                            # Count if at least one end is open
                            if forward_open or backward_open:
                                broken_three_count += 1
        
        return broken_three_count

    def count_broken_fours(self, state, player):
        """Count the number of Broken Four patterns (XXX.X, XX.XX, X.XXX) for a player."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        broken_four_count = 0
        
        for r in range(state.shape[0]):
            for c in range(state.shape[1]):
                if state[r, c] == player:
                    for dr, dc in directions:
                        # Skip if not the start of a pattern
                        prev_r, prev_c = r - dr, c - dc
                        if 0 <= prev_r < state.shape[0] and 0 <= prev_c < state.shape[1] and state[prev_r, prev_c] == player:
                            continue
                        
                        # Look for Broken Four patterns (XXX.X, XX.XX, X.XXX)
                        found_pattern = False
                        positions = []
                        gap_position = None
                        
                        # Check for XXX.X pattern
                        if r + 4*dr < state.shape[0] and c + 4*dc < state.shape[1] and r + 4*dr >= 0 and c + 4*dc >= 0:
                            if (state[r, c] == player and 
                                state[r + dr, c + dc] == player and 
                                state[r + 2*dr, c + 2*dc] == player and 
                                state[r + 3*dr, c + 3*dc] == 0 and 
                                state[r + 4*dr, c + 4*dc] == player):
                                found_pattern = True
                                positions = [(r, c), (r + dr, c + dc), (r + 2*dr, c + 2*dc), (r + 4*dr, c + 4*dc)]
                                gap_position = (r + 3*dr, c + 3*dc)
                        
                        # Check for XX.XX pattern
                        if not found_pattern and r + 4*dr < state.shape[0] and c + 4*dc < state.shape[1] and r + 4*dr >= 0 and c + 4*dc >= 0:
                            if (state[r, c] == player and 
                                state[r + dr, c + dc] == player and 
                                state[r + 2*dr, c + 2*dc] == 0 and 
                                state[r + 3*dr, c + 3*dc] == player and 
                                state[r + 4*dr, c + 4*dc] == player):
                                found_pattern = True
                                positions = [(r, c), (r + dr, c + dc), (r + 3*dr, c + 3*dc), (r + 4*dr, c + 4*dc)]
                                gap_position = (r + 2*dr, c + 2*dc)
                        
                        # Check for X.XXX pattern
                        if not found_pattern and r + 4*dr < state.shape[0] and c + 4*dc < state.shape[1] and r + 4*dr >= 0 and c + 4*dc >= 0:
                            if (state[r, c] == player and 
                                state[r + dr, c + dc] == 0 and 
                                state[r + 2*dr, c + 2*dc] == player and 
                                state[r + 3*dr, c + 3*dc] == player and 
                                state[r + 4*dr, c + 4*dc] == player):
                                found_pattern = True
                                positions = [(r, c), (r + 2*dr, c + 2*dc), (r + 3*dr, c + 3*dc), (r + 4*dr, c + 4*dc)]
                                gap_position = (r + dr, c + dc)
                        
                        # If found, check if ends are open
                        if found_pattern:
                            # Check if forward end is open
                            forward_r, forward_c = r + 5*dr, c + 5*dc
                            forward_open = (0 <= forward_r < state.shape[0] and 
                                           0 <= forward_c < state.shape[1] and 
                                           state[forward_r, forward_c] == 0)
                            
                            # Check if backward end is open
                            backward_r, backward_c = r - dr, c - dc
                            backward_open = (0 <= backward_r < state.shape[0] and 
                                            0 <= backward_c < state.shape[1] and 
                                            state[backward_r, backward_c] == 0)
                            
                            # Count if at least one end is open
                            if forward_open or backward_open:
                                broken_four_count += 1
        
        return broken_four_count

if __name__ == "__main__":
    game = Connect6Game()
    game.run()