# aggressive_heuristic.py
import numpy as np

class SungkaHeuristic:
    def __init__(self, game):
        self.original_game = game

    def simulate_move_complete(self, game, hole):
        """Complete move simulation with proper burned hole handling and relay"""
        board = game.board.copy()
        current_player = game.current_player
        burned_holes = {0: set(game.burned_holes[0]), 1: set(game.burned_holes[1])}

        if hole in burned_holes[current_player]:
            return None
        if not ((current_player == 0 and 0 <= hole <= 6 and board[hole] > 0) or
                (current_player == 1 and 8 <= hole <= 14 and board[hole] > 0)):
            return None

        originally_empty = set(i for i in range(16) if board[i] == 0)
        
        total_captured = 0
        extra_turns = 0
        burns_created = 0
        
        def distribute_from_hole(start_hole):
            nonlocal total_captured, extra_turns, burns_created
            
            stones = board[start_hole]
            board[start_hole] = 0
            current_hole = start_hole

            while stones > 0:
                current_hole = (current_hole + 1) % 16
                
                # Skip opponent's head
                if (current_player == 0 and current_hole == 15) or (current_player == 1 and current_hole == 7):
                    continue
                    
                # Skip burned holes
                if current_hole in burned_holes[0] or current_hole in burned_holes[1]:
                    continue
                    
                board[current_hole] += 1
                stones -= 1

            # Check for extra turn
            if (current_player == 0 and current_hole == 7) or (current_player == 1 and current_hole == 15):
                extra_turns += 1
                return current_hole, True
            
            # Check for relay (continue distribution)
            if current_hole not in (7, 15) and board[current_hole] > 1:
                return distribute_from_hole(current_hole)
            
            # Check for capture - FIXED: Use originally_empty from the ORIGINAL move
            if ((current_player == 0 and 0 <= current_hole <= 6 and board[current_hole] == 1) or
                (current_player == 1 and 8 <= current_hole <= 14 and board[current_hole] == 1)):
                opposite_hole = 14 - current_hole
                if board[opposite_hole] > 0:
                    # Capture occurs
                    captured = board[current_hole] + board[opposite_hole]
                    total_captured += captured
                    head = 7 if current_player == 0 else 15
                    board[head] += captured
                    board[current_hole] = 0
                    board[opposite_hole] = 0
                elif board[opposite_hole] == 0:
                    # Sunog occurs - but only if landing in originally empty hole
                    if current_hole in originally_empty:
                        seeds = board[current_hole]
                        board[current_hole] = 0
                        opponent_head = 15 if current_player == 0 else 7
                        board[opponent_head] += seeds
                        burned_holes[current_player].add(current_hole)
                        burns_created += 1

            return current_hole, False

        last_hole, got_extra_turn = distribute_from_hole(hole)
        
        return {
            'board': board,
            'burned_holes': burned_holes,
            'total_captured': total_captured,
            'extra_turns': extra_turns,
            'burns_created': burns_created,
            'last_hole': last_hole
        }

    def evaluate_board_position(self, game, board_after):
        """Evaluate overall board position strength"""
        current_player = game.current_player
        score = 0
        
        # 1. Head advantage (most important)
        if current_player == 0:
            head_diff = board_after[7] - board_after[15]
            own_stones = sum(board_after[0:7])
            opponent_stones = sum(board_after[8:15])
        else:
            head_diff = board_after[15] - board_after[7]
            own_stones = sum(board_after[8:15])
            opponent_stones = sum(board_after[0:7])
        
        score += head_diff * 15  # Very high weight for head advantage
        
        # 2. Material advantage on board
        material_diff = own_stones - opponent_stones
        score += material_diff * 3
        
        # 3. Positional control - stones in key holes
        if current_player == 0:
            key_holes = [3, 4, 5]  # Center-right holes for Player 1
            key_stones = sum(board_after[i] for i in key_holes)
        else:
            key_holes = [9, 10, 11]  # Center-left holes for Player 2
            key_stones = sum(board_after[i] for i in key_holes)
        
        score += key_stones * 2
        
        # 4. Opponent mobility restriction
        opponent_range = range(8, 15) if current_player == 0 else range(0, 7)
        opponent_moves = sum(1 for i in opponent_range 
                           if board_after[i] > 0 and i not in game.burned_holes[1-current_player])
        
        if opponent_moves <= 3:
            score += (4 - opponent_moves) * 20  # Huge bonus for restricting opponent
        
        return score

    def count_capture_opportunities(self, game, board):
        """Count immediate capture opportunities"""
        current_player = game.current_player
        captures = 0
        capture_value = 0
        
        own_range = range(0, 7) if current_player == 0 else range(8, 15)
        
        for hole in own_range:
            if hole in game.burned_holes[current_player] or board[hole] == 0:
                continue
                
            stones = board[hole]
            # Simple landing calculation (ignoring burned holes for quick estimate)
            landing = (hole + stones) % 14
            if current_player == 1:
                landing = max(0, landing - 8) + 8
                
            if ((current_player == 0 and 0 <= landing <= 6) or
                (current_player == 1 and 8 <= landing <= 14)):
                if board[landing] == 0:  # Empty hole
                    opposite = 14 - landing
                    if board[opposite] > 0:
                        captures += 1
                        capture_value += board[opposite] + 1
        
        return captures, capture_value

    def evaluate_endgame(self, game, board):
        """Special evaluation for endgame positions"""
        current_player = game.current_player
        own_range = range(0, 7) if current_player == 0 else range(8, 15)
        opponent_range = range(8, 15) if current_player == 0 else range(0, 7)
        
        own_stones = sum(board[i] for i in own_range)
        opponent_stones = sum(board[i] for i in opponent_range)
        total_stones = own_stones + opponent_stones
        
        if total_stones <= 15:  # Endgame threshold
            # Focus on head advantage and clearing our side
            head_diff = board[7] - board[15] if current_player == 0 else board[15] - board[7]
            
            # Bonus for having fewer stones on our side (easier to clear)
            clear_bonus = max(0, (8 - own_stones)) * 5
            
            return head_diff * 25 + clear_bonus
        
        return 0

    def evaluate_move_verbose(self, hole):
        game = self.original_game
        current_player = game.current_player
        
        # Simulate the complete move
        result = self.simulate_move_complete(game, hole)
        if result is None:
            return -float('inf'), {"Error": "Invalid move"}

        board_after = result['board']
        
        # Core scoring components
        scores = {}
        
        # 1. Immediate gains (captures and extra turns)
        capture_score = result['total_captured'] * 25
        extra_turn_score = result['extra_turns'] * 40
        burn_penalty = result['burns_created'] * -15
        
        scores['Captures'] = capture_score
        scores['Extra Turns'] = extra_turn_score
        scores['Burn Penalty'] = burn_penalty
        
        # 2. Positional evaluation
        position_score = self.evaluate_board_position(game, board_after)
        scores['Position'] = position_score
        
        # 3. Future opportunities
        capture_ops, capture_val = self.count_capture_opportunities(game, board_after)
        opportunity_score = capture_ops * 8 + capture_val * 2
        scores['Opportunities'] = opportunity_score
        
        # 4. Endgame evaluation
        endgame_score = self.evaluate_endgame(game, board_after)
        scores['Endgame'] = endgame_score
        
        # 5. Aggressive play bonus
        aggressive_bonus = 0
        if result['total_captured'] > 0:
            aggressive_bonus += 15  # Reward any capture
        if result['extra_turns'] > 0:
            aggressive_bonus += 20  # Reward extra turns
        
        scores['Aggression'] = aggressive_bonus
        
        # 6. Risk assessment
        risk_penalty = 0
        own_range = range(0, 7) if current_player == 0 else range(8, 15)
        stones_after_move = sum(board_after[i] for i in own_range)
        
        # Penalty for leaving too few options
        if stones_after_move <= 3:
            available_moves = sum(1 for i in own_range if board_after[i] > 0)
            if available_moves <= 2:
                risk_penalty = -30
        
        scores['Risk'] = risk_penalty
        
        # Total score calculation
        total_score = sum(scores.values())
        
        # Add original stones for context
        scores['Stones Used'] = game.board[hole]
        scores['Total Score'] = total_score
        
        return total_score, scores