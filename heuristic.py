# completely_fixed_heuristic.py
import numpy as np
import random
import hashlib

class SungkaHeuristic:
    def __init__(self, game):
        self.original_game = game
        # Add instance-specific randomization to break determinism
        self.instance_id = random.randint(1, 1000000)

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
            
            # Check for capture
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

    def analyze_opponent_threats(self, game, board_after):
        """Analyze immediate threats from opponent"""
        current_player = game.current_player
        opponent = 1 - current_player
        threat_score = 0
        
        opponent_range = range(8, 15) if current_player == 0 else range(0, 7)
        
        for opp_hole in opponent_range:
            if board_after[opp_hole] == 0 or opp_hole in game.burned_holes[opponent]:
                continue
                
            stones = board_after[opp_hole]
            
            # Simple simulation of opponent's potential move
            current_hole = opp_hole
            temp_stones = stones
            
            while temp_stones > 0:
                current_hole = (current_hole + 1) % 16
                
                # Skip current player's head (opponent's perspective)
                if (opponent == 0 and current_hole == 15) or (opponent == 1 and current_hole == 7):
                    continue
                    
                # Skip burned holes
                if current_hole in game.burned_holes[0] or current_hole in game.burned_holes[1]:
                    continue
                    
                temp_stones -= 1
            
            # Check what opponent could achieve
            if ((opponent == 0 and current_hole == 7) or (opponent == 1 and current_hole == 15)):
                threat_score -= 5  # Opponent gets extra turn
            elif ((opponent == 0 and 0 <= current_hole <= 6) or 
                  (opponent == 1 and 8 <= current_hole <= 14)):
                opposite = 14 - current_hole
                if board_after[current_hole] == 1 and board_after[opposite] > 0:
                    threat_score -= board_after[opposite] * 2  # Opponent can capture
        
        return threat_score

    def evaluate_endgame_strategy(self, game, board_after):
        """Enhanced endgame evaluation"""
        current_player = game.current_player
        
        if current_player == 0:
            my_head = board_after[7]
            opponent_head = board_after[15]
            my_stones = sum(board_after[0:7])
            opponent_stones = sum(board_after[8:15])
        else:
            my_head = board_after[15]
            opponent_head = board_after[7]
            my_stones = sum(board_after[8:15])
            opponent_stones = sum(board_after[0:7])
        
        total_remaining = my_stones + opponent_stones
        
        if total_remaining <= 20:  # Endgame threshold
            head_diff = my_head - opponent_head
            
            # If we're ahead, prioritize clearing our side safely
            if head_diff > 0:
                clear_bonus = max(0, (15 - my_stones)) * 3
                return head_diff * 15 + clear_bonus
            
            # If we're behind, prioritize gaining stones
            elif head_diff < 0:
                if my_stones > opponent_stones:
                    # We have more stones to work with
                    comeback_bonus = (my_stones - opponent_stones) * 5
                    return head_diff * 15 + comeback_bonus
                else:
                    # Desperate situation
                    return head_diff * 20
            
            # If tied, prioritize maintaining material advantage
            else:
                return (my_stones - opponent_stones) * 8
        
        return 0

    def get_strategic_variation(self, board_after, hole, game_progress, move_number=0):
        """
        COMPLETELY REWRITTEN: Generate truly random variation to break determinism
        This adds controlled randomness to prevent identical games
        """
        if game_progress <= 0.1:
            return 0.0
        
        # Use multiple sources of variation
        base_random = random.random() * 4 - 2  # -2 to +2 range
        
        # Add small positional bias to prevent completely random play
        current_player = self.original_game.current_player
        if current_player == 0:
            relative_hole_pos = hole  # 0-6
        else:
            relative_hole_pos = hole - 8  # Normalize to 0-6 range
        
        # Small positional preference (much smaller than base random)
        positional_bias = (relative_hole_pos - 3) * 0.1  # Small bias toward middle holes
        
        # Game progress influence
        progress_factor = game_progress * 0.5
        
        # Combine factors
        total_variation = base_random + positional_bias + progress_factor
        
        # Clamp to reasonable range
        return max(-3, min(3, total_variation))

    def evaluate_move_verbose(self, hole):
        game = self.original_game
        current_player = game.current_player
        
        # Simulate the complete move
        result = self.simulate_move_complete(game, hole)
        if result is None:
            return -float('inf'), {"Error": "Invalid move"}

        board_after = result['board']
        scores = {}
        
        # Calculate game progress
        total_stones_on_board = sum(board_after[i] for i in range(16) if i not in (7, 15))
        game_progress = 1 - (total_stones_on_board / 98)
        
        # 1. Immediate tactical gains
        capture_score = result['total_captured'] * 12
        extra_turn_score = result['extra_turns'] * 20
        burn_penalty = result['burns_created'] * -30
        
        scores['Captures'] = capture_score
        scores['Extra Turns'] = extra_turn_score
        scores['Burn Penalty'] = burn_penalty
        
        # 2. Strategic position evaluation
        if current_player == 0:
            my_head = board_after[7]
            opponent_head = board_after[15]
            my_stones = sum(board_after[0:7])
            opponent_stones = sum(board_after[8:15])
            my_range = range(0, 7)
        else:
            my_head = board_after[15]
            opponent_head = board_after[7]
            my_stones = sum(board_after[8:15])
            opponent_stones = sum(board_after[0:7])
            my_range = range(8, 15)
        
        head_diff = my_head - opponent_head
        material_diff = my_stones - opponent_stones
        
        # Dynamic scoring based on game phase
        if game_progress < 0.3:  # Early game
            scores['Head Advantage'] = head_diff * 8
            scores['Material Control'] = material_diff * 3
            
            # Reward maintaining options
            active_holes = sum(1 for i in my_range if board_after[i] > 0 and i not in game.burned_holes[current_player])
            if active_holes >= 5:
                scores['Development'] = 8
            elif active_holes <= 2:
                scores['Development'] = -15
            else:
                scores['Development'] = 0
                
        elif game_progress > 0.6:  # Late game
            endgame_score = self.evaluate_endgame_strategy(game, board_after)
            scores['Endgame Strategy'] = endgame_score
            scores['Material Control'] = material_diff * 1
            
        else:  # Mid game
            scores['Head Advantage'] = head_diff * 10
            scores['Material Control'] = material_diff * 2
            
            # Mid-game tactical focus
            active_holes = sum(1 for i in my_range if board_after[i] > 0 and i not in game.burned_holes[current_player])
            if active_holes >= 3:
                scores['Flexibility'] = 5
            else:
                scores['Flexibility'] = -8
        
        # 3. Threat analysis
        threat_score = self.analyze_opponent_threats(game, board_after)
        scores['Threat Analysis'] = threat_score
        
        # 4. Move efficiency and quality
        stones_used = game.board[hole]
        efficiency_score = 0
        
        # Reward efficient captures and extra turns
        if result['total_captured'] > 0:
            efficiency = result['total_captured'] / max(1, stones_used)
            efficiency_score += efficiency * 8
        
        if result['extra_turns'] > 0:
            if stones_used <= 6:
                efficiency_score += 6  # Efficient extra turn
            else:
                efficiency_score += 2  # Less efficient but still good
        
        # Penalty for wasteful large moves
        if stones_used > 12 and result['total_captured'] == 0 and result['extra_turns'] == 0:
            efficiency_score -= 8
        
        scores['Move Efficiency'] = efficiency_score
        
        # 5. Advanced tactical considerations
        tactical_score = 0
        
        # Count immediate capture opportunities after this move
        immediate_captures = 0
        for next_hole in my_range:
            if (board_after[next_hole] == 0 or 
                next_hole in game.burned_holes[current_player]):
                continue
            
            next_stones = board_after[next_hole]
            current_sim_hole = next_hole
            temp_stones = next_stones
            
            # Simulate the move
            while temp_stones > 0:
                current_sim_hole = (current_sim_hole + 1) % 16
                
                # Skip opponent's head
                if ((current_player == 0 and current_sim_hole == 15) or
                    (current_player == 1 and current_sim_hole == 7)):
                    continue
                    
                # Skip burned holes
                if (current_sim_hole in game.burned_holes[0] or 
                    current_sim_hole in game.burned_holes[1]):
                    continue
                    
                temp_stones -= 1
            
            # Check if we can capture
            if ((current_player == 0 and 0 <= current_sim_hole <= 6) or
                (current_player == 1 and 8 <= current_sim_hole <= 14)):
                if board_after[current_sim_hole] == 0:
                    opposite = 14 - current_sim_hole
                    if board_after[opposite] > 0:
                        immediate_captures += 1
                        tactical_score += board_after[opposite] * 0.5
        
        scores['Tactical Setup'] = tactical_score
        
        # 6. FIXED: Add controlled randomness to break determinism
        variation = self.get_strategic_variation(board_after, hole, game_progress)
        scores['Strategic Variation'] = variation
        
        # Total score calculation
        total_score = sum(scores.values())
        
        # Add context info
        scores['Stones Used'] = stones_used
        scores['Total Score'] = total_score
        
        return total_score, scores  