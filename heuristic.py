# enhanced_heuristic.py - SIGNIFICANTLY IMPROVED VERSION
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
        capture_opportunities = 0  # Track potential future captures
        
        def distribute_from_hole(start_hole):
            nonlocal total_captured, extra_turns, burns_created, capture_opportunities
            
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
        
        # Count potential future capture setups
        my_range = range(0, 7) if current_player == 0 else range(8, 15)
        for my_hole in my_range:
            if board[my_hole] == 0 and my_hole not in burned_holes[current_player]:
                opposite = 14 - my_hole
                if board[opposite] > 0:
                    capture_opportunities += board[opposite]
        
        return {
            'board': board,
            'burned_holes': burned_holes,
            'total_captured': total_captured,
            'extra_turns': extra_turns,
            'burns_created': burns_created,
            'last_hole': last_hole,
            'capture_opportunities': capture_opportunities
        }

    def analyze_opponent_threats(self, game, board_after):
        """Enhanced opponent threat analysis"""
        current_player = game.current_player
        opponent = 1 - current_player
        threat_score = 0
        
        opponent_range = range(8, 15) if current_player == 0 else range(0, 7)
        
        for opp_hole in opponent_range:
            if board_after[opp_hole] == 0 or opp_hole in game.burned_holes[opponent]:
                continue
                
            stones = board_after[opp_hole]
            
            # Enhanced simulation of opponent's potential move
            current_hole = opp_hole
            temp_stones = stones
            temp_board = board_after.copy()
            
            while temp_stones > 0:
                current_hole = (current_hole + 1) % 16
                
                # Skip current player's head (opponent's perspective)
                if (opponent == 0 and current_hole == 15) or (opponent == 1 and current_hole == 7):
                    continue
                    
                # Skip burned holes
                if current_hole in game.burned_holes[0] or current_hole in game.burned_holes[1]:
                    continue
                    
                temp_board[current_hole] += 1
                temp_stones -= 1
            
            # Enhanced threat evaluation
            if ((opponent == 0 and current_hole == 7) or (opponent == 1 and current_hole == 15)):
                threat_score -= 8  # Higher penalty for opponent extra turns
            elif ((opponent == 0 and 0 <= current_hole <= 6) or 
                  (opponent == 1 and 8 <= current_hole <= 14)):
                opposite = 14 - current_hole
                if temp_board[current_hole] == 1 and board_after[opposite] > 0:
                    # Opponent can capture - penalize based on capture value
                    capture_value = board_after[opposite]
                    if capture_value >= 8:
                        threat_score -= capture_value * 3  # High penalty for big captures
                    else:
                        threat_score -= capture_value * 2
        
        return threat_score

    def evaluate_capture_potential(self, game, board_after):
        """Evaluate potential for future captures"""
        current_player = game.current_player
        my_range = range(0, 7) if current_player == 0 else range(8, 15)
        
        capture_potential = 0
        
        for my_hole in my_range:
            if board_after[my_hole] == 0 and my_hole not in game.burned_holes[current_player]:
                opposite = 14 - my_hole
                if board_after[opposite] > 0:
                    # This empty hole could capture opponent stones
                    stones_available = board_after[opposite]
                    if stones_available >= 8:
                        capture_potential += stones_available * 2  # High value targets
                    else:
                        capture_potential += stones_available * 1.5
        
        return capture_potential

    def evaluate_positional_strength(self, game, board_after):
        """Evaluate overall positional strength"""
        current_player = game.current_player
        my_range = range(0, 7) if current_player == 0 else range(8, 15)
        opponent_range = range(8, 15) if current_player == 0 else range(0, 7)
        
        positional_score = 0
        
        # Count active holes (non-zero, non-burned)
        my_active = sum(1 for i in my_range if board_after[i] > 0 and i not in game.burned_holes[current_player])
        opponent_active = sum(1 for i in opponent_range if board_after[i] > 0 and i not in game.burned_holes[1-current_player])
        
        # Reward maintaining more active holes than opponent
        positional_score += (my_active - opponent_active) * 3
        
        # Evaluate stone distribution quality
        my_stones = [board_after[i] for i in my_range if i not in game.burned_holes[current_player]]
        if my_stones:
            # Reward balanced distribution (avoid too many large piles)
            stone_variance = np.var(my_stones) if len(my_stones) > 1 else 0
            if stone_variance < 20:  # Well distributed
                positional_score += 5
            elif stone_variance > 50:  # Too concentrated
                positional_score -= 8
        
        # Penalize opponent's large concentrations
        opponent_stones = [board_after[i] for i in opponent_range if i not in game.burned_holes[1-current_player]]
        large_opponent_piles = sum(1 for stones in opponent_stones if stones >= 12)
        positional_score -= large_opponent_piles * 4
        
        return positional_score

    def evaluate_endgame_strategy(self, game, board_after):
        """Enhanced endgame evaluation with better strategy"""
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
        head_diff = my_head - opponent_head
        
        if total_remaining <= 25:  # Endgame threshold
            # If we're significantly ahead, prioritize safe play
            if head_diff >= 10:
                safety_bonus = max(0, (30 - my_stones)) * 2  # Reward clearing our side
                return head_diff * 20 + safety_bonus
            
            # If we're behind, prioritize aggressive play
            elif head_diff <= -5:
                if my_stones > opponent_stones + 3:
                    # We have material advantage - press it
                    comeback_bonus = (my_stones - opponent_stones) * 8
                    return head_diff * 15 + comeback_bonus
                else:
                    # Desperate situation - need maximum aggression
                    return head_diff * 25
            
            # Close game - prioritize material and tempo
            else:
                material_bonus = (my_stones - opponent_stones) * 6
                return head_diff * 18 + material_bonus
        
        return 0

    def get_strategic_variation(self, board_after, hole, game_progress, move_number=0):
        """Reduced randomness for more consistent strong play"""
        if game_progress <= 0.05:  # Only very early game gets variation
            return 0.0
        
        # Much smaller random variation
        base_random = random.random() * 1 - 0.5  # -0.5 to +0.5 range
        
        # Small positional preference
        current_player = self.original_game.current_player
        if current_player == 0:
            relative_hole_pos = hole  # 0-6
        else:
            relative_hole_pos = hole - 8  # Normalize to 0-6 range
        
        # Slight preference for middle holes
        positional_bias = (3 - abs(relative_hole_pos - 3)) * 0.05
        
        total_variation = base_random + positional_bias
        
        # Much tighter clamp
        return max(-0.8, min(0.8, total_variation))

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
        
        # 1. ENHANCED Immediate tactical gains (higher weights)
        capture_score = result['total_captured'] * 18  # Increased from 12
        extra_turn_score = result['extra_turns'] * 25  # Increased from 20
        burn_penalty = result['burns_created'] * -40   # Increased penalty from -30
        
        scores['Captures'] = capture_score
        scores['Extra Turns'] = extra_turn_score
        scores['Burn Penalty'] = burn_penalty
        
        # 2. Enhanced strategic position evaluation
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
        
        # Phase-specific scoring with better balance
        if game_progress < 0.25:  # Early game - focus on development and captures
            scores['Head Advantage'] = head_diff * 12  # Increased from 8
            scores['Material Control'] = material_diff * 4  # Increased from 3
            
            # NEW: Reward capture potential
            capture_potential = self.evaluate_capture_potential(game, board_after)
            scores['Capture Potential'] = capture_potential * 0.8
            
            # Better development scoring
            active_holes = sum(1 for i in my_range if board_after[i] > 0 and i not in game.burned_holes[current_player])
            if active_holes >= 6:
                scores['Development'] = 12  # Increased reward
            elif active_holes >= 4:
                scores['Development'] = 6
            elif active_holes <= 2:
                scores['Development'] = -20  # Increased penalty
            else:
                scores['Development'] = 0
                
        elif game_progress > 0.7:  # Late game - endgame strategy
            endgame_score = self.evaluate_endgame_strategy(game, board_after)
            scores['Endgame Strategy'] = endgame_score
            scores['Material Control'] = material_diff * 2  # Increased from 1
            
        else:  # Mid game - tactical focus with better evaluation
            scores['Head Advantage'] = head_diff * 15  # Increased from 10
            scores['Material Control'] = material_diff * 3  # Increased from 2
            
            # Enhanced positional evaluation
            positional_strength = self.evaluate_positional_strength(game, board_after)
            scores['Positional Strength'] = positional_strength
            
            # Better flexibility scoring
            active_holes = sum(1 for i in my_range if board_after[i] > 0 and i not in game.burned_holes[current_player])
            if active_holes >= 4:
                scores['Flexibility'] = 8  # Increased from 5
            elif active_holes >= 2:
                scores['Flexibility'] = 2
            else:
                scores['Flexibility'] = -12  # Increased penalty from -8
        
        # 3. Enhanced threat analysis
        threat_score = self.analyze_opponent_threats(game, board_after)
        scores['Threat Analysis'] = threat_score
        
        # 4. GREATLY Enhanced move efficiency
        stones_used = game.board[hole]
        efficiency_score = 0
        
        # Better efficiency calculations
        if result['total_captured'] > 0:
            efficiency = result['total_captured'] / max(1, stones_used)
            if efficiency >= 1.0:  # Very efficient capture
                efficiency_score += 15
            elif efficiency >= 0.5:  # Good efficiency
                efficiency_score += 10
            else:
                efficiency_score += 6
        
        if result['extra_turns'] > 0:
            if stones_used <= 4:
                efficiency_score += 12  # Very efficient extra turn
            elif stones_used <= 8:
                efficiency_score += 8   # Good efficiency
            else:
                efficiency_score += 4   # Less efficient but still valuable
        
        # Enhanced penalty for wasteful moves
        if stones_used > 10 and result['total_captured'] == 0 and result['extra_turns'] == 0:
            efficiency_score -= 15  # Increased penalty from -8
        
        # NEW: Reward moves that don't help opponent
        if result['total_captured'] == 0 and result['extra_turns'] == 0:
            # Check if this move gives opponent easy captures
            opponent_easy_captures = 0
            opponent_range = range(8, 15) if current_player == 0 else range(0, 7)
            for opp_hole in opponent_range:
                if board_after[opp_hole] > 0:
                    opposite = 14 - opp_hole
                    if board_after[opposite] == 1:  # We just set up an easy capture
                        opponent_easy_captures += 1
            
            if opponent_easy_captures > 0:
                efficiency_score -= opponent_easy_captures * 8
        
        scores['Move Efficiency'] = efficiency_score
        
        # 5. Enhanced tactical considerations
        tactical_score = 0
        
        # Better evaluation of follow-up opportunities
        immediate_captures = 0
        total_follow_up_value = 0
        
        for next_hole in my_range:
            if (board_after[next_hole] == 0 or 
                next_hole in game.burned_holes[current_player]):
                continue
            
            next_stones = board_after[next_hole]
            # Quick simulation for follow-up potential
            if 1 <= next_stones <= 12:  # Reasonable move range
                # Award points for having good follow-up moves available
                tactical_score += 1
                
                # Estimate if this could lead to capture
                estimated_landing = (next_hole + next_stones) % 16
                if ((current_player == 0 and 0 <= estimated_landing <= 6) or
                    (current_player == 1 and 8 <= estimated_landing <= 14)):
                    if estimated_landing != 7 and estimated_landing != 15:
                        opposite = 14 - estimated_landing
                        if board_after[opposite] > 0:
                            tactical_score += board_after[opposite] * 0.3
        
        # NEW: Reward creating multiple threats
        empty_holes_with_opposite = 0
        for my_hole in my_range:
            if board_after[my_hole] == 0 and my_hole not in game.burned_holes[current_player]:
                opposite = 14 - my_hole
                if board_after[opposite] > 0:
                    empty_holes_with_opposite += 1
        
        if empty_holes_with_opposite >= 3:
            tactical_score += 8  # Multiple capture threats
        elif empty_holes_with_opposite >= 2:
            tactical_score += 4
        
        scores['Tactical Setup'] = tactical_score
        
        # 6. Reduced strategic variation for more consistent play
        variation = self.get_strategic_variation(board_after, hole, game_progress)
        scores['Strategic Variation'] = variation
        
        # 7. NEW: Combo and chain bonus
        combo_bonus = 0
        if result['total_captured'] > 0 and result['extra_turns'] > 0:
            combo_bonus = 10  # Reward capture + extra turn combo
        
        if result['total_captured'] >= 8:  # Big capture
            combo_bonus += 8
        
        scores['Combo Bonus'] = combo_bonus
        
        # Total score calculation
        total_score = sum(scores.values())
        
        # Add context info
        scores['Stones Used'] = stones_used
        scores['Total Score'] = total_score
        
        return total_score, scores