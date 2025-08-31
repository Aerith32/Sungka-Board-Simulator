# improved_simulator.py
from main import SungkaGame
from heuristic import SungkaHeuristic
from game_logger import GameLogger
import time
import random
import pandas as pd

class RandomBot:
    def __init__(self, player_index):
        self.player_index = player_index
    
    def get_move(self, game):
        valid_moves = game.get_valid_moves(self.player_index)
        return random.choice(valid_moves) if valid_moves else None

class MaxBot:
    """Always picks the hole with the most stones"""
    def __init__(self, player_index):
        self.player_index = player_index
    
    def get_move(self, game):
        valid_moves = game.get_valid_moves(self.player_index)
        if not valid_moves:
            return None
        
        # Pick the hole with maximum stones
        return max(valid_moves, key=lambda x: game.board[x])

class ExactBot:
    """Uses exact calculation for best immediate outcome"""
    def __init__(self, player_index):
        self.player_index = player_index
    
    def get_move(self, game):
        valid_moves = game.get_valid_moves(self.player_index)
        if not valid_moves:
            return None
        
        best_move = None
        best_immediate_gain = -float('inf')
        
        for move in valid_moves:
            # Calculate immediate gain (captured stones + extra turn value)
            immediate_gain = self.calculate_immediate_gain(game, move)
            if immediate_gain > best_immediate_gain:
                best_immediate_gain = immediate_gain
                best_move = move
        
        return best_move if best_move is not None else valid_moves[0]
    
    def calculate_immediate_gain(self, game, hole):
        """Calculate immediate gain from a move"""
        # Simple simulation to get immediate results
        stones = game.board[hole]
        current_hole = hole
        
        # Simulate distribution
        while stones > 0:
            current_hole = (current_hole + 1) % 16
            
            # Skip opponent's head
            if (self.player_index == 0 and current_hole == 15) or \
               (self.player_index == 1 and current_hole == 7):
                continue
                
            # Skip burned holes
            if (current_hole in game.burned_holes[0] or 
                current_hole in game.burned_holes[1]):
                continue
                
            stones -= 1
        
        gain = 0
        
        # Check for extra turn
        if (self.player_index == 0 and current_hole == 7) or \
           (self.player_index == 1 and current_hole == 15):
            gain += 10  # Value extra turn as 10 points
        
        # Check for potential capture
        elif ((self.player_index == 0 and 0 <= current_hole <= 6) or
              (self.player_index == 1 and 8 <= current_hole <= 14)):
            if game.board[current_hole] == 0:  # Would land in empty hole
                opposite = 14 - current_hole
                if game.board[opposite] > 0:
                    gain += game.board[opposite] + 1  # Capture value
        
        return gain

class RewardBasedHeuristicBot:
    """Heuristic bot that uses reward = my_score - opponent_score"""
    def __init__(self, player_index):
        self.player_index = player_index
    
    def get_move(self, game):
        valid_moves = game.get_valid_moves(self.player_index)
        if not valid_moves:
            return None
        
        # Create a temporary heuristic for this game state
        heuristic = SungkaHeuristic(game)
        
        best_move = None
        best_reward = -float('inf')
        
        for move in valid_moves:
            try:
                # Calculate reward for this move
                reward = self.calculate_move_reward(game, heuristic, move)
                if reward > best_reward:
                    best_reward = reward
                    best_move = move
            except Exception as e:
                print(f"Error evaluating move {move}: {e}")
                continue
        
        return best_move if best_move is not None else valid_moves[0]
    
    def calculate_move_reward(self, game, heuristic, move):
        """Calculate reward = my_score - opponent_score for this move"""
        # Get current scores
        if self.player_index == 0:
            my_current_score = game.board[7]
            opponent_current_score = game.board[15]
        else:
            my_current_score = game.board[15]
            opponent_current_score = game.board[7]
        
        # Simulate the move
        result = heuristic.simulate_move_complete(game, move)
        if result is None:
            return -float('inf')
        
        board_after = result['board']
        
        # Get scores after move
        if self.player_index == 0:
            my_new_score = board_after[7]
            opponent_new_score = board_after[15]
        else:
            my_new_score = board_after[15]
            opponent_new_score = board_after[7]
        
        # Calculate reward as score difference change
        current_diff = my_current_score - opponent_current_score
        new_diff = my_new_score - opponent_new_score
        immediate_reward = new_diff - current_diff
        
        # Add some strategic considerations
        strategic_bonus = 0
        
        # Bonus for extra turns
        if result['extra_turns'] > 0:
            strategic_bonus += 8
        
        # Penalty for burns
        if result['burns_created'] > 0:
            strategic_bonus -= 15
        
        # Small positional bonus
        if self.player_index == 0:
            my_stones = sum(board_after[0:7])
            opponent_stones = sum(board_after[8:15])
        else:
            my_stones = sum(board_after[8:15])
            opponent_stones = sum(board_after[0:7])
        
        material_bonus = (my_stones - opponent_stones) * 0.2
        
        return immediate_reward + strategic_bonus + material_bonus

class ImprovedSimulator:
    def __init__(self, opponent_type, num_simulations=500, max_moves_per_game=200, random_seed=None, save_excel=True):
        self.opponent_type = opponent_type
        self.num_simulations = num_simulations
        self.max_moves_per_game = max_moves_per_game
        self.save_excel = save_excel
        if random_seed is not None:
            random.seed(random_seed)
        self.per_game_rows = []

    def get_opponent_bot(self, player_index):
        if self.opponent_type == 1:
            return RandomBot(player_index)
        elif self.opponent_type == 2:
            return MaxBot(player_index)
        elif self.opponent_type == 3:
            return ExactBot(player_index)
        elif self.opponent_type == 4:
            return RewardBasedHeuristicBot(player_index)

    def simulate_single_game(self, game_number, heuristic_is_first_player=True):
        """Simulate one game with specified turn order"""
        game = SungkaGame()
        
        # Set starting player based on test scenario
        if heuristic_is_first_player:
            game.current_player = 0  # Heuristic starts
            heuristic_player = 0
            opponent_player = 1
        else:
            game.current_player = 1  # Opponent starts  
            heuristic_player = 1
            opponent_player = 0
        
        # Create bots
        heuristic_bot = RewardBasedHeuristicBot(heuristic_player)
        opponent_bot = self.get_opponent_bot(opponent_player)
        
        move_count = 0
        heuristic_rewards = []
        
        while not game.is_game_over() and move_count < self.max_moves_per_game:
            current_player = game.current_player
            
            # Store pre-move state for reward calculation
            if heuristic_player == 0:
                pre_my_score = game.board[7]
                pre_opp_score = game.board[15]
            else:
                pre_my_score = game.board[15]
                pre_opp_score = game.board[7]
            
            # Get move from appropriate bot
            if current_player == heuristic_player:
                move = heuristic_bot.get_move(game)
            else:
                move = opponent_bot.get_move(game)

            if move is None:
                break

            try:
                result = game.play_turn(move)
                move_count += 1
                
                # Calculate reward for heuristic player moves
                if current_player == heuristic_player:
                    if heuristic_player == 0:
                        post_my_score = game.board[7]
                        post_opp_score = game.board[15]
                    else:
                        post_my_score = game.board[15]
                        post_opp_score = game.board[7]
                    
                    # Reward = change in (my_score - opponent_score)
                    pre_diff = pre_my_score - pre_opp_score
                    post_diff = post_my_score - post_opp_score
                    reward = post_diff - pre_diff
                    heuristic_rewards.append(reward)
                
                if result == "Game Over":
                    break
                    
            except ValueError as e:
                print(f"Invalid move attempted: {e}")
                break

        # Determine winner - FIXED LOGIC
        final_p1_score = game.board[7]
        final_p2_score = game.board[15]
        
        if final_p1_score > final_p2_score:
            winner = 0  # Player 1 wins
        elif final_p2_score > final_p1_score:
            winner = 1  # Player 2 wins  
        else:
            winner = None  # Draw
        
        # Did the heuristic win? - FIXED
        heuristic_won = (winner == heuristic_player)
        
        # Calculate final scores and metrics
        if heuristic_player == 0:
            heuristic_final_score = final_p1_score
            opponent_final_score = final_p2_score
            heuristic_metrics = game.metrics
        else:
            heuristic_final_score = final_p2_score
            opponent_final_score = final_p1_score
            # For player 2, we don't track metrics in the game object
            heuristic_metrics = {"marbles_captured": 0, "extra_turns": 0, 
                               "burned_created": 0, "burned_suffered": 0, "moves": 0}

        # Record results
        row = {
            'game_number': game_number,
            'heuristic_is_first': heuristic_is_first_player,
            'heuristic_player': heuristic_player,
            'winner': winner,
            'heuristic_won': heuristic_won,
            'heuristic_final_score': heuristic_final_score,
            'opponent_final_score': opponent_final_score,
            'score_difference': heuristic_final_score - opponent_final_score,
            'total_moves': move_count,
            'avg_reward_per_move': sum(heuristic_rewards) / len(heuristic_rewards) if heuristic_rewards else 0,
            'total_reward': sum(heuristic_rewards),
            'heuristic_metrics': heuristic_metrics
        }

        self.per_game_rows.append(row)
        return row

    def run_comprehensive_test(self):
        """Run tests with both turn orders like the research paper"""
        start = time.time()
        
        # Test both scenarios
        scenarios = [
            (True, "Heuristic First"),
            (False, "Heuristic Second")
        ]
        
        for heuristic_first, scenario_name in scenarios:
            print(f"\n🧪 Testing: {scenario_name}")
            
            for i in range(1, self.num_simulations // 2 + 1):
                if i % 50 == 0:
                    print(f"  {scenario_name}: {i}/{self.num_simulations//2} games...")
                self.simulate_single_game(i, heuristic_is_first_player=heuristic_first)

        elapsed = time.time() - start
        df = pd.DataFrame(self.per_game_rows)
        
        # Analyze results by turn order
        print("\n" + "="*60)
        print("COMPREHENSIVE TURN ORDER ANALYSIS")
        print("="*60)
        
        # Overall results
        total_games = len(df)
        overall_wins = df['heuristic_won'].sum()
        overall_win_rate = overall_wins / total_games * 100
        
        print(f"Overall Performance: {overall_wins}/{total_games} wins ({overall_win_rate:.1f}%)")
        
        # By turn order
        first_player_games = df[df['heuristic_is_first'] == True]
        second_player_games = df[df['heuristic_is_first'] == False]
        
        if len(first_player_games) > 0:
            first_wins = first_player_games['heuristic_won'].sum()
            first_win_rate = first_wins / len(first_player_games) * 100
            first_avg_score_diff = first_player_games['score_difference'].mean()
            first_avg_reward = first_player_games['avg_reward_per_move'].mean()
            
            print(f"\nAs FIRST Player:  {first_wins}/{len(first_player_games)} wins ({first_win_rate:.1f}%)")
            print(f"  Avg Score Difference: {first_avg_score_diff:+.2f}")
            print(f"  Avg Reward per Move: {first_avg_reward:+.3f}")
        
        if len(second_player_games) > 0:
            second_wins = second_player_games['heuristic_won'].sum()
            second_win_rate = second_wins / len(second_player_games) * 100
            second_avg_score_diff = second_player_games['score_difference'].mean()
            second_avg_reward = second_player_games['avg_reward_per_move'].mean()
            
            print(f"As SECOND Player: {second_wins}/{len(second_player_games)} wins ({second_win_rate:.1f}%)")
            print(f"  Avg Score Difference: {second_avg_score_diff:+.2f}")
            print(f"  Avg Reward per Move: {second_avg_reward:+.3f}")
        
        # Calculate first-player advantage
        if len(first_player_games) > 0 and len(second_player_games) > 0:
            advantage = first_win_rate - second_win_rate
            print(f"\nFirst-Player Advantage: {advantage:+.1f}% points")
        
        print("="*60)
        
        # Opponent-specific analysis
        opponent_names = {1: "Random", 2: "Max", 3: "Exact", 4: "Reward-Heuristic"}
        opponent_name = opponent_names.get(self.opponent_type, "Unknown")
        
        print(f"\nDetailed Analysis vs {opponent_name}:")
        print(f"Total Games: {total_games}")
        print(f"Total Time: {elapsed:.2f} seconds")
        print(f"Games per second: {total_games/elapsed:.2f}")
        
        if self.save_excel:
            timestamp = int(time.time())
            outname = f"comprehensive_results_{opponent_name.lower()}_{timestamp}.xlsx"
            df.to_excel(outname, index=False)
            print(f"\nSaved detailed results to: {outname}")
        
        return df

    def run(self):
        """Legacy method for backward compatibility"""
        return self.run_comprehensive_test()

class RewardBasedHeuristicBot:
    """Fair heuristic bot that creates new instance each turn (like opponent)"""
    def __init__(self, player_index):
        self.player_index = player_index
    
    def get_move(self, game):
        # Create fresh heuristic instance each turn for fairness
        heuristic = SungkaHeuristic(game)
        valid_moves = game.get_valid_moves(self.player_index)
        if not valid_moves:
            return None
        
        best_move = None
        best_score = -float('inf')
        
        for move in valid_moves:
            try:
                # Use your existing heuristic but remove randomization for fair comparison
                score, details = heuristic.evaluate_move_verbose(move)
                
                # Remove randomization component for fair mirror matches
                if 'Variation' in details:
                    score -= details['Variation']
                
                if score > best_score:
                    best_score = score
                    best_move = move
                    
            except Exception as e:
                print(f"Heuristic evaluation failed for move {move}: {e}")
                continue
        
        return best_move if best_move is not None else valid_moves[0]

if __name__ == "__main__":
    print("🎮 COMPREHENSIVE SUNGKA SIMULATION")
    print("Choose opponent:")
    print("1 = Random Bot")
    print("2 = Max Bot (always picks hole with most stones)")
    print("3 = Exact Bot (calculates immediate best outcome)")
    print("4 = Reward-Based Heuristic (fair mirror match)")
    
    while True:
        try:
            choice = int(input("Enter choice (1-4): "))
            if choice in [1, 2, 3, 4]:
                break
            print("Please enter 1, 2, 3, or 4.")
        except ValueError:
            print("Please enter a number.")

    print(f"\nRunning 1000 games (500 as first player, 500 as second player)...")
    sim = ImprovedSimulator(opponent_type=choice, num_simulations=100, random_seed=42)
    results = sim.run()
    print("\n🎉 Comprehensive simulation complete!")
    
    # Quick summary
    if choice == 4:
        print("\n💡 Mirror Match Analysis:")
        print("In a fair mirror match, both players should win ~50% of games.")
        print("Any significant deviation suggests heuristic bias or first-player advantage.")