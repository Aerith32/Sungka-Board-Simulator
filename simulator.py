# enhanced_simulator.py
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

class MaxPolicyBot:
    """Always chooses the house with the most stones"""
    def __init__(self, player_index):
        self.player_index = player_index
    
    def get_move(self, game):
        valid_moves = game.get_valid_moves(self.player_index)
        if not valid_moves:
            return None
        
        # Find the move with the most stones
        return max(valid_moves, key=lambda move: game.board[move])

class ExactPolicyBot:
    """Chooses house where stones equal distance to head for extra turn"""
    def __init__(self, player_index):
        self.player_index = player_index
    
    def get_move(self, game):
        valid_moves = game.get_valid_moves(self.player_index)
        if not valid_moves:
            return None
        
        head_position = 7 if self.player_index == 0 else 15
        exact_moves = []
        
        for move in valid_moves:
            stones = game.board[move]
            # Calculate distance to head, accounting for skipped holes
            distance_to_head = self._calculate_distance_to_head(game, move, head_position)
            
            if stones == distance_to_head:
                exact_moves.append(move)
        
        if exact_moves:
            # Pick the one nearest to head (highest index for P1, lowest for P2)
            if self.player_index == 0:
                return max(exact_moves)  # Nearest to head for P1
            else:
                return min(exact_moves)  # Nearest to head for P2
        
        # Fallback to Max Policy
        return max(valid_moves, key=lambda move: game.board[move])
    
    def _calculate_distance_to_head(self, game, start_hole, head_position):
        """Calculate actual distance considering skipped holes"""
        current_hole = start_hole
        distance = 0
        
        while True:
            current_hole = (current_hole + 1) % 16
            
            # Skip opponent's head
            if (self.player_index == 0 and current_hole == 15) or \
               (self.player_index == 1 and current_hole == 7):
                continue
                
            # Skip burned holes
            if (current_hole in game.burned_holes[0] or 
                current_hole in game.burned_holes[1]):
                continue
            
            distance += 1
            
            if current_hole == head_position:
                break
                
            # Prevent infinite loop
            if distance > 20:
                break
        
        return distance

class RealisticBasicRuleBot:
    """Bot that uses actual basic Sungka strategy"""
    def __init__(self, player_index):
        self.player_index = player_index
    
    def get_move(self, game):
        valid_moves = game.get_valid_moves(self.player_index)
        if not valid_moves:
            return None
        
        # Priority 1: Look for immediate captures
        for move in valid_moves:
            if self.can_capture(game, move):
                return move
        
        # Priority 2: Look for extra turns (land in own head)
        extra_turn_moves = []
        for move in valid_moves:
            if self.gives_extra_turn(game, move):
                extra_turn_moves.append(move)
        
        if extra_turn_moves:
            # Pick the extra turn move with fewest stones (most efficient)
            return min(extra_turn_moves, key=lambda x: game.board[x])
        
        # Priority 3: Pick a reasonable move (moderate stones, middle holes preferred)
        scored_moves = []
        for move in valid_moves:
            score = 0
            stones = game.board[move]
            
            # Prefer moderate stone counts (4-8 stones)
            if 4 <= stones <= 8:
                score += 10
            elif stones <= 2:
                score -= 5
            elif stones >= 12:
                score -= 8
            
            # Prefer middle holes
            if self.player_index == 0:  # Player 1
                if move in [2, 3, 4]:
                    score += 5
                elif move in [0, 6]:
                    score -= 3
            else:  # Player 2
                if move in [10, 11, 12]:
                    score += 5
                elif move in [8, 14]:
                    score -= 3
            
            scored_moves.append((move, score))
        
        return max(scored_moves, key=lambda x: x[1])[0]
    
    def can_capture(self, game, hole):
        """Check if this move leads to a capture"""
        stones = game.board[hole]
        current_hole = hole
        
        # Simple simulation
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
        
        # Check if we land in our own empty hole with stones in opposite
        if ((self.player_index == 0 and 0 <= current_hole <= 6) or
            (self.player_index == 1 and 8 <= current_hole <= 14)):
            if game.board[current_hole] == 0:
                opposite = 14 - current_hole
                return game.board[opposite] > 0
        
        return False
    
    def gives_extra_turn(self, game, hole):
        """Check if this move gives an extra turn"""
        stones = game.board[hole]
        current_hole = hole
        
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
        
        # Check if we land in our own head
        return ((self.player_index == 0 and current_hole == 7) or
                (self.player_index == 1 and current_hole == 15))

class BasicRuleBot:
    def __init__(self, player_index):
        self.player_index = player_index
        self.realistic_bot = RealisticBasicRuleBot(player_index)
    
    def get_move(self, game):
        return self.realistic_bot.get_move(game)

class HeuristicBot:
    def __init__(self, player_index):
        self.player_index = player_index
    
    def get_move(self, game):
        heuristic = SungkaHeuristic(game)
        valid_moves = game.get_valid_moves(self.player_index)
        if not valid_moves:
            return None
        
        # Get scored moves using the heuristic
        scored = []
        for move in valid_moves:
            try:
                score, _ = heuristic.evaluate_move_verbose(move)
                scored.append((move, score))
            except Exception as e:
                print(f"Heuristic evaluation failed for move {move}: {e}")
                scored.append((move, -1000))
        
        if not scored:
            return valid_moves[0] if valid_moves else None
            
        return max(scored, key=lambda x: x[1])[0]

class Simulator:
    def __init__(self, opponent_type, num_simulations=100, max_moves_per_game=200, random_seed=None, save_excel=True):
        self.opponent_type = opponent_type
        self.num_simulations = num_simulations
        self.max_moves_per_game = max_moves_per_game
        self.save_excel = save_excel
        if random_seed is not None:
            random.seed(random_seed)
        self.per_game_rows = []

    def get_heuristic_move(self, game, heuristic):
        valid_moves = game.get_valid_moves(game.current_player)
        if not valid_moves:
            return None
        
        scored = []
        for move in valid_moves:
            try:
                score, _ = heuristic.evaluate_move_verbose(move)
                scored.append((move, score))
            except Exception as e:
                print(f"Heuristic evaluation failed for move {move}: {e}")
                scored.append((move, -1000))
        
        if not scored:
            return valid_moves[0] if valid_moves else None
            
        return max(scored, key=lambda x: x[1])[0]

    def get_opponent_bot(self, player_index):
        if self.opponent_type == 1:
            return RandomBot(player_index)
        elif self.opponent_type == 2:
            return BasicRuleBot(player_index)
        elif self.opponent_type == 3:
            return HeuristicBot(player_index)
        elif self.opponent_type == 4:
            return MaxPolicyBot(player_index)
        elif self.opponent_type == 5:
            return ExactPolicyBot(player_index)

    def simulate_single_game(self, game_number, heuristic_goes_first=True, logger=None):
        game = SungkaGame()

        # Set starting player based on turn order scenario
        if heuristic_goes_first:
            game.current_player = 0  # Heuristic is Player 1 (goes first)
            heuristic_player = 0
            opponent_player = 1
        else:
            game.current_player = 1  # Opponent is Player 1 (goes first), Heuristic is Player 2
            heuristic_player = 1
            opponent_player = 0

        heuristic = SungkaHeuristic(game)
        opponent = self.get_opponent_bot(opponent_player)
        move_count = 0

        while not game.is_game_over() and move_count < self.max_moves_per_game:
            current_player = game.current_player
            
            # Check if current player has valid moves
            valid_moves = game.get_valid_moves(current_player)
            if not valid_moves:
                game.collect_remaining_stones()
                break

            if current_player == heuristic_player:
                # Heuristic player
                move = self.get_heuristic_move(game, heuristic)
            else:
                # Opponent bot
                move = opponent.get_move(game)

            if move is None:
                game.collect_remaining_stones()
                break

            try:
                result = game.play_turn(move)
                move_count += 1
                
                if result == "Game Over":
                    break
                    
            except ValueError as e:
                print(f"Invalid move attempted: {e}")
                game.collect_remaining_stones()
                break

        winner = game.get_winner()
        
        # Determine if heuristic won
        heuristic_won = None
        if winner is not None:
            heuristic_won = (winner == heuristic_player)

        # Record game results
        row = {
            'game_number': game_number,
            'heuristic_goes_first': heuristic_goes_first,
            'heuristic_player_index': heuristic_player,
            'winner': winner,
            'heuristic_won': heuristic_won,
            'final_p1_head': game.board[7],
            'final_p2_head': game.board[15],
            'moves_played': game.metrics['moves'],
            'marbles_captured_by_heuristic': game.metrics['marbles_captured'],
            'extra_turns_by_heuristic': game.metrics['extra_turns'],
            'burned_created_by_heuristic': game.metrics['burned_created'],
            'burned_suffered_by_heuristic': game.metrics['burned_suffered'],
            'burned_holes_p0': ','.join(map(str, sorted(list(game.burned_holes[0])))) if game.burned_holes[0] else '',
            'burned_holes_p1': ','.join(map(str, sorted(list(game.burned_holes[1])))) if game.burned_holes[1] else ''
        }

        self.per_game_rows.append(row)
        return row

    def run_turn_order_analysis(self):
        """Run simulations testing both turn orders"""
        start = time.time()
        
        # Split simulations between first/second player scenarios
        first_player_games = self.num_simulations // 2
        second_player_games = self.num_simulations - first_player_games
        
        print(f"Running {first_player_games} games as first player...")
        for i in range(1, first_player_games + 1):
            if i % 10 == 0:
                print(f"  First player games: {i}/{first_player_games}")
            self.simulate_single_game(i, heuristic_goes_first=True)
        
        print(f"Running {second_player_games} games as second player...")
        for i in range(first_player_games + 1, self.num_simulations + 1):
            if (i - first_player_games) % 10 == 0:
                print(f"  Second player games: {i - first_player_games}/{second_player_games}")
            self.simulate_single_game(i, heuristic_goes_first=False)

        elapsed = time.time() - start
        df = pd.DataFrame(self.per_game_rows)
        
        self.analyze_results(df, elapsed)
        return df

    def run_standard(self):
        """Run standard simulation with random turn order"""
        start = time.time()

        for i in range(1, self.num_simulations + 1):
            if i % 10 == 0:
                print(f"Completed {i}/{self.num_simulations} simulations...")
            # Randomly choose who goes first
            heuristic_first = random.choice([True, False])
            self.simulate_single_game(i, heuristic_goes_first=heuristic_first)

        elapsed = time.time() - start
        df = pd.DataFrame(self.per_game_rows)
        
        self.analyze_results(df, elapsed)
        return df

    def analyze_results(self, df, elapsed):
        """Analyze and print simulation results"""
        total = len(df)
        
        if total == 0:
            print("No games completed successfully.")
            return

        # Overall statistics
        heuristic_wins = df['heuristic_won'].eq(True).sum()
        opponent_wins = df['heuristic_won'].eq(False).sum()
        draws = df['heuristic_won'].isna().sum()

        # Turn order analysis
        first_player_df = df[df['heuristic_goes_first'] == True]
        second_player_df = df[df['heuristic_goes_first'] == False]
        
        first_wins = first_player_df['heuristic_won'].eq(True).sum() if len(first_player_df) > 0 else 0
        first_total = len(first_player_df)
        second_wins = second_player_df['heuristic_won'].eq(True).sum() if len(second_player_df) > 0 else 0
        second_total = len(second_player_df)

        # Performance metrics
        total_moves = df['moves_played'].sum()
        if total_moves > 0:
            avg_capture = df['marbles_captured_by_heuristic'].sum() / total_moves
            avg_extra = df['extra_turns_by_heuristic'].sum() / total_moves
            avg_burn_created = df['burned_created_by_heuristic'].sum() / total_moves
            avg_burn_suffered = df['burned_suffered_by_heuristic'].sum() / total_moves
        else:
            avg_capture = avg_extra = avg_burn_created = avg_burn_suffered = 0

        # Count games with burned holes
        games_with_burns = len(df[(df['burned_holes_p0'] != '') | 
                                 (df['burned_holes_p1'] != '')])

        # Print results
        opponent_names = {
            1: 'Random',
            2: 'Realistic Basic Rules', 
            3: 'Heuristic vs Heuristic',
            4: 'Max Policy',
            5: 'Exact Policy'
        }
        
        print("\n" + "="*60)
        print("HEURISTIC PERFORMANCE ANALYSIS")
        print("="*60)
        print(f"Opponent Type: {self.opponent_type} ({opponent_names.get(self.opponent_type, 'Unknown')})")
        print(f"Total Games: {total}")
        print(f"Total Time: {elapsed:.2f} seconds")
        print(f"Average Game Length: {total_moves/total:.1f} moves" if total > 0 else "N/A")
        
        print("\n--- OVERALL RESULTS ---")
        print(f"Heuristic Wins: {heuristic_wins}/{total} ({heuristic_wins/total*100:.1f}%)")
        print(f"Opponent Wins: {opponent_wins}/{total} ({opponent_wins/total*100:.1f}%)")
        print(f"Draws: {draws}/{total} ({draws/total*100:.1f}%)")
        
        print("\n--- TURN ORDER ANALYSIS ---")
        if first_total > 0:
            print(f"As First Player:  {first_wins}/{first_total} ({first_wins/first_total*100:.1f}% wins)")
        if second_total > 0:
            print(f"As Second Player: {second_wins}/{second_total} ({second_wins/second_total*100:.1f}% wins)")
        
        if first_total > 0 and second_total > 0:
            first_rate = first_wins/first_total
            second_rate = second_wins/second_total
            advantage = first_rate - second_rate
            print(f"First Player Advantage: {advantage*100:+.1f} percentage points")
        
        print("\n--- PERFORMANCE METRICS (Heuristic) ---")
        print(f"Avg Marbles Captured per Move: {avg_capture:.4f}")
        print(f"Avg Extra Turns per Move: {avg_extra:.4f}")
        print(f"Avg Burned Holes Created per Move: {avg_burn_created:.6f}")
        print(f"Avg Burned Holes Suffered per Move: {avg_burn_suffered:.6f}")
        print(f"Games with Burned Holes: {games_with_burns}/{total} ({games_with_burns/total*100:.1f}%)")
        
        print("="*60)

        if self.save_excel:
            timestamp = int(time.time())
            outname = f"simulation_results_{opponent_names.get(self.opponent_type, 'unknown').lower().replace(' ', '_')}_{timestamp}.xlsx"
            df.to_excel(outname, index=False)
            print(f"Saved detailed results to: {outname}")

if __name__ == "__main__":
    print("🎮 ENHANCED SUNGKA SIMULATION")
    print("Choose opponent:")
    print("1 = Random Bot")
    print("2 = Basic Rules Bot") 
    print("3 = Heuristic vs Heuristic")
    print("4 = Max Policy Bot (always picks house with most stones)")
    print("5 = Exact Policy Bot (picks house where stones = distance to head)")
    
    while True:
        try:
            choice = int(input("Enter choice (1-5): "))
            if choice in [1, 2, 3, 4, 5]:
                break
            print("Please enter 1, 2, 3, 4, or 5.")
        except ValueError:
            print("Please enter a number.")
    
    print("\nChoose simulation type:")
    print("1 = Standard (random turn order)")
    print("2 = Turn Order Analysis (test first/second player scenarios)")
    
    while True:
        try:
            sim_type = int(input("Enter choice (1-2): "))
            if sim_type in [1, 2]:
                break
            print("Please enter 1 or 2.")
        except ValueError:
            print("Please enter a number.")
    
    # Ask about number of simulations
    while True:
        try:
            num_sims = int(input("Number of simulations (default 100): ") or "100")
            if num_sims > 0:
                break
            print("Please enter a positive number.")
        except ValueError:
            print("Please enter a number.")
    
    # Create simulator without fixed seed for varied results
    sim = Simulator(opponent_type=choice, num_simulations=num_sims)
    
    if sim_type == 1:
        print(f"\n🚀 Running {num_sims} games with random turn order...")
        sim.run_standard()
    else:
        print(f"\n🚀 Running turn order analysis with {num_sims} games...")
        sim.run_turn_order_analysis()
    
    print("\n🎉 Simulation complete!")