# simulate_with_prompt.py
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

class BasicRuleBot:
    def __init__(self, player_index):
        self.player_index = player_index
    
    def get_move(self, game):
        valid_moves = game.get_valid_moves(self.player_index)
        return valid_moves[0] if valid_moves else None

class HeuristicBot:
    def __init__(self, player_index):
        self.player_index = player_index
    
    def get_move(self, game):
        heuristic = SungkaHeuristic(game)
        valid_moves = game.get_valid_moves(self.player_index)
        if not valid_moves:
            return None
        
        # Get scored moves using the corrected heuristic
        scored = []
        for move in valid_moves:
            try:
                score, _ = heuristic.evaluate_move_verbose(move)
                scored.append((move, score))
            except Exception as e:
                # If heuristic evaluation fails, give it a low score
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
                # If heuristic evaluation fails, give it a low score
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

    def simulate_single_game(self, game_number, force_sunog=False, logger=None):
        game = SungkaGame()

        if force_sunog:
            # Set up a scenario that should trigger Sunog
            game.board = [
                0, 1, 0, 0, 0, 0, 0,     # Player 1: only hole 1 has 1 stone
                0,                       # Player 1 head (empty)
                0, 0, 0, 0, 0, 0, 0,     # Player 2: all holes empty  
                0                        # Player 2 head (empty)
            ]
            game.current_player = 0  # Player 1's turn
            game.burned_holes = {0: set(), 1: set()}  # No burned holes initially
            print("🧪 FORCED SUNOG TEST SCENARIO:")
            print("Player 1 will play hole 1 -> stone lands in hole 2 (empty)")
            print("Opposite hole 12 is also empty -> Should trigger Sunog")
        else:
            game.current_player = random.choice([0, 1])

        heuristic = SungkaHeuristic(game)
        opponent = self.get_opponent_bot(1)
        move_count = 0

        if force_sunog:
            print("Initial board:")
            game.print_board_state()

        while not game.is_game_over() and move_count < self.max_moves_per_game:
            current_player = game.current_player

            if current_player == 0:
                # Heuristic player
                move = self.get_heuristic_move(game, heuristic)
            else:
                # Opponent bot
                move = opponent.get_move(game)

            if move is None:
                break

            if force_sunog and move_count == 0:
                print(f"Playing move: {move}")

            try:
                result = game.play_turn(move)
                move_count += 1
                
                if force_sunog and move_count == 1:
                    print("Board after forced Sunog test:")
                    game.print_board_state()
                    print("Burned holes:", game.burned_holes)
                    force_sunog = False  # Only do this for the first move
                
                if result == "Game Over":
                    break
            except ValueError as e:
                print(f"Invalid move attempted: {e}")
                break

        winner = game.get_winner()

        # Record game results
        row = {
            'game_number': game_number,
            'winner': winner,
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

    def run(self):
        start = time.time()

        for i in range(1, self.num_simulations + 1):
            if i % 10 == 0:
                print(f"Completed {i}/{self.num_simulations} simulations...")
            self.simulate_single_game(i)

        elapsed = time.time() - start
        df = pd.DataFrame(self.per_game_rows)

        # Calculate statistics
        df = pd.DataFrame(self.per_game_rows)
        total = len(df)
        
        if total > 0:
            p1_wins = df['winner'].eq(0).sum()
            p2_wins = df['winner'].eq(1).sum()
            draws = df['winner'].isna().sum()

            total_moves = df['moves_played'].sum()
            avg_capture = df['marbles_captured_by_heuristic'].sum() / max(1, total_moves)
            avg_extra = df['extra_turns_by_heuristic'].sum() / max(1, total_moves)
            avg_burn_created = df['burned_created_by_heuristic'].sum() / max(1, total_moves)
            avg_burn_suffered = df['burned_suffered_by_heuristic'].sum() / max(1, total_moves)

            # Count games with burned holes
            games_with_burns = len(df[(df['burned_holes_p0'] != '') | 
                                     (df['burned_holes_p1'] != '')])
            
            print("\nHEURISTIC VS SELECTED OPPONENT SUMMARY")
            print("==================================================")
            print(f"Opponent Type: {self.opponent_type} ({'Random' if self.opponent_type==1 else 'Basic Rules' if self.opponent_type==2 else 'Heuristic'})")
            print(f"Total Games: {total}")
            print(f"Total Time: {elapsed:.2f} seconds")
            print(f"Player 1 (Heuristic) Wins: {p1_wins} ({p1_wins/total*100:.1f}%)")
            print(f"Player 2 (Opponent) Wins: {p2_wins} ({p2_wins/total*100:.1f}%)")
            print(f"Draws: {draws} ({draws/total*100:.1f}%)")
            print("==================================================")
            print("AVERAGE PERFORMANCE METRICS (Heuristic)")
            print(f"  Avg Marbles Captured per Move: {avg_capture:.4f}")
            print(f"  Avg Extra Turns per Move: {avg_extra:.4f}")
            print(f"  Avg Burned Holes Created per Move: {avg_burn_created:.6f}")
            print(f"  Avg Burned Holes Suffered per Move: {avg_burn_suffered:.6f}")
            print(f"  Games with Burned Holes: {games_with_burns}/{total} ({games_with_burns/total*100:.1f}%)")
            print("==================================================")

            if self.save_excel:
                outname = f"simulation_results_corrected_sunog_{int(time.time())}.xlsx"
                df.to_excel(outname, index=False)
                print(f"Saved per-game results to: {outname}")
        else:
            print("No games completed successfully.")

        return df

if __name__ == "__main__":
    print("🎮 SUNGKA SIMULATION")
    print("Choose opponent: 1=Random, 2=Basic Rules, 3=Heuristic")
    
    while True:
        try:
            choice = int(input("Enter choice (1-3): "))
            if choice in [1, 2, 3]:
                break
            print("Please enter 1, 2, or 3.")
        except ValueError:
            print("Please enter a number.")

    sim = Simulator(opponent_type=choice, num_simulations=100, random_seed=42)
    sim.run()
    print("\n🎉 Simulation complete!")