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
        scored = [(m, heuristic.evaluate_move_verbose(m)[0]) for m in valid_moves]
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
        scored = [(m, heuristic.evaluate_move_verbose(m)[0]) for m in valid_moves]
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
            game.board = [
                0, 0, 1, 0, 0, 0, 0,
                0,
                0, 0, 0, 0, 0, 0, 0,
                0
            ]
            game.current_player = 0
            game.burned_holes = {0: set(), 1: set()}
        else:
            game.current_player = random.choice([0, 1])

        heuristic = SungkaHeuristic(game)
        opponent = self.get_opponent_bot(1)
        move_count = 0

        while not game.is_game_over() and move_count < self.max_moves_per_game:
            current_player = game.current_player

            if current_player == 0:
                move = self.get_heuristic_move(game, heuristic)
            else:
                move = opponent.get_move(game)

            if move is None:
                break

            try:
                result = game.play_turn(move)
                move_count += 1
                if result == "Game Over":
                    break
            except ValueError:
                break

        winner = game.get_winner()

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

    def run(self, force_one_sunog_test=True):
        start = time.time()

        if force_one_sunog_test:
            print("Running forced Sunog test...")
            self.simulate_single_game(0, force_sunog=True)

        for i in range(1, self.num_simulations + 1):
            self.simulate_single_game(i)

        elapsed = time.time() - start
        df = pd.DataFrame(self.per_game_rows)

        total = len(df[df['game_number'] != 0])
        p1_wins = df[df['game_number'] != 0]['winner'].eq(0).sum()
        p2_wins = df[df['game_number'] != 0]['winner'].eq(1).sum()
        draws = df[df['game_number'] != 0]['winner'].isna().sum()

        total_moves = df[df['game_number'] != 0]['moves_played'].sum()
        avg_capture = df[df['game_number'] != 0]['marbles_captured_by_heuristic'].sum() / max(1, total_moves)
        avg_extra = df[df['game_number'] != 0]['extra_turns_by_heuristic'].sum() / max(1, total_moves)
        avg_burn_created = df[df['game_number'] != 0]['burned_created_by_heuristic'].sum() / max(1, total_moves)
        avg_burn_suffered = df[df['game_number'] != 0]['burned_suffered_by_heuristic'].sum() / max(1, total_moves)

        print("\nHEURISTIC VS SELECTED OPPONENT SUMMARY")
        print("==================================================")
        print(f"Opponent Type: {self.opponent_type} ({'Random' if self.opponent_type==1 else 'Basic Rules' if self.opponent_type==2 else 'Heuristic'})")
        print(f"Total Games (excluding forced test): {total}")
        print(f"Total Time: {elapsed:.2f} seconds")
        print(f"Player 1 (Heuristic) Wins: {p1_wins} ({p1_wins/max(1,total)*100:.1f}%)")
        print(f"Player 2 (Opponent) Wins: {p2_wins} ({p2_wins/max(1,total)*100:.1f}%)")
        print(f"Draws: {draws} ({draws/max(1,total)*100:.1f}%)")
        print("==================================================")
        print("AVERAGE PERFORMANCE METRICS (Heuristic)")
        print(f"  Avg Marbles Captured per Move: {avg_capture:.2f}")
        print(f"  Avg Extra Turns per Move: {avg_extra:.2f}")
        print(f"  Avg Burned Holes Created per Move: {avg_burn_created:.4f}")
        # print(f"  Avg Burned Holes Suffered per Move: {avg_burn_suffered:.4f}")
        print("==================================================")

        if self.save_excel:
            outname = f"simulation_results_{int(time.time())}.xlsx"
            df.to_excel(outname, index=False)
            print(f"Saved per-game results to: {outname}")

        return df

if __name__ == "__main__":
    print("Choose opponent type:")
    print("1. Random Bot")
    print("2. Basic Rules Bot")
    print("3. Heuristic Bot")
    while True:
        try:
            opponent_choice = int(input("Enter choice (1-3): "))
            if opponent_choice in [1, 2, 3]:
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    sim = Simulator(opponent_type=opponent_choice, num_simulations=100, random_seed=42, save_excel=True)
    sim.run(force_one_sunog_test=True)
