from heuristic import SungkaHeuristic
from game_logger import GameLogger

class SungkaGame:
    def __init__(self):
        print("✅ Load Complete")
        self.board = [7] * 7 + [0] + [7] * 7 + [0]
        self.current_player = 0
        self.burned_holes = {0: set(), 1: set()}
        # Track performance metrics
        self.metrics = {
            "marbles_captured": 0,
            "extra_turns": 0,
            "burned_created": 0,
            "burned_suffered": 0,
            "moves": 0
        }

    def is_valid_move(self, hole):
        if hole in self.burned_holes[self.current_player]:
            return False
        if self.current_player == 0:
            return 0 <= hole <= 6 and self.board[hole] > 0
        else:
            return 8 <= hole <= 14 and self.board[hole] > 0

    
    def distribute_stones(self, starting_hole, show_intermediate=True):
        current_hole = starting_hole
        stones = self.board[current_hole]
        self.board[current_hole] = 0
        extra_turn = False
        
        # Track which holes were originally empty for Sunog rule
        originally_empty_holes = set()
        for i in range(16):
            if self.board[i] == 0:
                originally_empty_holes.add(i)
        
        while stones > 0:
            current_hole = (current_hole + 1) % 16
            
            # Skip opponent's head
            if (self.current_player == 0 and current_hole == 15) or \
            (self.current_player == 1 and current_hole == 7):
                continue
                
            # Skip burned holes for ALL players
            if (current_hole in self.burned_holes[0] or 
                current_hole in self.burned_holes[1]):
                continue
                
            self.board[current_hole] += 1
            stones -= 1

        if show_intermediate:
            print("\n--- Intermediate Board State ---")
            print(f"Last stone landed in hole {current_hole}")
            self.print_board_state()
        
        # Check for extra turn (landed in own head)
        if (self.current_player == 0 and current_hole == 7) or \
        (self.current_player == 1 and current_hole == 15):
            extra_turn = True
            return current_hole, False, extra_turn, originally_empty_holes
        
        # Check for relay (continue distribution)
        if self.board[current_hole] > 1 and current_hole not in (7, 15):
            if show_intermediate:
                print(f"Continuing distribution from hole {current_hole}...")
            return self.distribute_stones(current_hole, show_intermediate)
        
        # Check for capture (landed in own empty hole with stones in opposite hole)
        should_capture = (
            (self.current_player == 0 and 0 <= current_hole <= 6 and 
            current_hole in originally_empty_holes and self.board[14 - current_hole] > 0) or
            (self.current_player == 1 and 8 <= current_hole <= 14 and 
            current_hole in originally_empty_holes and self.board[14 - current_hole] > 0)
        )
        
        return current_hole, should_capture, False, originally_empty_holes

    def check_capture(self, last_hole):
        if self.current_player == 0 and 0 <= last_hole <= 6:
            opposite_hole = 14 - last_hole
            if self.board[opposite_hole] > 0:
                captured_stones = self.board[last_hole] + self.board[opposite_hole]
                self.board[7] += captured_stones
                self.board[last_hole] = 0
                self.board[opposite_hole] = 0
                # 🔥 Burn after capture (Sunog)
                self.burned_holes[0].add(last_hole)
                print(f"Player 1 captured {captured_stones} stones from holes {last_hole} and {opposite_hole}")
                print(f"🔥 SUNOG! Player 1's hole {last_hole} is now burned.")

        elif self.current_player == 1 and 8 <= last_hole <= 14:
            opposite_hole = 14 - last_hole
            if self.board[opposite_hole] > 0:
                captured_stones = self.board[last_hole] + self.board[opposite_hole]
                self.board[15] += captured_stones
                self.board[last_hole] = 0
                self.board[opposite_hole] = 0
                # 🔥 Burn after capture (Sunog)
                self.burned_holes[1].add(last_hole)
                print(f"Player 2 captured {captured_stones} stones from holes {last_hole} and {opposite_hole}")
                print(f"🔥 SUNOG! Player 2's hole {last_hole} is now burned.")


    def apply_sunog_rule(self, last_hole, originally_empty_holes):
        """
        Correct Sunog rule: If you land in an originally empty hole on your own side
        AND the opposite hole is also empty (no capture), then Sunog occurs.
        """
        burned = False
        
        # Check if landed in originally empty hole on own side
        if last_hole not in originally_empty_holes:
            return burned
        
        if self.current_player == 0 and 0 <= last_hole <= 6:
            # Player 1 landed in own originally empty hole
            opposite_hole = 14 - last_hole
            if self.board[opposite_hole] == 0:  # Opposite hole is also empty
                # Sunog occurs
                seeds = self.board[last_hole]
                self.board[last_hole] = 0
                self.board[7] += seeds  # Give to opponent's head
                self.burned_holes[0].add(last_hole)
                burned = True
                print(f"🔥 SUNOG! Player 1's hole {last_hole} burned. {seeds} seed(s) moved to Player 1's head.")
        
        elif self.current_player == 1 and 8 <= last_hole <= 14:
            # Player 2 landed in own originally empty hole  
            opposite_hole = 14 - last_hole
            if self.board[opposite_hole] == 0:  # Opposite hole is also empty
                # Sunog occurs
                seeds = self.board[last_hole]
                self.board[last_hole] = 0
                self.board[15] += seeds  # Give to opponent's head
                self.burned_holes[1].add(last_hole)
                burned = True
                print(f"🔥 SUNOG! Player 2's hole {last_hole} burned. {seeds} seed(s) moved to Player 2's head.")
        
        return burned



    def is_game_over(self):
        if self.current_player == 0:
            return sum(self.board[0:7]) == 0
        else:
            return sum(self.board[8:15]) == 0

    def collect_remaining_stones(self):
        if self.current_player == 0:
            remaining = sum(self.board[8:15])
            self.board[15] += remaining
            for i in range(8, 15):
                self.board[i] = 0
        else:
            remaining = sum(self.board[0:7])
            self.board[7] += remaining
            for i in range(0, 7):
                self.board[i] = 0

    def get_valid_moves(self, player):
        valid_moves = []
        if player == 0:
            for i in range(0, 7):
                if self.is_valid_move(i):
                    valid_moves.append(i)
        else:
            for i in range(8, 15):
                if self.is_valid_move(i):
                    valid_moves.append(i)
        return valid_moves

    def get_winner(self):
        if self.board[7] > self.board[15]:
            return 0
        elif self.board[15] > self.board[7]:
            return 1
        return None

    def play_turn(self, hole):
        before_burned = {0: set(self.burned_holes[0]), 1: set(self.burned_holes[1])}
        before_score = (self.board[7], self.board[15])
        current_player = self.current_player

        if self.is_game_over():
            self.collect_remaining_stones()
            self.print_metrics_summary()
            return "Game Over"

        if not self.is_valid_move(hole):
            raise ValueError(f"Invalid move: Hole {hole} is not valid for Player {self.current_player + 1}")

        last_hole, should_capture, extra_turn, originally_empty_holes = self.distribute_stones(hole)

        if should_capture:
            self.check_capture(last_hole)
        else:
            # Only check Sunog if no capture occurred
            self.apply_sunog_rule(last_hole, originally_empty_holes)

        # Update metrics
        after_burned = self.burned_holes
        after_score = (self.board[7], self.board[15])

        burned_created = len(after_burned[current_player] - before_burned[current_player])
        burned_suffered = len(after_burned[1 - current_player] - before_burned[1 - current_player])
        marbles_captured = (after_score[0] - before_score[0]) if current_player == 0 else (after_score[1] - before_score[1])

        self.metrics["marbles_captured"] += max(0, marbles_captured)
        self.metrics["extra_turns"] += 1 if extra_turn else 0
        self.metrics["burned_created"] += burned_created
        self.metrics["burned_suffered"] += burned_suffered
        self.metrics["moves"] += 1

        if extra_turn:
            return "Extra Turn"

        self.current_player = 1 - self.current_player

        if self.is_game_over():
            self.collect_remaining_stones()
            self.print_metrics_summary()
            return "Game Over"

        return "Turn Complete"

    def print_metrics_summary(self):
        print("\n=== PERFORMANCE METRICS SUMMARY ===")
        total_moves = self.metrics["moves"]
        if total_moves == 0:
            print("No moves were made.")
            return
        print(f"Total Moves: {total_moves}")
        print(f"Total Marbles Captured: {self.metrics['marbles_captured']} (Avg: {self.metrics['marbles_captured']/total_moves:.2f} per move)")
        print(f"Total Extra Turns: {self.metrics['extra_turns']} (Avg: {self.metrics['extra_turns']/total_moves:.2f} per move)")
        print(f"Burned Holes Created: {self.metrics['burned_created']} (Avg: {self.metrics['burned_created']/total_moves:.4f} per move)")
        print(f"Burned Holes Suffered: {self.metrics['burned_suffered']} (Avg: {self.metrics['burned_suffered']/total_moves:.4f} per move)")
        print("===================================")

    def print_board_state(self):
        print("\n--- Current Board State ---")
        print(f"Current Player: {'Player 1 (Bottom)' if self.current_player == 0 else 'Player 2 (Top)'}")
        print(f"Player 2 (Top) Head (15): {self.board[15]}")
        print("Player 2 (Top) Holes (14-8):", self.board[14:7:-1])  
        print(f"Player 1 (Bottom) Head (7): {self.board[7]}")
        print("Player 1 (Bottom) Holes (0-6):", self.board[0:7])
        if self.burned_holes[0] or self.burned_holes[1]:
            print("🔥 Burned Holes:")
            if self.burned_holes[0]:
                print(f"   Player 1: {sorted(list(self.burned_holes[0]))}")
            if self.burned_holes[1]:
                print(f"   Player 2: {sorted(list(self.burned_holes[1]))}")
        print("-------------------------")



def manual_test_game():
    game = SungkaGame()
    heuristic = SungkaHeuristic(game) 

    while True:
        print("\n" + "="*40)
        game.print_board_state()

        valid_moves = game.get_valid_moves(game.current_player)
        if valid_moves:
            print(f"\nEvaluating possible moves for Player {game.current_player + 1}:\n")

            scored_moves = []
            for move in valid_moves:
                score, details = heuristic.evaluate_move_verbose(move)
                print(f"Hole {move}:")
                for k, v in details.items():
                    print(f"  - {k}: {v:+.2f}")
                print(f"  => Total Score: {details['Total Score']:+.2f}\n")
                scored_moves.append((move, score))

            if scored_moves:
                best_move = max(scored_moves, key=lambda x: x[1])[0]
                print(f"Suggested Best Move: Hole {best_move}")
        
        try:
            player_side = "1 (0-6)" if game.current_player == 0 else "2 (8-14)"
            hole = input(f"\nPlayer {game.current_player + 1}'s turn\n"
                        f"Choose a hole to distribute stones ({player_side}): ")
            
            if hole.lower() == 'q':
                break
                
            hole = int(hole)
            result = game.play_turn(hole)
            
            print(f"\n=== Turn Result: {result} ===")
            
            if result == "Game Over":
                game.print_board_state()
                winner = game.get_winner()
                if winner is not None:
                    print(f"\n🏆 Winner: Player {winner + 1} 🏆")
                else:
                    print("\n🤝 It's a draw! 🤝")
                break
                
        except ValueError as e:
            print(f"\nError: {e}")
            continue
            
        # Pause between turns
        if result != "Extra Turn":
            input("\nPress Enter to continue to next turn...")

def main():
    print("Sungka Game with Corrected Sunog Rule")
    manual_test_game()


if __name__ == "__main__":
    main()