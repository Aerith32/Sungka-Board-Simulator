# heuristic.py
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class SungkaHeuristic:
    def __init__(self, game):
        self.original_game = game

        # Define fuzzy variables
        self.stones = ctrl.Antecedent(np.arange(0, 17, 1), 'stones')
        self.capture = ctrl.Antecedent(np.arange(0, 20, 1), 'capture')
        self.head_landing = ctrl.Antecedent(np.arange(0, 2, 1), 'head_landing')
        self.score = ctrl.Consequent(np.arange(0, 101, 1), 'score')

        # Membership functions
        self.stones['low'] = fuzz.trimf(self.stones.universe, [0, 0, 6])
        self.stones['medium'] = fuzz.trimf(self.stones.universe, [4, 7, 10])
        self.stones['high'] = fuzz.trimf(self.stones.universe, [8, 14, 16])

        self.capture['none'] = fuzz.trimf(self.capture.universe, [0, 0, 1])
        self.capture['small'] = fuzz.trimf(self.capture.universe, [1, 3, 7])
        self.capture['large'] = fuzz.trimf(self.capture.universe, [5, 12, 20])

        self.head_landing['no'] = fuzz.trimf(self.head_landing.universe, [0, 0, 0.5])
        self.head_landing['yes'] = fuzz.trimf(self.head_landing.universe, [0.5, 1, 1])

        self.score['poor'] = fuzz.trimf(self.score.universe, [0, 20, 40])
        self.score['average'] = fuzz.trimf(self.score.universe, [30, 50, 70])
        self.score['good'] = fuzz.trimf(self.score.universe, [60, 75, 85])
        self.score['excellent'] = fuzz.trimf(self.score.universe, [80, 90, 100])

        # Fuzzy rules
        rule1 = ctrl.Rule(self.stones['high'] & self.capture['large'], self.score['excellent'])
        rule2 = ctrl.Rule(self.stones['medium'] & self.capture['small'], self.score['good'])
        rule3 = ctrl.Rule(self.stones['low'] & self.capture['none'], self.score['poor'])
        rule4 = ctrl.Rule(self.head_landing['yes'], self.score['good'])

        self.move_score_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])

    def evaluate_move_verbose(self, hole):
        game = self.original_game
        board = game.board.copy()
        current_player = game.current_player

        if not ((current_player == 0 and 0 <= hole <= 6 and board[hole] > 0) or
                (current_player == 1 and 8 <= hole <= 14 and board[hole] > 0)):
            return -float('inf'), {}

        original_stones = board[hole]
        board[hole] = 0
        current_hole = hole
        stones = original_stones

        while stones > 0:
            current_hole = (current_hole + 1) % 16
            if (current_player == 0 and current_hole == 15) or (current_player == 1 and current_hole == 7):
                continue
            board[current_hole] += 1
            stones -= 1

        head = 7 if current_player == 0 else 15
        opponent_head = 15 if current_player == 0 else 7

        # Metric: Capture amount
        capture_value = 0
        burn_created = 0
        burn_suffered = 0

        if ((current_player == 0 and 0 <= current_hole <= 6 and board[current_hole] == 1) or
            (current_player == 1 and 8 <= current_hole <= 14 and board[current_hole] == 1)):
            opposite_hole = 14 - current_hole
            capture_value = board[opposite_hole]

        # Metric: Extra turn
        extra_turn_flag = 1 if current_hole == head else 0

        # Burn detection
        if hasattr(game, 'burned_holes'):
            prev_burned = set(game.burned_holes.get(1-current_player, []))
            new_burned = set(i for i in range(8) if board[i] == 0)  # adjust to your game's burned-hole rules
            burn_created = len(new_burned - prev_burned)

            prev_burned_own = set(game.burned_holes.get(current_player, []))
            new_burned_own = set(i for i in range(8) if board[i] == 0)
            burn_suffered = len(new_burned_own - prev_burned_own)

        # Fuzzy score
        try:
            move_score_simulator = ctrl.ControlSystemSimulation(self.move_score_ctrl)
            move_score_simulator.input['stones'] = original_stones
            move_score_simulator.input['capture'] = capture_value
            move_score_simulator.input['head_landing'] = extra_turn_flag
            move_score_simulator.compute()
            fuzzy_score = move_score_simulator.output.get('score', 0)
        except Exception:
            fuzzy_score = 0

        # Hybrid adjustments
        starvation_bonus = 0
        opponent_range = range(8, 15) if current_player == 0 else range(0, 7)
        opponent_empty_count = sum(1 for i in opponent_range if board[i] == 0)
        starvation_bonus = opponent_empty_count * 2

        total_score = fuzzy_score + starvation_bonus
        breakdown = {
            "Stones in Hole": original_stones,
            "Capture Value": capture_value,
            "Extra Turn": extra_turn_flag,
            "Burn Created": burn_created,
            "Burn Suffered": burn_suffered,
            "Fuzzy Score": fuzzy_score,
            "Starvation Bonus": starvation_bonus,
            "Total Score": total_score
        }

        return total_score, breakdown
