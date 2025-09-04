# For your 84% win rate with 100 games:
import math
win_rate = 0.84
n = 100
se = math.sqrt(win_rate * (1 - win_rate) / n)
margin = 1.96 * se
ci_lower = (win_rate - margin) * 100  # ≈ 76.8%
ci_upper = (win_rate + margin) * 100  # ≈ 91.2%