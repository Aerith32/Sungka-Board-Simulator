import pandas as pd
from datetime import datetime

class GameLogger:
    def __init__(self):
        """Initialize the game recorder with empty data"""
        self.session_data = {
            'Timestamp': [],
            'Current Player': [],
            'Action': [],
            'Hole Played': [],
            'Player 1 Score': [],
            'Player 2 Score': [],
            'Board State': [],
            'Best Move': [],
            'Best Score': []
        }
        self.session_start = datetime.now()
        self.filename = f"sungka_session_{self.session_start.strftime('%Y%m%d_%H%M%S')}.xlsx"
    
    def record_move(self, game, action, hole_played=None, best_move=None, best_score=None):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        current_player = f"Player {game.current_player + 1}"

        self.session_data['Timestamp'].append(timestamp)
        self.session_data['Current Player'].append(current_player)
        self.session_data['Action'].append(action)
        self.session_data['Hole Played'].append(hole_played)
        self.session_data['Player 1 Score'].append(game.board[7])
        self.session_data['Player 2 Score'].append(game.board[15])
        self.session_data['Board State'].append(str(game.board))
        self.session_data['Best Move'].append(best_move)
        self.session_data['Best Score'].append(best_score)
    
    def save_to_excel(self):
        df = pd.DataFrame(self.session_data)
        df.to_excel(self.filename, index=False)
        print(f"Game session saved to {self.filename}")

