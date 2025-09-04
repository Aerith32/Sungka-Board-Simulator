import pandas as pd
from datetime import datetime
import os

class GameLogger:
    def __init__(self, save_directory=None):
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
        
        # New detailed move logging
        self.move_log_data = {
            'Move Number': [],
            'Timestamp': [],
            'Player': [],
            'Hole Selected': [],
            'Stones Distributed': [],
            'Board Before Move': [],
            'Board After Move': [],
            'Action Result': [],
            'Stones Captured': [],
            'Extra Turn': [],
            'Burned Holes Created': [],
            'Player 1 Score': [],
            'Player 2 Score': [],
            'Score Difference': []
        }
        
        self.session_start = datetime.now()
        self.move_counter = 0
        
        # Set save directory
        if save_directory is None:
            # Default save directory - CHANGE THIS to your preferred location
            save_directory = "F:/Oppah~/Programs/thesis/from dylan/game_logs/"  # Your custom default path
            # Alternative: save_directory = "./" for current directory
        
        # Normalize the path and handle spaces properly
        self.save_directory = os.path.normpath(save_directory)
        
        # Create directory if it doesn't exist
        try:
            if not os.path.exists(self.save_directory):
                os.makedirs(self.save_directory)
                print(f"Created directory: {self.save_directory}")
        except Exception as e:
            print(f"Warning: Could not create directory '{self.save_directory}': {e}")
            print("Falling back to current directory")
            self.save_directory = "./"
        
        self.filename = f"sungka_session_{self.session_start.strftime('%Y%m%d_%H%M%S')}.xlsx"
        self.filepath = os.path.join(self.save_directory, self.filename)
    
    def record_move(self, game, action, hole_played=None, best_move=None, best_score=None):
        """Record basic session data (original functionality)"""
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
    
    def record_detailed_move(self, game, hole_selected, board_before, action_result, 
                           stones_captured=0, extra_turn=False, burned_holes_created=None):
        """Record detailed move information"""
        self.move_counter += 1
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        if burned_holes_created is None:
            burned_holes_created = []
        
        stones_distributed = board_before[hole_selected] if hole_selected is not None else 0
        
        self.move_log_data['Move Number'].append(self.move_counter)
        self.move_log_data['Timestamp'].append(timestamp)
        self.move_log_data['Player'].append(f"Player {game.current_player + 1}")
        self.move_log_data['Hole Selected'].append(hole_selected)
        self.move_log_data['Stones Distributed'].append(stones_distributed)
        self.move_log_data['Board Before Move'].append(str(board_before))
        self.move_log_data['Board After Move'].append(str(game.board))
        self.move_log_data['Action Result'].append(action_result)
        self.move_log_data['Stones Captured'].append(stones_captured)
        self.move_log_data['Extra Turn'].append(extra_turn)
        self.move_log_data['Burned Holes Created'].append(','.join(map(str, burned_holes_created)) if burned_holes_created else '')
        self.move_log_data['Player 1 Score'].append(game.board[7])
        self.move_log_data['Player 2 Score'].append(game.board[15])
        self.move_log_data['Score Difference'].append(game.board[7] - game.board[15])
    
    def save_to_excel(self):
        """Save both session data and detailed move log to Excel with multiple sheets"""
        try:
            # Ensure directory exists before saving
            if not os.path.exists(self.save_directory):
                os.makedirs(self.save_directory)
            
            print(f"Attempting to save to: {self.filepath}")
            
            with pd.ExcelWriter(self.filepath, engine='openpyxl') as writer:
                # Sheet 1: Original session data
                session_df = pd.DataFrame(self.session_data)
                session_df.to_excel(writer, sheet_name='Session_Log', index=False)
                
                # Sheet 2: Detailed move log
                if self.move_log_data['Move Number']:  # Only if we have move data
                    move_df = pd.DataFrame(self.move_log_data)
                    move_df.to_excel(writer, sheet_name='Detailed_Moves', index=False)
                
                # Sheet 3: Game statistics summary
                if self.move_log_data['Move Number']:
                    stats_data = self.calculate_game_statistics()
                    stats_df = pd.DataFrame([stats_data])
                    stats_df.to_excel(writer, sheet_name='Game_Statistics', index=False)
            
            print(f"✅ Game session saved to {self.filepath}")
            print(f"📊 Excel file contains {len(self.move_log_data['Move Number'])} detailed moves")
            
        except Exception as e:
            print(f"❌ Error saving to Excel: {e}")
            print(f"Attempted path: {self.filepath}")
            
            # Fallback 1: Try with quotes around path
            try:
                quoted_path = f'"{self.filepath}"'
                print(f"Trying with quotes: {quoted_path}")
                # Remove quotes for actual file operation
                with pd.ExcelWriter(self.filepath, engine='openpyxl') as writer:
                    session_df = pd.DataFrame(self.session_data)
                    session_df.to_excel(writer, sheet_name='Session_Log', index=False)
                    if self.move_log_data['Move Number']:
                        move_df = pd.DataFrame(self.move_log_data)
                        move_df.to_excel(writer, sheet_name='Detailed_Moves', index=False)
                print(f"✅ Saved successfully with quoted path handling")
            except Exception as e2:
                print(f"❌ Quoted path also failed: {e2}")
                
                # Fallback 2: Save to current directory with safe filename
                safe_filename = self.filename.replace(' ', '_')
                fallback_path = os.path.join("./", safe_filename)
                try:
                    print(f"Trying fallback location: {fallback_path}")
                    with pd.ExcelWriter(fallback_path, engine='openpyxl') as writer:
                        session_df = pd.DataFrame(self.session_data)
                        session_df.to_excel(writer, sheet_name='Session_Log', index=False)
                        if self.move_log_data['Move Number']:
                            move_df = pd.DataFrame(self.move_log_data)
                            move_df.to_excel(writer, sheet_name='Detailed_Moves', index=False)
                    print(f"✅ Saved to fallback location: {fallback_path}")
                except Exception as e3:
                    print(f"❌ All save attempts failed: {e3}")
                    print("Try manually creating the directory or using a path without spaces")
    
    def calculate_game_statistics(self):
        """Calculate summary statistics from move log"""
        if not self.move_log_data['Move Number']:
            return {}
        
        total_moves = len(self.move_log_data['Move Number'])
        total_captures = sum(self.move_log_data['Stones Captured'])
        total_extra_turns = sum(self.move_log_data['Extra Turn'])
        
        # Count burned holes
        total_burns = sum(1 for burns in self.move_log_data['Burned Holes Created'] if burns)
        
        # Final scores
        final_p1_score = self.move_log_data['Player 1 Score'][-1] if self.move_log_data['Player 1 Score'] else 0
        final_p2_score = self.move_log_data['Player 2 Score'][-1] if self.move_log_data['Player 2 Score'] else 0
        
        return {
            'Game Duration': self.session_start.strftime('%Y-%m-%d %H:%M:%S'),
            'Total Moves': total_moves,
            'Total Stones Captured': total_captures,
            'Total Extra Turns': total_extra_turns,
            'Total Burned Holes': total_burns,
            'Final Player 1 Score': final_p1_score,
            'Final Player 2 Score': final_p2_score,
            'Final Score Difference': final_p1_score - final_p2_score,
            'Winner': 'Player 1' if final_p1_score > final_p2_score else 'Player 2' if final_p2_score > final_p1_score else 'Draw',
            'Avg Captures per Move': total_captures / total_moves if total_moves > 0 else 0,
            'Avg Extra Turns per Move': total_extra_turns / total_moves if total_moves > 0 else 0
        }