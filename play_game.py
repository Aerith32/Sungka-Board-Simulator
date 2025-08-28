from main import SungkaGame
from game_logger import GameLogger

def play_game_with_recording():
    game = SungkaGame()
    recorder = GameLogger()
    
    recorder.record_move(game, "Game Started")
    
    while True:
        game.print_board_state()
        
        try:
            player_side = "1 (0-6)" if game.current_player == 0 else "2 (8-14)"
            hole = int(input(f"Player {game.current_player + 1}'s turn\n"
                           f"Choose a hole to distribute stones ({player_side}): "))
            
            recorder.record_move(game, "Turn Started", hole)
            result = game.play_turn(hole)
            recorder.record_move(game, result)
            
            print("Turn result:", result)
            
            if result == "Game Over":
                game.print_board_state()
                winner = game.get_winner()
                if winner is not None:
                    print(f"Winner: Player {winner + 1}")
                else:
                    print("It's a draw!")
                recorder.record_move(game, f"Game Over - Winner: {winner}" if winner else "Game Over - Draw")
                break
        
        except ValueError as e:
            print("Error:", str(e))
            recorder.record_move(game, f"Invalid Move: {str(e)}")
        
        cont = input("Continue? (y/n): ").lower()
        if cont != 'y':
            recorder.record_move(game, "Game Aborted by User")
            break
    
    recorder.save_to_excel()
    print("Game session recorded")

if __name__ == "__main__":
    play_game_with_recording()