"""
Script to visualize AI agents playing against each other
"""

import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

from models.game import Game
from ml.ai_player import get_random_move, get_heuristic_move, create_model
from tensorflow.keras.models import load_model
import tensorflow as tf

# Define piece colors and emojis
PIECE_COLORS = {
    "mouse": "#40C4FF",    # Light blue
    "cat": "#FFEB3B",      # Yellow
    "dog": "#FF9800",      # Orange
    "bear": "#9C27B0",     # Purple
    "robot": "#F44336",    # Red
    "dragon": "#4CAF50"    # Green
}

PIECE_EMOJIS = {
    "mouse": "1",
    "cat": "2",
    "dog": "3",
    "bear": "4",
    "robot": "5", 
    "dragon": "6"
}

PLAYER_COLORS = {
    1: '#ff9999',  # Softer red
    2: '#9999ff'   # Softer blue
}

class GameVisualizer:
    """Class to visualize a game of Skillego"""
    
    def __init__(self, game):
        """Initialize the visualizer"""
        self.game = game
        self.board_size = game.BOARD_SIZE
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_xlim(-0.5, self.board_size - 0.5)
        self.ax.set_ylim(-0.5, self.board_size - 0.5)
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis()  # Invert y-axis to match board coordinates
        self.ax.set_xticks(range(self.board_size))
        self.ax.set_yticks(range(self.board_size))
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.grid(True, color='black', linewidth=1, alpha=0.3)
        
        # Title for the current state
        self.title = self.ax.set_title(f"Player {self.game.current_player}'s Turn", 
                                      fontsize=16, pad=10)
        
        # Create the board visualization
        self.board_squares = []
        self.piece_texts = []
        
        for row in range(self.board_size):
            for col in range(self.board_size):
                square = patches.Rectangle((col - 0.5, row - 0.5), 1, 1, 
                                         fill=True, color='#e0c9a6', alpha=0.7)
                self.board_squares.append(self.ax.add_patch(square))
                
                # Add text for pieces
                text = self.ax.text(col, row, "", ha='center', va='center', fontsize=24)
                self.piece_texts.append(text)
        
        # Initial update
        self.update_visualization()
    
    def update_visualization(self):
        """Update the visualization based on the current game state"""
        # Update the title
        self.title.set_text(f"Player {self.game.current_player}'s Turn")
        
        # Update board squares and pieces
        for row in range(self.board_size):
            for col in range(self.board_size):
                idx = row * self.board_size + col
                piece = self.game.board[row][col]
                
                # Update square
                if piece is None:
                    # Empty square
                    self.board_squares[idx].set_color('#e0c9a6')
                    self.piece_texts[idx].set_text("")
                elif piece["covered"]:
                    # Covered piece
                    self.board_squares[idx].set_color('#9a8866')
                    self.piece_texts[idx].set_text("?")
                    self.piece_texts[idx].set_color('white')
                else:
                    # Uncovered piece
                    player_color = PLAYER_COLORS[piece["player"]]
                    self.board_squares[idx].set_color(player_color)
                    self.piece_texts[idx].set_text(piece["power"])
                    self.piece_texts[idx].set_color('black')
        
        # Force redraw
        self.fig.canvas.draw()
    
    def animate_move(self, move):
        """Animate a move on the board"""
        if move["type"] == "uncover":
            row, col = move["row"], move["col"]
            idx = row * self.board_size + col
            piece = self.game.board[row][col]
            
            # Animate the uncovering
            self.board_squares[idx].set_color(PLAYER_COLORS[piece["player"]])
            self.piece_texts[idx].set_text(piece["power"])
            self.piece_texts[idx].set_color('black')
        
        elif move["type"] == "move":
            from_row, from_col = move["from_row"], move["from_col"]
            to_row, to_col = move["to_row"], move["to_col"]
            
            from_idx = from_row * self.board_size + from_col
            to_idx = to_row * self.board_size + to_col
            
            # Clear source square
            self.board_squares[from_idx].set_color('#e0c9a6')
            self.piece_texts[from_idx].set_text("")
            
            piece = self.game.board[to_row][to_col]
            
            # Update destination square
            self.board_squares[to_idx].set_color(PLAYER_COLORS[piece["player"]])
            self.piece_texts[to_idx].set_text(piece["power"])
            self.piece_texts[to_idx].set_color('black')
        
        # Update title
        self.title.set_text(f"Player {self.game.current_player}'s Turn")
        
        # Force redraw
        self.fig.canvas.draw()
        plt.pause(0.5)  # Pause to see the move


def get_ml_model_move(game, model_path=None):
    """
    Use a trained ML model to get a move for the current game state
    
    Args:
        game: Current Game instance
        model_path: Path to the trained model file
        
    Returns:
        dict: Move decision
    """
    try:
        # Load model if provided
        if model_path and os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
        else:
            # Use default model path based on current player
            player = game.current_player
            default_path = f"models/player{player}/skillego_model_final.h5"
            if os.path.exists(default_path):
                model = tf.keras.models.load_model(default_path)
            else:
                print(f"No model found at {model_path or default_path}, falling back to heuristic")
                return get_heuristic_move(game)
                
        # Get ALL valid moves - both uncovering and movement - without prioritizing either
        all_valid_moves = []
        
        # Add uncovering moves
        for row in range(game.BOARD_SIZE):
            for col in range(game.BOARD_SIZE):
                piece = game.board[row][col]
                if piece and piece["covered"]:
                    all_valid_moves.append({
                        "type": "uncover",
                        "row": row,
                        "col": col
                    })
        
        # Add movement moves
        for row in range(game.BOARD_SIZE):
            for col in range(game.BOARD_SIZE):
                piece = game.board[row][col]
                if piece and not piece["covered"] and piece["player"] == game.current_player:
                    piece_moves = game.get_valid_moves(row, col)
                    for move in piece_moves:
                        all_valid_moves.append({
                            "type": "move",
                            "from_row": row,
                            "from_col": col,
                            "to_row": move["row"],
                            "to_col": move["col"]
                        })
        
        if not all_valid_moves:
            print("No valid moves found, using heuristic fallback")
            return get_heuristic_move(game)
            
        # Convert game state to model input
        state = game.get_numpy_state()
        state_tensor = tf.expand_dims(state, axis=0)  # Add batch dimension
        
        # Get Q-values from model
        q_values = model.predict(state_tensor, verbose=0)[0]
        
        # Choose the move with highest expected value
        best_move = None
        best_q_value = float('-inf')
        
        # Evaluate ALL moves - let the model decide between uncovering and movement
        for move in all_valid_moves:
            move_idx = -1  # Default invalid index
            
            if move["type"] == "move":
                from_row, from_col = move["from_row"], move["from_col"]
                to_row, to_col = move["to_row"], move["to_col"]
                
                # Calculate direction (0=up, 1=right, 2=down, 3=left)
                directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
                dr = to_row - from_row
                dc = to_col - from_col
                
                for i, (dir_row, dir_col) in enumerate(directions):
                    if dir_row == dr and dir_col == dc:
                        move_idx = (from_row * game.BOARD_SIZE + from_col) * 5 + i
                        break
                        
            elif move["type"] == "uncover":
                row, col = move["row"], move["col"]
                # Uncover action is type 4
                move_idx = (row * game.BOARD_SIZE + col) * 5 + 4
            
            # Check if we have a valid index and evaluate the move
            if move_idx >= 0 and move_idx < len(q_values):
                q_value = q_values[move_idx]
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_move = move
        
        # Return the best move, whether it's uncovering or movement
        if best_move:
            return best_move
        else:
            # Fall back to heuristic if model prediction fails
            print("ML model didn't select a valid move, using heuristic fallback")
            return get_heuristic_move(game)
            
    except Exception as e:
        print(f"Error using ML model: {e}")
        # Fall back to heuristic move if there's any issue with the model
        return get_heuristic_move(game)


def play_visual_game(agent1_type="random", agent2_type="heuristic", model1_path=None, model2_path=None, delay=0.5):
    """Play a game between two AI agents with visualization"""
    game = Game()
    visualizer = GameVisualizer(game)
    
    # Set up base agents
    agent_functions = {
        "random": get_random_move,
        "heuristic": get_heuristic_move
    }
    
    # Create agent functions based on type
    if agent1_type == "ml":
        # For player 1, create a function that calls get_ml_model_move with the right model path
        agent1 = lambda game: get_ml_model_move(game, model_path=model1_path)
    else:
        agent1 = agent_functions.get(agent1_type, get_random_move)
        
    if agent2_type == "ml":
        # For player 2, create a function that calls get_ml_model_move with the right model path
        agent2 = lambda game: get_ml_model_move(game, model_path=model2_path)
    else:
        agent2 = agent_functions.get(agent2_type, get_heuristic_move)
    
    print(f"Starting game: {agent1_type} (Player 1) vs {agent2_type} (Player 2)")
    
    plt.ion()  # Turn on interactive mode
    plt.show()
    
    # Play the game
    move_count = 0
    while not game.game_over:
        current_agent = agent1 if game.current_player == 1 else agent2
        agent_type = agent1_type if game.current_player == 1 else agent2_type
        
        print(f"Move {move_count+1}: Player {game.current_player} ({agent_type}) thinking...")
        time.sleep(delay)  # Add delay to see the moves
        
        # Get move from agent
        move = current_agent(game)
        
        # Apply the move
        if move["type"] == "uncover":
            result = game.uncover(move["row"], move["col"])
            print(f"Player {game.current_player} uncovered piece at ({move['row']}, {move['col']})")
        elif move["type"] == "move":
            result = game.move(move["from_row"], move["from_col"], move["to_row"], move["to_col"])
            print(f"Player {game.current_player} moved from ({move['from_row']}, {move['from_col']}) to ({move['to_row']}, {move['to_col']})")
            if "captured" in result:
                print(f"  Captured a {result['captured']['type']} (power: {result['captured']['power']})")
        
        # Animate the move
        visualizer.animate_move(move)
        move_count += 1
        
        # Add a slight delay
        time.sleep(delay)
    
    # Game is over - display result
    if game.state == "player1_won":
        winner = 1
        winner_type = agent1_type
    else:
        winner = 2
        winner_type = agent2_type
    
    print(f"\nGame over! Player {winner} ({winner_type}) wins in {move_count} moves!")
    visualizer.title.set_text(f"Game over! Player {winner} ({winner_type}) wins!")
    visualizer.fig.canvas.draw()
    
    plt.ioff()  # Turn off interactive mode
    plt.show(block=True)  # Keep the window open


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize AI vs AI games in Skillego")
    parser.add_argument("--agent1", type=str, default="random", choices=["random", "heuristic", "ml"],
                      help="Type of agent for Player 1 (default: random)")
    parser.add_argument("--agent2", type=str, default="heuristic", choices=["random", "heuristic", "ml"],
                      help="Type of agent for Player 2 (default: heuristic)")
    parser.add_argument("--model1", type=str, default=None,
                      help="Path to the model file for Player 1 when using ML agent (default: uses models/player1/skillego_model_final.h5)")
    parser.add_argument("--model2", type=str, default=None,
                      help="Path to the model file for Player 2 when using ML agent (default: uses models/player2/skillego_model_final.h5)")
    parser.add_argument("--delay", type=float, default=0.5,
                      help="Delay between moves in seconds (default: 0.5)")
    
    args = parser.parse_args()
    
    play_visual_game(agent1_type=args.agent1, agent2_type=args.agent2, 
                   model1_path=args.model1, model2_path=args.model2, delay=args.delay)
