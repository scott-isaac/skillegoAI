"""
AI Player Module for Skillego
Contains implementations for various AI strategies for playing Skillego
"""

import os
import random
import numpy as np
from typing import Dict, List, Tuple, Any
import tensorflow as tf


def get_random_move(game) -> Dict:
    """
    Return a random valid move from all possible moves
    
    Args:
        game: The current game instance
        
    Returns:
        Dict: A move with from_row, from_col, to_row, to_col or uncover with row, col
    """
    # Collect all possible moves
    all_moves = []
    
    # Add all uncover moves first (for covered pieces)
    for row in range(game.BOARD_SIZE):
        for col in range(game.BOARD_SIZE):
            piece = game.board[row][col]
            if piece and piece["covered"]:
                all_moves.append({
                    "type": "uncover",
                    "row": row,
                    "col": col
                })
    
    # If there are uncovered moves, return a random one
    if all_moves:
        return random.choice(all_moves)
    
    # Add all move options (for uncovered pieces belonging to current player)
    for row in range(game.BOARD_SIZE):
        for col in range(game.BOARD_SIZE):
            piece = game.board[row][col]
            if piece and not piece["covered"] and piece["player"] == game.current_player:
                valid_moves = game.get_valid_moves(row, col)
                for move in valid_moves:
                    all_moves.append({
                        "type": "move",
                        "from_row": row,
                        "from_col": col,
                        "to_row": move["row"],
                        "to_col": move["col"]
                    })
    
    if all_moves:
        return random.choice(all_moves)
    else:
        # No valid moves available
        return {"error": "No valid moves available"}


def get_heuristic_move(game) -> Dict:
    """
    Return a move based on a simple heuristic
    
    Args:
        game: The current game instance
        
    Returns:
        Dict: A move with from_row, from_col, to_row, to_col or uncover with row, col
    """    # Collect uncover moves, but don't immediately return one
    uncovered_moves = []
    for row in range(game.BOARD_SIZE):
        for col in range(game.BOARD_SIZE):
            piece = game.board[row][col]
            if piece and piece["covered"]:
                uncovered_moves.append({
                    "type": "uncover",
                    "row": row,
                    "col": col
                })
    
    # For movement, prioritize captures, especially high-value captures
    capture_moves = []
    safe_moves = []
    other_moves = []
    
    for row in range(game.BOARD_SIZE):
        for col in range(game.BOARD_SIZE):
            piece = game.board[row][col]
            if piece and not piece["covered"] and piece["player"] == game.current_player:
                valid_moves = game.get_valid_moves(row, col)
                
                for move in valid_moves:
                    to_row, to_col = move["row"], move["col"]
                    target = game.board[to_row][to_col]
                    move_info = {
                        "type": "move",
                        "from_row": row,
                        "from_col": col,
                        "to_row": to_row,
                        "to_col": to_col
                    }
                    
                    if target is not None:  # Capture move
                        # Special case: mouse capturing dragon
                        if piece["type"] == "mouse" and target["type"] == "dragon":
                            # Highest priority
                            return move_info
                        
                        # Add to capture moves with the value of captured piece as priority
                        capture_moves.append((target["power"], move_info))
                    elif _is_safe_move(game, row, col, to_row, to_col):
                        safe_moves.append(move_info)
                    else:
                        other_moves.append(move_info)    # Strategic decision-making balancing uncovering and movement
    
    # If we can capture a high-value piece (power >= 4), do it regardless of uncovered pieces
    high_value_captures = [move for score, move in capture_moves if score >= 4]
    if high_value_captures:
        return high_value_captures[0]
        
    # If we have special captures (like mouse captures dragon), prioritize those
    special_captures = [move for score, move in capture_moves if score >= 6]
    if special_captures:
        return special_captures[0]
    
    # If there are covered pieces but we also have good moves, make a decision:
    # - Randomly choose to uncover if we have less than 50% of our pieces uncovered
    # - Otherwise, prefer making a strong move
    
    # Count how many of our pieces are already uncovered
    player_pieces = 0
    uncovered_player_pieces = 0
    for row in range(game.BOARD_SIZE):
        for col in range(game.BOARD_SIZE):
            piece = game.board[row][col]
            if piece and piece["player"] == game.current_player:
                player_pieces += 1
                if not piece["covered"]:
                    uncovered_player_pieces += 1
    
    uncovered_ratio = uncovered_player_pieces / max(1, player_pieces)
    
    # If we have capture moves and more than 50% of pieces uncovered, prioritize captures
    if capture_moves and uncovered_ratio > 0.5:
        capture_moves.sort(key=lambda x: x[0], reverse=True)
        return capture_moves[0][1]
    
    # If we have pieces to uncover and less than 50% uncovered, prioritize uncovering
    if uncovered_moves and uncovered_ratio < 0.5:
        return random.choice(uncovered_moves)
        
    # Otherwise, use normal priorities:
    if capture_moves:
        capture_moves.sort(key=lambda x: x[0], reverse=True)
        return capture_moves[0][1]
    
    if safe_moves:
        return random.choice(safe_moves)
    
    if uncovered_moves:
        return random.choice(uncovered_moves)
        
    if other_moves:
        return random.choice(other_moves)
    
    # No valid moves
    return {"error": "No valid moves available"}


_DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

def _is_safe_move(game, from_row: int, from_col: int, to_row: int, to_col: int) -> bool:
    """Return True if moving to (to_row, to_col) won't be immediately captured.

    No board mutation needed — we just check whether any uncovered opponent
    piece is orthogonally adjacent to the destination and can capture the piece.
    """
    piece = game.board[from_row][from_col]
    for dr, dc in _DIRECTIONS:
        nr, nc = to_row + dr, to_col + dc
        if 0 <= nr < game.BOARD_SIZE and 0 <= nc < game.BOARD_SIZE:
            opp = game.board[nr][nc]
            if (opp and not opp["covered"]
                    and opp["player"] != game.current_player
                    and (opp["power"] >= piece["power"]
                         or (opp["type"] == "mouse" and piece["type"] == "dragon"))):
                return False
    return True


def calculate_power_score(game, player: int) -> float:
    """Calculate the total power of a player's uncovered pieces."""
    return sum(
        game.board[r][c]["power"]
        for r in range(game.BOARD_SIZE)
        for c in range(game.BOARD_SIZE)
        if game.board[r][c] and not game.board[r][c]["covered"] and game.board[r][c]["player"] == player
    )


def calculate_relative_power_score(game, player: int) -> float:
    """Calculate a player's power advantage over their opponent."""
    return calculate_power_score(game, player) - calculate_power_score(game, 3 - player)


# The main function to be used by the API
def get_ai_move(game, ai_type='heuristic', model_path=None) -> Dict:
    """
    Get an AI move for the current game state.

    Args:
        game: The current game instance
        ai_type: 'random', 'heuristic', or 'ml'
        model_path: Path to a saved model file (for 'ml' type only)

    Returns:
        Dict: The selected move
    """
    if ai_type == 'random':
        return get_random_move(game)
    elif ai_type == 'ml':
        if model_path is None:
            # Prefer the unified challenger model; fall back to legacy per-player models
            challenger_path = "models/challenger/skillego_model_final.h5"
            legacy_path = f"models/player{game.current_player}/skillego_model_final.h5"
            model_path = challenger_path if os.path.exists(challenger_path) else legacy_path
        model = _get_cached_model(model_path)
        if model is None:
            return get_heuristic_move(game)
        return get_model_move(game, model)
    else:
        return get_heuristic_move(game)


_model_cache = {}

def _get_cached_model(model_path):
    if model_path not in _model_cache:
        try:
            with tf.keras.utils.custom_object_scope({'SkillegalDQNModel': SkillegalDQNModel}):
                _model_cache[model_path] = tf.keras.models.load_model(model_path)
            print(f"ML model loaded: {model_path}")
        except Exception as e:
            print(f"ML model load failed ({e}), falling back to heuristic")
            return None
    return _model_cache[model_path]


class SkillegalDQNModel(tf.keras.Model):
    """
    Deep Q-Network model for Skillego AI
    This model processes game state and outputs Q-values for all possible actions
    """
    def __init__(self):
        super(SkillegalDQNModel, self).__init__()
        # Input shape is 15x6x6: 
        # - 13 layers for piece types/players
        # - 1 layer for covered pieces
        # - 1 layer for current player
        
        # Convolutional layers to process the spatial board state
        self.conv1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', 
                                           input_shape=(15, 6, 6))
        self.conv2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')
        self.conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')
        
        # Attention mechanism to focus on important areas of the board
        self.attention = SelfAttention(128)
        
        # Process flattened board state
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.3)  # Prevent overfitting
        self.dense2 = tf.keras.layers.Dense(256, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.3)  # Prevent overfitting
        
        # Dueling DQN architecture - separate value and advantage streams
        self.value_stream = tf.keras.layers.Dense(1)  # Value of the state
        self.advantage_stream = tf.keras.layers.Dense(180)  # Advantage of each action
        
        # Number of actions: 6x6 (source) * (4 directions + uncovering) = 36 * 5 = 180
        # We don't need a separate output layer as we'll combine value and advantage streams
        
    def call(self, inputs):
        # Process input through convolutional layers
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Apply attention mechanism
        x = self.attention(x)
        
        # Flatten and process through fully connected layers
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        
        # Dueling architecture: split into value and advantage streams
        value = self.value_stream(x)  # Shape: (batch_size, 1)
        advantage = self.advantage_stream(x)  # Shape: (batch_size, 180)
        
        # Combine value and advantage streams to get Q-values
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        advantage_mean = tf.reduce_mean(advantage, axis=1, keepdims=True)
        q_values = value + (advantage - advantage_mean)
        
        return q_values


class SelfAttention(tf.keras.layers.Layer):
    """Self-attention mechanism to focus on important parts of the board state"""
    
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.query = tf.keras.layers.Conv2D(channels // 8, 1, padding='same')
        self.key = tf.keras.layers.Conv2D(channels // 8, 1, padding='same')
        self.value = tf.keras.layers.Conv2D(channels, 1, padding='same')
        self.gamma = self.add_weight(name='gamma', shape=[1], initializer='zeros')
        
    def call(self, inputs):
        batch_size, height, width, channels = inputs.shape
        
        # Create query, key, value projections
        query = self.query(inputs)  # B x H x W x C/8
        key = self.key(inputs)      # B x H x W x C/8
        value = self.value(inputs)  # B x H x W x C
          # Reshape query and key for matrix multiplication
        query_flat = tf.reshape(query, [tf.shape(query)[0], -1, tf.shape(query)[-1]])  # B x (H*W) x C/8
        key_flat = tf.reshape(key, [tf.shape(key)[0], -1, tf.shape(key)[-1]])          # B x (H*W) x C/8
        
        # Compute attention scores
        energy = tf.matmul(query_flat, key_flat, transpose_b=True)  # B x (H*W) x (H*W)
        attention = tf.nn.softmax(energy, axis=-1)        # Apply attention to value
        value_flat = tf.reshape(value, [tf.shape(value)[0], -1, tf.shape(value)[-1]])  # B x (H*W) x C
        out = tf.matmul(attention, value_flat)  # B x (H*W) x C
        out = tf.reshape(out, [tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], tf.shape(inputs)[3]])
        
        # Residual connection with learnable weight
        return self.gamma * out + inputs


def get_model_move(game, model) -> Dict:
    """
    Get a move using a trained model
    
    Args:
        game: The current game instance
        model: The trained model to use
        
    Returns:
        Dict: The move to make
    """
    # Get the current state as a numpy array
    state = game.get_numpy_state()
    
    # Expand dimensions to create a batch of size 1
    state_batch = np.expand_dims(state, axis=0)
    
    # Get predictions from the model
    q_values = model.predict(state_batch, verbose=0)[0]
    
    # Convert Q-values to potential moves and filter valid ones
    all_moves = []
    
    # Generate all potential moves
    for source_row in range(game.BOARD_SIZE):
        for source_col in range(game.BOARD_SIZE):
            # Check if there's an uncovered piece belonging to the current player
            piece = game.board[source_row][source_col]
            
            # Add uncovering action if the piece is covered
            if piece and piece["covered"]:
                # Map to action index (uncovering is the 5th action type)
                action_idx = (source_row * game.BOARD_SIZE + source_col) * 5 + 4
                all_moves.append({
                    "type": "uncover",
                    "row": source_row,
                    "col": source_col,
                    "q_value": q_values[action_idx]
                })
            
            # Add movement actions for pieces of current player
            elif piece and not piece["covered"] and piece["player"] == game.current_player:
                # Get valid moves
                valid_moves = game.get_valid_moves(source_row, source_col)
                
                # Directions: up, right, down, left
                directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
                
                for move in valid_moves:
                    # Determine the direction
                    dr = move["row"] - source_row
                    dc = move["col"] - source_col
                    direction_idx = directions.index((dr, dc))
                    
                    # Map to action index
                    action_idx = (source_row * game.BOARD_SIZE + source_col) * 5 + direction_idx
                    
                    all_moves.append({
                        "type": "move",
                        "from_row": source_row,
                        "from_col": source_col,
                        "to_row": move["row"],
                        "to_col": move["col"],
                        "q_value": q_values[action_idx]
                    })
    
    # If there are moves available, choose the one with the highest Q-value
    if all_moves:
        return max(all_moves, key=lambda x: x["q_value"])
    
    # Fall back to heuristic if no moves available (shouldn't happen)
    return get_heuristic_move(game)


def create_model():
    """Create and compile the DQN model."""
    model = SkillegalDQNModel()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse'
    )
    return model
