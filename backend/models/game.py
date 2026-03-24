import random
from enum import Enum
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

class GameState:
    """Enum representing the state of the game"""
    ONGOING = "ongoing"
    PLAYER1_WON = "player1_won"
    PLAYER2_WON = "player2_won"

class PieceType:
    """Class representing piece types and their properties"""
    MOUSE = {"type": "mouse", "power": 1, "quantity": 6, "emoji": "🐭"}
    CAT = {"type": "cat", "power": 2, "quantity": 4, "emoji": "😸"}
    DOG = {"type": "dog", "power": 3, "quantity": 4, "emoji": "🐶"}
    WIZARD = {"type": "bear", "power": 4, "quantity": 2, "emoji": "🧙‍♂️"}  # Named bear in frontend
    ROBOT = {"type": "robot", "power": 5, "quantity": 1, "emoji": "🤖"}
    DRAGON = {"type": "dragon", "power": 6, "quantity": 1, "emoji": "🐉"}
    
    @classmethod
    def get_all_pieces(cls) -> List[Dict]:
        """Get all piece types"""
        return [cls.MOUSE, cls.CAT, cls.DOG, cls.WIZARD, cls.ROBOT, cls.DRAGON]

class Game:
    """Represents a Skillego game"""
    
    def __init__(self):
        """Initialize a new game"""
        self.BOARD_SIZE = 6
        self.board = [[None for _ in range(self.BOARD_SIZE)] for _ in range(self.BOARD_SIZE)]
        self.current_player = 1
        self.state = GameState.ONGOING
        self.game_over = False
        
        # Initialize the board with pieces
        self._initialize_board()
    
    def _initialize_board(self):
        """Place pieces randomly on the board"""
        all_pieces = []
        
        # Create pieces for both players
        for player in [1, 2]:
            for piece_info in PieceType.get_all_pieces():
                for _ in range(piece_info["quantity"]):
                    piece = {
                        "type": piece_info["type"],
                        "power": piece_info["power"],
                        "player": player,
                        "covered": True,
                        "emoji": piece_info["emoji"]
                    }
                    all_pieces.append(piece)
        
        # Shuffle pieces
        random.shuffle(all_pieces)
        
        # Place pieces on the board
        index = 0
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                self.board[row][col] = all_pieces[index]
                index += 1
    
    def get_state(self) -> Dict:
        """Get the current state of the game"""
        # Deep copy the board to avoid modifying the original
        board_copy = []
        
        for row in range(self.BOARD_SIZE):
            board_row = []
            for col in range(self.BOARD_SIZE):
                piece = self.board[row][col]
                if piece is None:
                    board_row.append(None)
                elif piece["covered"]:
                    # Only send minimal info for covered pieces
                    board_row.append({"covered": True})
                else:
                    # Send full info for uncovered pieces
                    board_row.append(piece)
            board_copy.append(board_row)
        
        return {
            "board": board_copy,
            "currentPlayer": self.current_player,
            "gameOver": self.game_over,
            "state": self.state
        }
    
    def uncover(self, row: int, col: int) -> Dict:
        """Uncover a piece on the board"""
        # Validate input
        if not self._is_valid_position(row, col):
            return {"error": "Invalid position"}
        
        # Check if game is over
        if self.game_over:
            return {"error": "Game is over"}
        
        piece = self.board[row][col]
        
        # Check if the piece is already uncovered
        if not piece["covered"]:
            return {"error": "Piece is already uncovered"}
        
        # Uncover the piece
        piece["covered"] = False
        
        # Switch turns
        self.current_player = 3 - self.current_player  # Toggle between 1 and 2
        
        # Check if the game is over after uncovering
        self._check_game_over()
        
        return {"uncovered": piece}
    
    def get_valid_moves(self, row: int, col: int) -> List[Dict]:
        """Get valid moves for a piece"""
        if not self._is_valid_position(row, col):
            return []
        
        piece = self.board[row][col]
        
        # If there's no piece or the piece is covered or doesn't belong to current player
        if (piece is None or piece["covered"] or piece["player"] != self.current_player):
            return []
        
        valid_moves = []
        
        # Check the four adjacent positions (up, right, down, left)
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            # Skip if out of bounds
            if not self._is_valid_position(new_row, new_col):
                continue
                
            target = self.board[new_row][new_col]
            
            # Skip if the target cell is covered
            if target is not None and target["covered"]:
                continue
                
            # Skip if the target cell has a piece of the current player
            if target is not None and not target["covered"] and target["player"] == self.current_player:
                continue
                
            # Special case: Mouse can capture Dragon
            if piece["type"] == "mouse" and target is not None and not target["covered"] and target["type"] == "dragon":
                valid_moves.append({"row": new_row, "col": new_col})
                continue
                
            # Special case: Dragon cannot capture Mouse
            if piece["type"] == "dragon" and target is not None and not target["covered"] and target["type"] == "mouse":
                continue
                
            # Normal case: Piece can capture pieces of equal or lower power
            if target is None or target["player"] != self.current_player and piece["power"] >= target["power"]:
                valid_moves.append({"row": new_row, "col": new_col})
        
        return valid_moves
    
    def move(self, from_row: int, from_col: int, to_row: int, to_col: int) -> Dict:
        """Move a piece on the board"""
        # Validate positions
        if not (self._is_valid_position(from_row, from_col) and self._is_valid_position(to_row, to_col)):
            return {"error": "Invalid position"}
        
        # Check if game is over
        if self.game_over:
            return {"error": "Game is over"}
        
        # Get the source piece
        source_piece = self.board[from_row][from_col]
        
        # Check if there's a piece to move
        if source_piece is None:
            return {"error": "No piece to move"}
        
        # Check if the piece is uncovered
        if source_piece["covered"]:
            return {"error": "Cannot move a covered piece"}
        
        # Check if the piece belongs to the current player
        if source_piece["player"] != self.current_player:
            return {"error": "Cannot move opponent's piece"}
        
        # Check if the move is valid
        valid_moves = self.get_valid_moves(from_row, from_col)
        move_is_valid = any(move["row"] == to_row and move["col"] == to_col for move in valid_moves)
        
        if not move_is_valid:
            return {"error": "Invalid move"}
        
        # Get the target piece (if any)
        target_piece = self.board[to_row][to_col]
        captured_piece = None
        
        # If there's a piece at the destination, it's captured
        if target_piece is not None:
            captured_piece = target_piece
        
        # Move the piece
        self.board[to_row][to_col] = source_piece
        self.board[from_row][from_col] = None
        
        # Switch turns
        self.current_player = 3 - self.current_player  # Toggle between 1 and 2
        
        # Check if the game is over after the move
        self._check_game_over()
        
        result = {"moved": True}
        if captured_piece:
            result["captured"] = captured_piece
        
        return result
    
    def _is_valid_position(self, row: int, col: int) -> bool:
        """Check if a position is valid"""
        return 0 <= row < self.BOARD_SIZE and 0 <= col < self.BOARD_SIZE
    
    def _check_game_over(self) -> None:
        """Check if the game is over"""
        player1_pieces = 0
        player2_pieces = 0
        covered_pieces = 0
        
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                piece = self.board[row][col]
                if piece is not None:
                    if piece["covered"]:
                        covered_pieces += 1
                    elif piece["player"] == 1:
                        player1_pieces += 1
                    elif piece["player"] == 2:
                        player2_pieces += 1
          # Game is over if all pieces are uncovered and one player has no pieces
        if covered_pieces == 0:
            if player1_pieces == 0:
                self.game_over = True
                self.state = GameState.PLAYER2_WON
            elif player2_pieces == 0:
                self.game_over = True
                self.state = GameState.PLAYER1_WON
                
    # Class-level constant — avoids rebuilding this dict on every call
    _PIECE_TYPE_TO_IDX = {"mouse": 0, "cat": 1, "dog": 2, "bear": 3, "robot": 4, "dragon": 5}

    def get_numpy_state(self) -> np.ndarray:
        """
        Convert the game state to a numpy array for machine learning

        Returns:
            np.ndarray: A representation of the board state
        """
        # Create a more informative board representation:
        # - 12 layers for pieces (6 piece types × 2 players)
        # - 1 layer for covered pieces
        # - 1 layer for empty cells
        # - 1 layer for current player
        state = np.zeros((15, self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.float32)

        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                piece = self.board[row][col]
                
                if piece is None:
                    # Empty cell
                    state[12, row, col] = 1
                elif piece["covered"]:
                    # Covered piece (unknown type)
                    state[13, row, col] = 1
                else:
                    # Uncovered piece
                    piece_type = piece["type"]
                    player = piece["player"]
                    
                    # Get base index for piece type
                    base_idx = self._PIECE_TYPE_TO_IDX[piece_type]
                    
                    # Adjust index for player 2 (offset by 6)
                    if player == 2:
                        base_idx += 6
                    
                    # Set the piece representation with its power as the value
                    state[base_idx, row, col] = piece["power"]
        
        # Set the current player (fill entire layer)
        current_player_value = self.current_player - 1  # 0 for player 1, 1 for player 2
        state[14, :, :] = current_player_value
        
        return state
    
    def get_power_score(self, player: int) -> float:
        """
        Calculate the total power score for a player based on their uncovered pieces
        
        Args:
            player: The player to calculate the score for (1 or 2)
            
        Returns:
            float: Total power of uncovered pieces
        """
        power_score = 0
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                piece = self.board[row][col]
                if piece and not piece["covered"] and piece["player"] == player:
                    power_score += piece["power"]
        return power_score
    
    def get_relative_power_score(self, player: int) -> float:
        """
        Calculate the relative power advantage a player has over their opponent
        
        Args:
            player: The player to calculate the score for (1 or 2)
            
        Returns:
            float: Power difference between player and opponent
        """
        player_score = self.get_power_score(player)
        opponent = 3 - player  # Toggle between 1 and 2
        opponent_score = self.get_power_score(opponent)
        
        return player_score - opponent_score
