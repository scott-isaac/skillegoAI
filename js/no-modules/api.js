// api.js - API integration with Python backend

const API_BASE_URL = 'http://127.0.0.1:8080/api';
let currentGameId = null;

// Function to create a new game
async function createNewGame() {
    try {
        const response = await fetch(`${API_BASE_URL}/game/new`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        
        const data = await response.json();
        currentGameId = data.game_id;
        debugLog(`New game created with ID: ${currentGameId}`);
        
        // Update the frontend with the initial state
        updateGameFromState(data.state);
        
        return data;
    } catch (error) {
        console.error("Error creating new game:", error);
        debugLog(`Error creating new game: ${error.message}`);
        return null;
    }
}

// Function to get the current game state
async function getGameState() {
    if (!currentGameId) {
        debugLog("No active game found");
        return null;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/game/${currentGameId}`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        
        const data = await response.json();
        return data.state;
    } catch (error) {
        console.error("Error getting game state:", error);
        debugLog(`Error getting game state: ${error.message}`);
        return null;
    }
}

// Function to uncover a piece
async function uncoverPiece(row, col) {
    if (!currentGameId) {
        debugLog("No active game found");
        return null;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/game/${currentGameId}/uncover`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ row, col })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `HTTP error! Status: ${response.status}`);
        }
        
        const data = await response.json();
        debugLog(`Uncovered piece at (${row}, ${col}): ${JSON.stringify(data.result)}`);
        
        // Update the frontend with the new state
        updateGameFromState(data.state);
        
        return data;
    } catch (error) {
        console.error("Error uncovering piece:", error);
        debugLog(`Error uncovering piece: ${error.message}`);
        return { error: error.message };
    }
}

// Function to move a piece
async function movePiece(fromRow, fromCol, toRow, toCol) {
    if (!currentGameId) {
        debugLog("No active game found");
        return null;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/game/${currentGameId}/move`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                from_row: fromRow, 
                from_col: fromCol, 
                to_row: toRow, 
                to_col: toCol 
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `HTTP error! Status: ${response.status}`);
        }
        
        const data = await response.json();
        debugLog(`Moved piece from (${fromRow}, ${fromCol}) to (${toRow}, ${toCol}): ${JSON.stringify(data.result)}`);
        
        // Update the frontend with the new state
        updateGameFromState(data.state);
        
        return data;
    } catch (error) {
        console.error("Error moving piece:", error);
        debugLog(`Error moving piece: ${error.message}`);
        return { error: error.message };
    }
}

// Function to get valid moves for a piece
async function getValidMoves(row, col) {
    if (!currentGameId) {
        debugLog("No active game found");
        return [];
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/game/${currentGameId}/valid_moves?row=${row}&col=${col}`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        
        const data = await response.json();
        return data.valid_moves;
    } catch (error) {
        console.error("Error getting valid moves:", error);
        debugLog(`Error getting valid moves: ${error.message}`);
        return [];
    }
}

// Function to get an AI move
async function getAIMove() {
    if (!currentGameId) {
        debugLog("No active game found");
        return null;
    }

    const aiType = document.getElementById('ai-type-select')?.value || 'heuristic';
    debugLog(`Requesting ${aiType} AI move...`);

    try {
        const response = await fetch(`${API_BASE_URL}/game/${currentGameId}/ai_move`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ ai_type: aiType })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Execute the AI move
        const move = data.move;
        if (move.type === "uncover") {
            await uncoverPiece(move.row, move.col);
        } else if (move.type === "move") {
            await movePiece(move.from_row, move.from_col, move.to_row, move.to_col);
        } else {
            debugLog(`Unknown AI move type: ${move.type}`);
            return null;
        }
        
        return move;
    } catch (error) {
        console.error("Error getting AI move:", error);
        debugLog(`Error getting AI move: ${error.message}`);
        return null;
    }
}

// Function to update the frontend with new state data from the backend
function updateGameFromState(state) {
    // Update the game state object
    gameState.board = state.board;
    gameState.currentPlayer = state.currentPlayer;
    gameState.gameOver = state.gameOver;
    gameState.state = state.state;
    
    // Render — renderBoard() handles turn indicator and game-over display
    renderBoard();
}

// Function to toggle AI player
let aiPlayer = null;
function toggleAI(playerNumber) {
    if (aiPlayer === playerNumber) {
        aiPlayer = null;
        debugLog(`AI player ${playerNumber} disabled`);
    } else {
        aiPlayer = playerNumber;
        debugLog(`AI player ${playerNumber} enabled`);
        
        // If it's the AI's turn, make a move
        if (gameState.currentPlayer === aiPlayer && !gameState.gameOver) {
            setTimeout(() => {
                getAIMove();
            }, 1000);
        }
    }
    
    // Update the UI
    const aiButton = document.getElementById('ai-button');
    if (aiButton) {
        aiButton.textContent = aiPlayer ? `Disable AI (Player ${aiPlayer})` : 'Enable AI (Player 2)';
    }
}
