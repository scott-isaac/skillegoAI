// renderers.js - Functions for rendering the board and other UI elements

function renderBoard() {
    const board = document.getElementById('board');
    if (!board) {
        console.error("Board element not found");
        return;
    }

    // Update each cell based on the current gameState
    for (let row = 0; row < BOARD_SIZE; row++) {
        for (let col = 0; col < BOARD_SIZE; col++) {
            const piece = gameState.board[row][col];
            const cellElement = document.querySelector(`.cell[data-row="${row}"][data-col="${col}"]`);
            
            if (!cellElement) {
                console.error(`Cell element not found for row ${row}, col ${col}`);
                continue;
            }
            
            // Clear cell
            cellElement.textContent = '';
            cellElement.classList.remove('covered');
            cellElement.style.backgroundColor = '#e0c9a6'; // Default board color
            
            if (!piece) {
                // Empty cell
                continue;
            }
            
            if (piece.covered) {
                // Covered piece
                cellElement.classList.add('covered');
                cellElement.style.backgroundColor = '#9a8866'; // Darker color for covered pieces
            } else {
                // Uncovered piece
                cellElement.textContent = piece.emoji;
                cellElement.style.backgroundColor = PLAYER_COLORS[piece.player];
            }
        }
    }
    
    // Update turn indicator
    updateTurnIndicator();
    
    // If game is over, show the winner message
    if (gameState.gameOver) {
        const winnerMessage = document.getElementById('winner-message');
        if (winnerMessage) {
            if (gameState.state === 'player1_won') {
                winnerMessage.textContent = 'Player 1 Wins!';
            } else if (gameState.state === 'player2_won') {
                winnerMessage.textContent = 'Player 2 Wins!';
            } else {
                winnerMessage.textContent = 'Game Over!';
            }
            winnerMessage.style.display = 'block';
        }
    }
}
