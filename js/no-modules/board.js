// board.js - Board DOM setup (no game logic — backend is source of truth)

function initializeBoard() {
    debugLog("Initializing game board");
    const board = document.getElementById('board');
    if (!board) {
        console.error("Board element not found!");
        debugLog("ERROR: Board element not found!");
        return;
    }
    board.innerHTML = '';

    for (let row = 0; row < BOARD_SIZE; row++) {
        for (let col = 0; col < BOARD_SIZE; col++) {
            const cell = document.createElement('div');
            cell.classList.add('cell', 'covered');
            cell.dataset.row = row;
            cell.dataset.col = col;
            cell.addEventListener('click', () => handleCellClick(row, col, cell));
            board.appendChild(cell);
        }
    }
}

function updateTurnIndicator() {
    const turnIndicator = document.getElementById('turn-indicator');
    if (turnIndicator) {
        turnIndicator.textContent = `Player ${gameState.currentPlayer}'s Turn`;
        turnIndicator.style.backgroundColor = PLAYER_COLORS[gameState.currentPlayer];
    }
}

function highlightCell(row, col, className) {
    const cell = document.querySelector(`.cell[data-row="${row}"][data-col="${col}"]`);
    if (cell) {
        cell.classList.add(className);
    }
}

function clearValidMoves() {
    document.querySelectorAll('.valid-move, .valid-capture').forEach(cell => {
        cell.classList.remove('valid-move', 'valid-capture');
    });
    gameState.validMoves = [];
}
