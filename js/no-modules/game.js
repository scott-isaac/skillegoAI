// game.js - Input handling (no game logic — all moves validated by backend)

async function handleCellClick(row, col, cellElement) {
    debugLog(`handleCellClick called at (${row}, ${col})`);

    if (gameState.gameOver) {
        debugLog("Ignoring click - game is over");
        return;
    }

    if (aiPlayer === gameState.currentPlayer) {
        debugLog("It's the AI's turn. Please wait.");
        return;
    }

    const cell = gameState.board[row][col];

    // Covered piece — uncover it via API
    if (cell && cell.covered) {
        await uncoverPiece(row, col);
        return;
    }

    // No selection yet — select an own uncovered piece and show valid moves
    if (!gameState.selectedCell && cell && !cell.covered && cell.player === gameState.currentPlayer) {
        gameState.selectedCell = { row, col };
        cellElement.classList.add('selected');

        const validMoves = await getValidMoves(row, col);
        gameState.validMoves = validMoves;

        for (const move of validMoves) {
            highlightCell(move.row, move.col, 'valid-move');
        }
        return;
    }

    // A piece is already selected
    if (gameState.selectedCell) {
        const isValid = gameState.validMoves.some(m => m.row === row && m.col === col);

        if (isValid) {
            await movePiece(gameState.selectedCell.row, gameState.selectedCell.col, row, col);
        }

        // Deselect regardless of whether move was valid
        const prevCell = document.querySelector(
            `.cell[data-row="${gameState.selectedCell.row}"][data-col="${gameState.selectedCell.col}"]`
        );
        if (prevCell) prevCell.classList.remove('selected');

        clearValidMoves();
        gameState.selectedCell = null;
    }
}
