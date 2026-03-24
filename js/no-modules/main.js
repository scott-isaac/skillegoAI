// main.js - Entry point for the application

// Function to initialize the game
async function initGame() {
    console.log('Initializing game...');
    
    // Initialize the board UI
    initializeBoard();
    
    // Create a new game on the backend
    await createNewGame();
    
    // Set up the reset button
    const resetButton = document.getElementById('reset-button');
    if (resetButton) {
        resetButton.addEventListener('click', async () => {
            debugLog("Resetting game...");
            clearValidMoves();
            gameState.selectedCell = null;
            const winnerMessage = document.getElementById('winner-message');
            if (winnerMessage) winnerMessage.style.display = 'none';
            await createNewGame();
        });
    }
    
    // Set up the AI button
    const aiButton = document.getElementById('ai-button');
    if (aiButton) {
        aiButton.addEventListener('click', () => {
            toggleAI(2); // Default to Player 2 as AI
        });
    }
    
    // Set up turn observer for AI (lock prevents concurrent requests)
    let isAiThinking = false;
    setInterval(() => {
        if (!isAiThinking && aiPlayer && gameState.currentPlayer === aiPlayer && !gameState.gameOver) {
            isAiThinking = true;
            getAIMove().finally(() => { isAiThinking = false; });
        }
    }, 500);
    
    debugLog("Game initialized and ready to play!");
}

// Initialize the game when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM fully loaded');
    
    try {
        // Initialize the game
        console.log('Starting game initialization...');
        initGame();
        
        console.log('Game initialized successfully');
    } catch (error) {
        console.error('Error in game initialization:', error);
        debugLog("Error during initialization: " + error.message);
    }
});

