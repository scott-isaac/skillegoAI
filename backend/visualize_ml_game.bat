@echo off
REM Visualize a game between ML agents
echo Running ML visualization...

REM Run with ML vs ML (using default model paths for each player)
python visualize_game.py --agent1 ml --agent2 ml --delay 1.0

REM Some other examples:
REM python visualize_game.py --agent1 ml --agent2 heuristic --delay 1.0
REM python visualize_game.py --agent1 ml --agent2 ml --model1 models/player1/skillego_model_ep100.h5 --model2 models/player2/skillego_model_ep100.h5 --delay 1.0

pause
