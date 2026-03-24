@echo off
REM Visualize strategic AI decision making
echo Running Strategic AI Visualization...
cd %~dp0
IF NOT EXIST ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    echo Installing requirements...
    pip install -r requirements.txt
) ELSE (
    call .venv\Scripts\activate.bat
)

echo Running visualization of Strategic AI vs Strategic AI...
echo This will show how the AI makes its own decisions about uncovering vs. moving
python visualize_game.py --agent1 ml --agent2 ml --model1 models/player1/skillego_model_final.h5 --model2 models/player2/skillego_model_final.h5 --delay 0.5

pause
