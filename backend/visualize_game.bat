@echo off
echo Visualizing Skillego AI Games
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

echo Running visualization...
python visualize_game.py --agent1 ml --agent2 ml --model1 models/player1/skillego_model_final.h5 --model2 models/player2/skillego_model_final.h5 --delay 0.25
pause
