@echo off
REM Visualize ML vs Heuristic comparison
echo Running ML vs Heuristic comparison...
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

echo Running visualization of ML model vs Heuristic player...
echo This will show how the ML model's balanced strategy performs
python visualize_game.py --agent1 ml --agent2 heuristic --model1 models/player1/skillego_model_final.h5 --delay 0.25

pause
