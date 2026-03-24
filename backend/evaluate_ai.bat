@echo off
echo Evaluating Skillego AI Models
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

echo Running evaluation...
python evaluate_ai.py --eval --model models/player1/skillego_model_final.h5 --opponent heuristic --games 10
pause
