@echo off
echo Running AI Tournament for Skillego
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

echo Running tournament (random vs. heuristic)...
python ai_tournament.py --agent1 random --agent2 heuristic --games 50
pause
