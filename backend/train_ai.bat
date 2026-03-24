@echo off
echo Starting Skillego AI Backend with Training
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

echo Running training script (this may take a while)...
echo This will train the AI to make strategic decisions between uncovering and moving
echo The AI will learn to fully decide when to uncover vs when to move without any predefined logic
python self_play_train.py --episodes 2000 --batch-size 128 --target-update 100 --epsilon-decay 0.999 %*
pause
