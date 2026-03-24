# Skillego

A two-player strategy board game with a self-learning AI opponent, built from scratch. The AI trains entirely through self-play using Deep Q-Networks — no hand-crafted strategy, no supervised data.

## The Game

Skillego is played on a 6×6 board. Each player has 18 pieces, all shuffled together face-down at the start:

| Piece  | Power | Count |
|--------|-------|-------|
| Mouse  | 1     | 6     |
| Cat    | 2     | 4     |
| Dog    | 3     | 4     |
| Wizard | 4     | 2     |
| Robot  | 5     | 1     |
| Dragon | 6     | 1     |

**On your turn**, choose one action:
- **Uncover** any face-down piece on the board (could be yours or your opponent's — you don't know until it flips)
- **Move** one of your uncovered pieces one square (up/down/left/right, not diagonal)

**Capturing**: move into an opponent's uncovered piece if your power ≥ theirs.
**Special rule**: Mouse (1) captures Dragon (6). Dragon cannot capture Mouse.
**Win condition**: all pieces revealed and your opponent has none left.

The hidden information creates a strategic tension — aggressive uncovering reveals the board faster but may expose strong opponent pieces.

## AI Approach

The AI learns entirely through self-play with no hard-coded strategy.

### Architecture: Dueling Double DQN

- **Input**: 15×6×6 state tensor (12 piece-type/player layers + covered + empty + current player)
- **Output**: 180 Q-values (36 positions × 5 actions: 4 move directions + uncover)
- **Dueling streams**: separate Value and Advantage heads, combined at output
- **Double DQN**: main network selects actions, frozen target network evaluates them — reduces Q-value overestimation

### Training: Champion/Challenger Self-Play

Rather than training against a fixed opponent, the AI uses an evolving target:

1. **Warm-up** (Phase 1): fill the replay buffer with heuristic games to bootstrap early learning
2. **Champion/Challenger** (Phase 2):
   - A *challenger* model trains with epsilon-greedy exploration (pure random, no heuristic bias)
   - A *champion* model is frozen — plays greedily, never updates
   - Every 100 episodes, challenger plays 50 evaluation games against champion
   - If challenger wins >55%, it *becomes* the new champion
   - The difficulty floor rises continuously — the model always has to beat its own best

This means the AI can never plateau against a fixed opponent. Each time it improves enough, the bar resets to its new level.

### Reward Shaping Philosophy

The reward signal is intentionally minimal:
- **Per step**: change in relative power score (small continuous signal)
- **Episode end**: ±10 for win/loss

No bonuses for uncovering, capturing, or positional features. The model discovers *why* those actions are valuable through their consequences, rather than being told. This allows it to find strategies that hand-crafted rewards might never incentivize.

## Project Structure

```
skillegoAI/
├── index.html              # Game UI
├── styles.css
├── js/no-modules/          # Frontend (thin display layer — no game logic)
│   ├── main.js             # Init, AI turn observer
│   ├── game.js             # Click handling → API calls
│   ├── board.js            # DOM setup
│   ├── renderers.js        # Render state from API response
│   ├── api.js              # All API communication
│   ├── state.js            # gameState object
│   ├── constants.js        # Display constants
│   └── utils.js            # Debug logging
└── backend/
    ├── app.py              # Flask API + frontend file server
    ├── run_overnight.py    # Champion/challenger training runner
    ├── training_config.py  # All hyperparameters in one place
    ├── evaluate_ai.py      # Head-to-head evaluation tool
    ├── ai_tournament.py    # Multi-agent tournament runner
    ├── visualize_game.py   # Matplotlib game visualizer
    ├── models/
    │   └── game.py         # Game engine (single source of truth)
    └── ml/
        └── ai_player.py    # DQN model, heuristic, and random agents
```

**Architecture principle**: the Python game engine is the single source of truth. The JavaScript frontend is a pure display/input layer — it renders state returned by the API and posts moves back. No game logic lives in the browser.

## Running Locally

**Requirements**: Python 3.8+

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:8080** in your browser.

Select **AI: Heuristic** for a rules-based opponent, or **AI: ML Model** if you have a trained model at `backend/models/challenger/skillego_model_final.h5`.

## Training the AI

```bash
cd backend

# Full run: warm-up then champion/challenger (overnight on CPU, faster with GPU)
python -u run_overnight.py

# Skip warm-up, resume from an existing checkpoint
python -u run_overnight.py --skip-warmup --resume models/challenger/skillego_model_final.h5
```

Training logs every 10 episodes and saves checkpoints every 200. The latest model is always written to `models/challenger/skillego_model_final.h5` so you can play against it mid-training without stopping the run.

All hyperparameters (episodes, epsilon schedule, champion threshold, replay buffer size, etc.) are in `training_config.py`.

## Tech Stack

- **Backend**: Python, Flask, TensorFlow/Keras, NumPy
- **Frontend**: Vanilla JavaScript, HTML/CSS
- **Communication**: REST API (JSON)
- **AI**: Deep Q-Network, experience replay, Double DQN, champion/challenger self-play
