"""
Skillego AI -- Training Configuration
======================================
Edit this file to control training behavior.
Run training with:  python run_overnight.py

TRAINING APPROACH: Champion/Challenger self-play evolution

  Phase 1 -- Warm-up (~5-10 min)
    Fill the replay buffer with heuristic-quality games.
    No model learning yet -- just generating good initial experiences.

  Phase 2 -- Champion/Challenger (hours, run overnight)
    Challenger trains against a frozen champion.
    Every EVAL_FREQ episodes, challenger plays EVAL_GAMES against champion.
    If challenger wins > CHAMPION_THRESHOLD, champion is replaced.
    The opponent difficulty rises continuously as the model improves.

Resume a previous run:  python run_overnight.py --resume path/to/model.h5
Skip warm-up:           python run_overnight.py --skip-warmup
"""

# ------------------------------------------------------------------------------
# EPISODES
# ------------------------------------------------------------------------------

# How many self-play games to run per phase
NUM_EPISODES = {
    1: 500,      # Warm-up: fill replay buffer (~5-10 min on CPU)
    2: 15000,    # Main: long overnight run (~40+ hrs on CPU, much less with GPU)
}

# ------------------------------------------------------------------------------
# EXPLORATION (epsilon-greedy)
# 1.0 = always use heuristic/random, 0.0 = always use the model
# ------------------------------------------------------------------------------

# Starting exploration rate for Phase 2
# Phase 1 is always 1.0 (pure heuristic -- no model decisions)
EPSILON_START = {
    1: 1.0,
    2: 0.5,   # resuming from trained model -- start mid-way through exploration
}

# Minimum exploration rate -- never go fully greedy during training
EPSILON_END = {
    1: 1.0,     # Phase 1: stays at 1.0 (pure heuristic)
    2: 0.05,    # Phase 2: ends with 5% exploration
}

# Epsilon drops by this amount per episode.
# 0.0001 with 15000 episodes: reaches EPSILON_END after ~9500 eps,
# leaving ~5500 episodes of near-pure exploitation against an evolved champion.
EPSILON_DECAY = {
    1: 0.0,
    2: 0.0001,
}

# ------------------------------------------------------------------------------
# CHAMPION / CHALLENGER
# The champion is frozen. The challenger trains against it.
# When the challenger beats the champion consistently, the champion is updated.
# ------------------------------------------------------------------------------

# Evaluate challenger vs champion every N episodes
EVAL_FREQ = 100

# Number of games per evaluation (half as P1, half as P2)
EVAL_GAMES = 50

# Win rate required to replace the champion (0.55 = 55%)
# Higher threshold = more conservative updates, more stable training
# Lower threshold = champion updates faster, potentially less stable
CHAMPION_THRESHOLD = 0.55

# ------------------------------------------------------------------------------
# REPLAY BUFFER
# ------------------------------------------------------------------------------

# Maximum stored transitions. ~750 MB at 50k with float32 15x6x6 states.
REPLAY_MEMORY_SIZE = 50000

# Don't start training until buffer has this many experiences.
MIN_REPLAY_SIZE = 500

# Cap games at this many moves to prevent infinite shuffling draws.
MAX_MOVES_PER_GAME = 300

# ------------------------------------------------------------------------------
# NEURAL NETWORK / OPTIMIZATION
# ------------------------------------------------------------------------------

# Samples per gradient update
BATCH_SIZE = 64

# Discount factor: 0.99 = values rewards ~100 moves ahead
GAMMA = 0.99

# Copy main -> target network every N gradient steps
TARGET_UPDATE_FREQ = 500

# Gradient update every N game steps
TRAIN_FREQ = 4

# ------------------------------------------------------------------------------
# LOGGING & SAVING
# ------------------------------------------------------------------------------

# Print progress every N episodes
LOG_FREQ = 10

# Save checkpoint every N episodes
SAVE_MODEL_FREQ = 200
