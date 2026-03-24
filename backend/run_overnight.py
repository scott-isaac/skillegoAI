"""
Champion/Challenger training for Skillego AI
=============================================
Phase 1: fill replay buffer with heuristic games (warm-up)
Phase 2: challenger trains against a frozen champion
         -- every EVAL_FREQ episodes, challenger plays EVAL_GAMES vs champion
         -- if challenger wins > CHAMPION_THRESHOLD, champion is replaced
         -- opponent difficulty rises continuously as the model evolves

Usage:
  python run_overnight.py                        # full run (Phase 1 + 2)
  python run_overnight.py --skip-warmup          # skip Phase 1
  python run_overnight.py --resume path/to.h5   # resume challenger from checkpoint

To play at any time:
  1. python app.py  (in a separate terminal)
  2. Open http://127.0.0.1:8080 and set AI type to "ml"
     Uses models/challenger/skillego_model_final.h5

Stop safely: Ctrl+C  (checkpoint saved every SAVE_MODEL_FREQ episodes)
"""

import numpy as np
import tensorflow as tf
import os
import shutil
import random
from collections import deque
from datetime import datetime
import argparse

from models.game import Game
from ml.ai_player import (create_model, get_heuristic_move, get_random_move,
                           calculate_relative_power_score)
# Note: get_heuristic_move is only used in the Phase 1 warm-up.
# The agent itself uses pure random exploration during Phase 2.
import training_config as cfg

RESUME_FROM_CHECKPOINT = False   # set True to auto-resume from last checkpoint


# ------------------------------------------------------------------------------
# Replay buffer
# ------------------------------------------------------------------------------

class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


# ------------------------------------------------------------------------------
# Agent (challenger and champion share this class)
# ------------------------------------------------------------------------------

class Agent:
    DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.model = create_model()
        self.target_model = create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.memory = ReplayMemory(cfg.REPLAY_MEMORY_SIZE)
        self.step_count = 0

    def load_weights(self, path):
        print(f"  Loading weights from {path}")
        loaded = tf.keras.models.load_model(path)
        self.model.set_weights(loaded.get_weights())
        self.target_model.set_weights(loaded.get_weights())

    def copy_weights_from(self, other):
        self.model.set_weights(other.model.get_weights())
        self.target_model.set_weights(other.model.get_weights())

    def get_move(self, game, greedy=False):
        """greedy=True disables epsilon exploration (used for champion and evaluation)."""
        if not greedy and random.random() < self.epsilon:
            return get_random_move(game)  # pure random exploration -- no heuristic bias
        valid_moves = self._get_valid_moves(game)
        if not valid_moves:
            return get_heuristic_move(game)
        state_t = np.expand_dims(np.array(game.get_numpy_state(), dtype=np.float32), 0)
        q = self.model(state_t, training=False)[0].numpy()
        return max(valid_moves, key=lambda m: q[self._encode_move(m, game)])

    def _get_valid_moves(self, game):
        moves = []
        for r in range(game.BOARD_SIZE):
            for c in range(game.BOARD_SIZE):
                p = game.board[r][c]
                if p and p["covered"]:
                    moves.append({"type": "uncover", "row": r, "col": c})
        for r in range(game.BOARD_SIZE):
            for c in range(game.BOARD_SIZE):
                p = game.board[r][c]
                if p and not p["covered"] and p["player"] == game.current_player:
                    for mv in game.get_valid_moves(r, c):
                        moves.append({"type": "move",
                                      "from_row": r, "from_col": c,
                                      "to_row": mv["row"], "to_col": mv["col"]})
        return moves

    def _encode_move(self, move, game):
        if move["type"] == "uncover":
            return (move["row"] * game.BOARD_SIZE + move["col"]) * 5 + 4
        dr = move["to_row"] - move["from_row"]
        dc = move["to_col"] - move["from_col"]
        direction = next((i for i, d in enumerate(self.DIRECTIONS) if d == (dr, dc)), 0)
        return (move["from_row"] * game.BOARD_SIZE + move["from_col"]) * 5 + direction

    def store_experience(self, state, action, reward, next_state, done, game):
        self.memory.add(state, self._encode_move(action, game), reward, next_state, done)

    def update_model(self):
        if len(self.memory) < max(cfg.BATCH_SIZE, cfg.MIN_REPLAY_SIZE):
            return
        batch = self.memory.sample(cfg.BATCH_SIZE)
        states      = np.array([b[0] for b in batch], dtype=np.float32)
        actions     = np.array([b[1] for b in batch])
        rewards     = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.array([b[3] for b in batch], dtype=np.float32)
        dones       = np.array([b[4] for b in batch], dtype=np.float32)

        current_q     = self.model(states,      training=False).numpy()
        next_q_main   = self.model(next_states,  training=False).numpy()
        next_q_target = self.target_model(next_states, training=False).numpy()
        target_q = current_q.copy()
        best_next = np.argmax(next_q_main, axis=1)
        for i in range(len(batch)):
            if dones[i]:
                target_q[i, actions[i]] = rewards[i]
            else:
                target_q[i, actions[i]] = rewards[i] + cfg.GAMMA * next_q_target[i, best_next[i]]
        self.model.train_on_batch(states, target_q)
        self.step_count += 1
        if self.step_count % cfg.TARGET_UPDATE_FREQ == 0:
            self.target_model.set_weights(self.model.get_weights())

    def update_epsilon(self, epsilon_end, epsilon_decay):
        self.epsilon = max(epsilon_end, self.epsilon - epsilon_decay)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        # Keep skillego_model_final.h5 pointing to latest so Flask can serve it
        final = os.path.join(os.path.dirname(path), "skillego_model_final.h5")
        shutil.copy2(path, final)


# ------------------------------------------------------------------------------
# Reward: power differential only (no hand-crafted uncover/capture bonuses)
# Win/loss bonus applied at episode end gives the outcome signal.
# ------------------------------------------------------------------------------

def _step_reward(game, player, prev_power):
    new_power = calculate_relative_power_score(game, player)
    reward = (new_power - prev_power) * 0.5
    return reward, new_power


# ------------------------------------------------------------------------------
# Evaluation: challenger vs champion, both greedy
# ------------------------------------------------------------------------------

def evaluate(challenger, champion):
    wins = 0
    for i in range(cfg.EVAL_GAMES):
        challenger_player = 1 if i % 2 == 0 else 2
        game = Game()
        steps = 0
        while not game.game_over and steps < cfg.MAX_MOVES_PER_GAME:
            cp = game.current_player
            agent = challenger if cp == challenger_player else champion
            action = agent.get_move(game, greedy=True)
            if action["type"] == "uncover":
                game.uncover(action["row"], action["col"])
            else:
                game.move(action["from_row"], action["from_col"],
                          action["to_row"], action["to_col"])
            steps += 1
        if game.state == f"player{challenger_player}_won":
            wins += 1
    return wins / cfg.EVAL_GAMES


# ------------------------------------------------------------------------------
# Phase 1: warm-up (heuristic play only, fills replay buffer)
# ------------------------------------------------------------------------------

def run_warmup(agent, episodes):
    print(f"\n-- Phase 1: Warm-up ({episodes} episodes, heuristic) --")
    start = datetime.now()

    for episode in range(1, episodes + 1):
        game = Game()
        prev_power = {1: 0.0, 2: 0.0}
        pending = {1: None, 2: None}
        steps = 0

        while not game.game_over and steps < cfg.MAX_MOVES_PER_GAME:
            cp = game.current_player
            state = np.array(game.get_numpy_state(), dtype=np.float32)
            action = get_heuristic_move(game)

            if action["type"] == "uncover":
                game.uncover(action["row"], action["col"])
            else:
                game.move(action["from_row"], action["from_col"],
                          action["to_row"], action["to_col"])

            next_state = np.array(game.get_numpy_state(), dtype=np.float32)
            reward, prev_power[cp] = _step_reward(game, cp, prev_power[cp])

            if pending[cp] is not None:
                agent.store_experience(*pending[cp], False, game)
            pending[cp] = (state, action, reward, next_state)
            steps += 1

        bonus = ({1: 10.0, 2: -10.0} if game.state == "player1_won" else
                 {1: -10.0, 2: 10.0} if game.state == "player2_won" else
                 {1: 0.0, 2: 0.0})
        for pid in [1, 2]:
            if pending[pid] is not None:
                ps, pa, pr, pns = pending[pid]
                agent.store_experience(ps, pa, pr + bonus[pid], pns, True, game)

        if episode % cfg.LOG_FREQ == 0:
            elapsed = max(1, (datetime.now() - start).seconds)
            print(f"  ep {episode:>4}/{episodes}  buf={len(agent.memory):>6}  "
                  f"{elapsed // 60}m{elapsed % 60:02d}s")

    agent.save("models/challenger/skillego_model_warmup.h5")
    elapsed_min = (datetime.now() - start).seconds / 60
    print(f"  Warm-up done in {elapsed_min:.1f} min  buf={len(agent.memory)}")


# ------------------------------------------------------------------------------
# Phase 2: champion/challenger
# ------------------------------------------------------------------------------

def run_champion_challenger(challenger, champion, episodes):
    eps_end   = cfg.EPSILON_END[2]
    eps_decay = cfg.EPSILON_DECAY[2]

    print(f"\n-- Phase 2: Champion/Challenger ({episodes} episodes) --")
    print(f"   epsilon {challenger.epsilon:.3f} -> {eps_end:.3f}  "
          f"eval every {cfg.EVAL_FREQ} eps  threshold {cfg.CHAMPION_THRESHOLD:.0%}")

    recent = deque(maxlen=100)
    champion_updates = 0
    start = datetime.now()

    for episode in range(1, episodes + 1):
        # Alternate which side challenger plays so it learns both perspectives
        challenger_player = 1 if episode % 2 == 0 else 2

        game = Game()
        prev_power = 0.0
        pending = None
        steps = 0

        while not game.game_over and steps < cfg.MAX_MOVES_PER_GAME:
            cp = game.current_player
            is_challenger = (cp == challenger_player)
            action = challenger.get_move(game) if is_challenger else champion.get_move(game, greedy=True)

            state = np.array(game.get_numpy_state(), dtype=np.float32)

            if action["type"] == "uncover":
                game.uncover(action["row"], action["col"])
            else:
                game.move(action["from_row"], action["from_col"],
                          action["to_row"], action["to_col"])

            next_state = np.array(game.get_numpy_state(), dtype=np.float32)

            if is_challenger:
                reward, prev_power = _step_reward(game, challenger_player, prev_power)
                if pending is not None:
                    challenger.store_experience(*pending, False, game)
                pending = (state, action, reward, next_state)

            steps += 1
            if steps % cfg.TRAIN_FREQ == 0:
                challenger.update_model()

        # Outcome
        if game.state == f"player{challenger_player}_won":
            bonus = 10.0
            recent.append("W")
        elif game.state != "ongoing":
            bonus = -10.0
            recent.append("L")
        else:
            bonus = 0.0
            recent.append("D")

        if pending is not None:
            ps, pa, pr, pns = pending
            challenger.store_experience(ps, pa, pr + bonus, pns, True, game)

        challenger.update_epsilon(eps_end, eps_decay)

        # Evaluate and maybe promote challenger to champion
        if episode % cfg.EVAL_FREQ == 0:
            win_rate = evaluate(challenger, champion)
            if win_rate >= cfg.CHAMPION_THRESHOLD:
                champion.copy_weights_from(challenger)
                champion_updates += 1
                os.makedirs("models/champion", exist_ok=True)
                challenger.model.save(f"models/champion/champion_{episode}.h5")
                print(f"  [ep {episode}] Champion updated #{champion_updates}  "
                      f"win rate: {win_rate:.1%}")
            else:
                print(f"  [ep {episode}] Champion holds  win rate: {win_rate:.1%}")

        if episode % cfg.SAVE_MODEL_FREQ == 0:
            challenger.save(f"models/challenger/skillego_model_ep{episode}.h5")
            print(f"  [checkpoint ep {episode}] play now: python app.py")

        if episode % cfg.LOG_FREQ == 0:
            wr = recent.count("W") / len(recent) if recent else 0
            elapsed = max(1, (datetime.now() - start).seconds)
            eta = (episodes - episode) / max(episode / elapsed, 0.001) / 60
            print(f"  ep {episode:>5}/{episodes}  "
                  f"win={wr:.0%}  eps={challenger.epsilon:.4f}  "
                  f"buf={len(challenger.memory):>6}  "
                  f"updates={champion_updates}  ETA {eta:.0f}m")

    challenger.save("models/challenger/skillego_model_final.h5")
    elapsed_min = (datetime.now() - start).seconds / 60
    w = recent.count("W")
    l = recent.count("L")
    d = recent.count("D")
    print(f"\n  Phase 2 done in {elapsed_min:.1f} min")
    print(f"  Last 100: {w}W / {l}L / {d}D  Champion updates: {champion_updates}")


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-warmup", action="store_true",
                        help="Skip Phase 1 warm-up")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to challenger checkpoint to resume from")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Skillego Champion/Challenger Training")
    print("  Challenger model -> models/challenger/skillego_model_final.h5")
    print("  Play at any time: python app.py")
    print("=" * 60)

    eps_start = cfg.EPSILON_START[2]
    challenger = Agent(eps_start)

    # Phase 1: warm-up
    if not args.skip_warmup:
        run_warmup(challenger, cfg.NUM_EPISODES[1])
    else:
        print("\n  Skipping warm-up")

    # Resume from checkpoint
    if args.resume and os.path.exists(args.resume):
        challenger.load_weights(args.resume)
    elif RESUME_FROM_CHECKPOINT:
        ckpt = "models/challenger/skillego_model_final.h5"
        if os.path.exists(ckpt):
            challenger.load_weights(ckpt)

    # Champion starts as a copy of the challenger (post warm-up)
    champion = Agent(0.0)   # epsilon=0: always greedy, no exploration
    champion.copy_weights_from(challenger)

    # Phase 2: champion/challenger
    run_champion_challenger(challenger, champion, cfg.NUM_EPISODES[2])

    print("\n" + "=" * 60)
    print("  Training complete!")
    print("  Start game server: python app.py")
    print("=" * 60)
