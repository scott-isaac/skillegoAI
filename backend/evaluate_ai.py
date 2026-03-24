"""
Script to integrate multiple AIs and evaluate them
"""

import os
import argparse
import numpy as np
import tensorflow as tf
import time
import json
import matplotlib.pyplot as plt
from datetime import datetime

from models.game import Game
from ml.ai_player import get_random_move, get_heuristic_move, create_model, get_model_move
from self_play_train import run_self_play_training


def compare_agents(agent1, agent2, num_games=10, max_moves=200, verbose=False):
    """
    Compare two agents by having them play against each other
    
    Args:
        agent1: First agent function or model
        agent2: Second agent function or model
        num_games: Number of games to play
        max_moves: Maximum moves per game
        verbose: Whether to print detailed game information
        
    Returns:
        dict: Statistics about the games
    """
    # Initialize statistics
    stats = {
        "wins": {1: 0, 2: 0, "draw": 0},
        "avg_moves": 0,
        "total_games": num_games
    }
    
    for game_num in range(num_games):
        if verbose:
            print(f"Game {game_num+1}/{num_games}")
        
        game = Game()
        move_count = 0
        
        while not game.game_over and move_count < max_moves:
            # Get the current agent
            current_agent = agent1 if game.current_player == 1 else agent2
            
            # Get and apply the move
            if isinstance(current_agent, tf.keras.Model):
                move = get_model_move(game, current_agent)
            else:
                move = current_agent(game)
            
            # Execute the move
            if move["type"] == "uncover":
                game.uncover(move["row"], move["col"])
            elif move["type"] == "move":
                game.move(move["from_row"], move["from_col"], move["to_row"], move["to_col"])
            
            move_count += 1
        
        # Record game outcome
        if game.state == "player1_won":
            stats["wins"][1] += 1
            if verbose:
                print(f"Player 1 wins in {move_count} moves")
        elif game.state == "player2_won":
            stats["wins"][2] += 1
            if verbose:
                print(f"Player 2 wins in {move_count} moves")
        else:
            stats["wins"]["draw"] += 1
            if verbose:
                print(f"Draw after {move_count} moves")
        
        stats["avg_moves"] += move_count
    
    # Calculate averages
    stats["avg_moves"] /= num_games
    
    # Calculate win rates
    stats["win_rate"] = {
        1: stats["wins"][1] / num_games,
        2: stats["wins"][2] / num_games,
        "draw": stats["wins"]["draw"] / num_games
    }
    
    return stats


def evaluate_model(model_path, opponent_type="heuristic", num_games=10, player_id=1):
    """
    Evaluate a trained model against a baseline opponent
    
    Args:
        model_path: Path to the trained model
        opponent_type: Type of opponent ("random" or "heuristic")
        num_games: Number of games to play
        player_id: Which player the model should play as (1 or 2)
        
    Returns:
        dict: Evaluation statistics
    """
    # Load the model
    print(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Set up the opponent
    if opponent_type == "random":
        opponent = get_random_move
    else:
        opponent = get_heuristic_move
    
    # Set up agents based on player_id
    if player_id == 1:
        agent1 = model
        agent2 = opponent
        agent1_name = "Model"
        agent2_name = opponent_type.capitalize()
    else:
        agent1 = opponent
        agent2 = model
        agent1_name = opponent_type.capitalize()
        agent2_name = "Model"
    
    # Evaluate the model
    print(f"Evaluating {agent1_name} (Player 1) vs {agent2_name} (Player 2) for {num_games} games")
    stats = compare_agents(agent1, agent2, num_games=num_games, verbose=True)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Player 1 ({agent1_name}) wins: {stats['wins'][1]} ({stats['win_rate'][1]:.1%})")
    print(f"Player 2 ({agent2_name}) wins: {stats['wins'][2]} ({stats['win_rate'][2]:.1%})")
    print(f"Draws: {stats['wins']['draw']} ({stats['win_rate']['draw']:.1%})")
    print(f"Average moves per game: {stats['avg_moves']:.2f}")
    
    return stats


def train_and_evaluate(episodes=100, eval_games=10):
    """
    Train using self-play and evaluate the resulting model
    
    Args:
        episodes: Number of training episodes
        eval_games: Number of evaluation games
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"training_run_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Train the agents
    print(f"Training for {episodes} episodes...")
    run_self_play_training(episodes=episodes)
    
    # Evaluate the resulting models
    model_paths = [
        "models/player1/skillego_model_final.h5",
        "models/player2/skillego_model_final.h5"
    ]
    
    results = {}
    
    for i, model_path in enumerate(model_paths):
        player_id = i + 1
        model_name = f"Player{player_id}"
        
        print(f"\nEvaluating {model_name} model...")
        
        # Evaluate against random
        print("\nAgainst Random:")
        random_stats = evaluate_model(
            model_path, opponent_type="random", num_games=eval_games, player_id=1
        )
        
        # Evaluate against heuristic
        print("\nAgainst Heuristic:")
        heuristic_stats = evaluate_model(
            model_path, opponent_type="heuristic", num_games=eval_games, player_id=1
        )
        
        results[model_name] = {
            "random": random_stats,
            "heuristic": heuristic_stats
        }
    
    # Save results to file
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    # Plot results
    plot_evaluation_results(results, output_dir)


def plot_evaluation_results(results, output_dir):
    """
    Plot evaluation results
    
    Args:
        results: Evaluation results dictionary
        output_dir: Directory to save the plot
    """
    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Colors for the bars
    colors = ['#ff9999', '#9999ff']
    
    # Plot win rates against random
    models = list(results.keys())
    random_win_rates = [results[model]["random"]["win_rate"][1] for model in models]
    
    axs[0, 0].bar(models, random_win_rates, color=colors)
    axs[0, 0].set_title('Win Rate Against Random')
    axs[0, 0].set_ylim(0, 1)
    axs[0, 0].set_ylabel('Win Rate')
    
    # Plot win rates against heuristic
    heuristic_win_rates = [results[model]["heuristic"]["win_rate"][1] for model in models]
    
    axs[0, 1].bar(models, heuristic_win_rates, color=colors)
    axs[0, 1].set_title('Win Rate Against Heuristic')
    axs[0, 1].set_ylim(0, 1)
    axs[0, 1].set_ylabel('Win Rate')
    
    # Plot average moves against random
    random_avg_moves = [results[model]["random"]["avg_moves"] for model in models]
    
    axs[1, 0].bar(models, random_avg_moves, color=colors)
    axs[1, 0].set_title('Average Moves Against Random')
    axs[1, 0].set_ylabel('Average Moves')
    
    # Plot average moves against heuristic
    heuristic_avg_moves = [results[model]["heuristic"]["avg_moves"] for model in models]
    
    axs[1, 1].bar(models, heuristic_avg_moves, color=colors)
    axs[1, 1].set_title('Average Moves Against Heuristic')
    axs[1, 1].set_ylabel('Average Moves')
    
    # Add overall title
    plt.suptitle('Model Evaluation Results', fontsize=16)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, "evaluation_results.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate Skillego AI models")
    parser.add_argument("--train", action="store_true",
                      help="Train using self-play")
    parser.add_argument("--episodes", type=int, default=100,
                      help="Number of training episodes (default: 100)")
    parser.add_argument("--eval", action="store_true",
                      help="Evaluate a trained model")
    parser.add_argument("--model", type=str,
                      help="Path to the model to evaluate")
    parser.add_argument("--opponent", type=str, default="heuristic", choices=["random", "heuristic"],
                      help="Opponent type for evaluation (default: heuristic)")
    parser.add_argument("--games", type=int, default=10,
                      help="Number of evaluation games (default: 10)")
    parser.add_argument("--as-player", type=int, default=1, choices=[1, 2],
                      help="Which player the model should play as (default: 1)")
    parser.add_argument("--train-and-eval", action="store_true",
                      help="Train using self-play and then evaluate")
    
    args = parser.parse_args()
    
    if args.train:
        # Train using self-play
        run_self_play_training(episodes=args.episodes)
    
    elif args.eval and args.model:
        # Evaluate a trained model
        evaluate_model(
            args.model,
            opponent_type=args.opponent,
            num_games=args.games,
            player_id=args.as_player
        )
    
    elif args.train_and_eval:
        # Train and evaluate
        train_and_evaluate(episodes=args.episodes, eval_games=args.games)
    
    else:
        parser.print_help()
