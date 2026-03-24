"""
Script to run multiple AI vs AI games and collect statistics
"""

import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import json
from datetime import datetime

from models.game import Game
from ml.ai_player import get_random_move, get_heuristic_move, calculate_power_score


def play_single_game(agent1_type="random", agent2_type="heuristic", max_moves=200):
    """
    Play a single game between two AI agents
    
    Args:
        agent1_type: Type of agent for Player 1
        agent2_type: Type of agent for Player 2
        max_moves: Maximum number of moves before considering the game a draw
        
    Returns:
        dict: Game statistics
    """
    game = Game()
    
    # Set up agents
    agent_functions = {
        "random": get_random_move,
        "heuristic": get_heuristic_move
        # Add more agent types here as they are implemented
    }
    
    agent1 = agent_functions.get(agent1_type, get_random_move)
    agent2 = agent_functions.get(agent2_type, get_heuristic_move)
    
    # Initialize statistics
    stats = {
        "winner": None,
        "moves": 0,
        "uncovered_pieces": 0,
        "captures": {1: 0, 2: 0},
        "final_power": {1: 0, 2: 0},
        "moves_by_type": {"uncover": 0, "move": 0, "capture": 0}
    }
    
    # Play the game
    move_count = 0
    while not game.game_over and move_count < max_moves:
        current_agent = agent1 if game.current_player == 1 else agent2
        
        # Get move from agent
        move = current_agent(game)
        
        # Apply the move
        if move["type"] == "uncover":
            result = game.uncover(move["row"], move["col"])
            stats["moves_by_type"]["uncover"] += 1
        elif move["type"] == "move":
            result = game.move(move["from_row"], move["from_col"], move["to_row"], move["to_col"])
            
            if "captured" in result:
                opponent = 3 - game.current_player  # Toggle between 1 and 2
                stats["captures"][opponent] += 1
                stats["moves_by_type"]["capture"] += 1
            else:
                stats["moves_by_type"]["move"] += 1
        
        move_count += 1
    
    # Count uncovered pieces
    uncovered_count = 0
    for row in range(game.BOARD_SIZE):
        for col in range(game.BOARD_SIZE):
            piece = game.board[row][col]
            if piece and not piece["covered"]:
                uncovered_count += 1
    
    # Calculate final power scores
    stats["final_power"][1] = calculate_power_score(game, 1)
    stats["final_power"][2] = calculate_power_score(game, 2)
    
    # Record game outcome and stats
    stats["moves"] = move_count
    stats["uncovered_pieces"] = uncovered_count
    
    if game.state == "player1_won":
        stats["winner"] = 1
    elif game.state == "player2_won":
        stats["winner"] = 2
    elif move_count >= max_moves:
        # Game ended due to move limit - determine winner by power score
        if stats["final_power"][1] > stats["final_power"][2]:
            stats["winner"] = 1
        elif stats["final_power"][2] > stats["final_power"][1]:
            stats["winner"] = 2
        else:
            stats["winner"] = 0  # Draw
    
    return stats


def run_multiple_games(agent1_type="random", agent2_type="heuristic", num_games=100, max_moves=200):
    """
    Run multiple games between two AI agents and collect statistics
    
    Args:
        agent1_type: Type of agent for Player 1
        agent2_type: Type of agent for Player 2
        num_games: Number of games to run
        max_moves: Maximum number of moves per game
        
    Returns:
        dict: Aggregated statistics
    """
    print(f"Running {num_games} games: {agent1_type} (Player 1) vs {agent2_type} (Player 2)")
    
    # Use process pool to parallelize game execution
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all games to the executor
        futures = [executor.submit(play_single_game, agent1_type, agent2_type, max_moves) 
                 for _ in range(num_games)]
        
        # Process results as they complete
        completed = 0
        for future in futures:
            results.append(future.result())
            completed += 1
            if completed % 10 == 0 or completed == num_games:
                print(f"Progress: {completed}/{num_games} games completed")
    
    # Aggregate statistics
    agg_stats = {
        "total_games": num_games,
        "wins": {1: 0, 2: 0, "draw": 0},
        "win_rate": {1: 0.0, 2: 0.0, "draw": 0.0},
        "avg_moves": 0,
        "avg_uncovered": 0,
        "avg_captures": {1: 0, 2: 0},
        "avg_final_power": {1: 0, 2: 0},
        "avg_moves_by_type": {"uncover": 0, "move": 0, "capture": 0}
    }
    
    # Calculate aggregates
    for stats in results:
        if stats["winner"] == 1:
            agg_stats["wins"][1] += 1
        elif stats["winner"] == 2:
            agg_stats["wins"][2] += 1
        else:
            agg_stats["wins"]["draw"] += 1
        
        agg_stats["avg_moves"] += stats["moves"]
        agg_stats["avg_uncovered"] += stats["uncovered_pieces"]
        
        for player in [1, 2]:
            agg_stats["avg_captures"][player] += stats["captures"][player]
            agg_stats["avg_final_power"][player] += stats["final_power"][player]
        
        for move_type in stats["moves_by_type"]:
            agg_stats["avg_moves_by_type"][move_type] += stats["moves_by_type"][move_type]
    
    # Calculate averages
    agg_stats["avg_moves"] /= num_games
    agg_stats["avg_uncovered"] /= num_games
    
    for player in [1, 2]:
        agg_stats["avg_captures"][player] /= num_games
        agg_stats["avg_final_power"][player] /= num_games
    
    for move_type in agg_stats["avg_moves_by_type"]:
        agg_stats["avg_moves_by_type"][move_type] /= num_games
    
    # Calculate win rates
    agg_stats["win_rate"][1] = agg_stats["wins"][1] / num_games
    agg_stats["win_rate"][2] = agg_stats["wins"][2] / num_games
    agg_stats["win_rate"]["draw"] = agg_stats["wins"]["draw"] / num_games
    
    return agg_stats


def save_stats_to_file(stats, agent1_type, agent2_type):
    """Save statistics to a JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"stats_{agent1_type}_vs_{agent2_type}_{timestamp}.json"
    
    # Create stats directory if it doesn't exist
    os.makedirs("stats", exist_ok=True)
    
    filepath = os.path.join("stats", filename)
    with open(filepath, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Statistics saved to {filepath}")
    return filepath


def plot_statistics(stats, agent1_type, agent2_type, save_path=None):
    """Plot statistics as charts"""
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Win rate pie chart
    win_labels = [f"Player 1 ({agent1_type})", f"Player 2 ({agent2_type})", "Draw"]
    win_values = [stats["win_rate"][1], stats["win_rate"][2], stats["win_rate"]["draw"]]
    axs[0, 0].pie(win_values, labels=win_labels, autopct='%1.1f%%', startangle=90)
    axs[0, 0].set_title('Win Rate')
    
    # 2. Move types bar chart
    move_types = list(stats["avg_moves_by_type"].keys())
    move_values = list(stats["avg_moves_by_type"].values())
    axs[0, 1].bar(move_types, move_values)
    axs[0, 1].set_title('Average Moves by Type')
    axs[0, 1].set_ylabel('Average Count')
    
    # 3. Captures bar chart
    players = [f"Player 1 ({agent1_type})", f"Player 2 ({agent2_type})"]
    capture_values = [stats["avg_captures"][1], stats["avg_captures"][2]]
    axs[1, 0].bar(players, capture_values)
    axs[1, 0].set_title('Average Captures')
    axs[1, 0].set_ylabel('Average Count')
    
    # 4. Final power bar chart
    power_values = [stats["avg_final_power"][1], stats["avg_final_power"][2]]
    axs[1, 1].bar(players, power_values)
    axs[1, 1].set_title('Average Final Power')
    axs[1, 1].set_ylabel('Average Score')
    
    # Add overall title
    plt.suptitle(f"{agent1_type} vs {agent2_type} - {stats['total_games']} Games\n"
                f"Avg Moves: {stats['avg_moves']:.1f}, Avg Uncovered: {stats['avg_uncovered']:.1f}",
                fontsize=16)
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plot_dir = os.path.join("stats", "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"plot_{save_path.split('/')[-1].replace('.json', '.png')}")
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
    
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multiple AI vs AI games in Skillego and collect statistics")
    parser.add_argument("--agent1", type=str, default="random", choices=["random", "heuristic"],
                      help="Type of agent for Player 1 (default: random)")
    parser.add_argument("--agent2", type=str, default="heuristic", choices=["random", "heuristic"],
                      help="Type of agent for Player 2 (default: heuristic)")
    parser.add_argument("--games", type=int, default=100,
                      help="Number of games to run (default: 100)")
    parser.add_argument("--max-moves", type=int, default=200,
                      help="Maximum number of moves per game (default: 200)")
    parser.add_argument("--no-plot", action="store_true",
                      help="Don't display plots (useful for headless environments)")
    
    args = parser.parse_args()
    
    # Run games and collect statistics
    start_time = time.time()
    stats = run_multiple_games(
        agent1_type=args.agent1,
        agent2_type=args.agent2,
        num_games=args.games,
        max_moves=args.max_moves
    )
    elapsed_time = time.time() - start_time
    
    # Print summary
    print("\nResults:")
    print(f"Total games: {stats['total_games']}")
    print(f"Player 1 ({args.agent1}) wins: {stats['wins'][1]} ({stats['win_rate'][1]:.1%})")
    print(f"Player 2 ({args.agent2}) wins: {stats['wins'][2]} ({stats['win_rate'][2]:.1%})")
    print(f"Draws: {stats['wins']['draw']} ({stats['win_rate']['draw']:.1%})")
    print(f"Average moves per game: {stats['avg_moves']:.2f}")
    print(f"Time taken: {elapsed_time:.2f} seconds")
    
    # Save statistics to file
    stats_file = save_stats_to_file(stats, args.agent1, args.agent2)
    
    # Plot statistics
    if not args.no_plot:
        plot_statistics(stats, args.agent1, args.agent2, save_path=stats_file)
