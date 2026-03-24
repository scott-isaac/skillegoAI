from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import random
import json
import os
from models.game import Game, GameState

app = Flask(__name__)
CORS(app)

FRONTEND_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), '..', '..'))

@app.route('/')
def index():
    return send_file(os.path.join(FRONTEND_DIR, 'index.html'))

@app.route('/<path:path>')
def static_files(path):
    full_path = os.path.join(FRONTEND_DIR, path)
    if not os.path.exists(full_path):
        return '', 404
    return send_file(full_path)

# Store active games in memory
active_games = {}

@app.route('/api/game/new', methods=['POST'])
def new_game():
    """Create a new game instance"""
    game_id = str(random.randint(1000, 9999))
    active_games[game_id] = Game()
    return jsonify({
        'game_id': game_id,
        'state': active_games[game_id].get_state()
    })

@app.route('/api/game/<game_id>', methods=['GET'])
def get_game(game_id):
    """Get the current state of a game"""
    if game_id not in active_games:
        return jsonify({'error': 'Game not found'}), 404
    
    return jsonify({
        'game_id': game_id,
        'state': active_games[game_id].get_state()
    })

@app.route('/api/game/<game_id>/uncover', methods=['POST'])
def uncover_piece(game_id):
    """Uncover a piece on the board"""
    if game_id not in active_games:
        return jsonify({'error': 'Game not found'}), 404
    
    data = request.json
    row = data.get('row')
    col = data.get('col')
    
    result = active_games[game_id].uncover(row, col)
    
    if 'error' in result:
        return jsonify(result), 400
    
    return jsonify({
        'game_id': game_id,
        'state': active_games[game_id].get_state(),
        'result': result
    })

@app.route('/api/game/<game_id>/move', methods=['POST'])
def move_piece(game_id):
    """Move a piece on the board"""
    if game_id not in active_games:
        return jsonify({'error': 'Game not found'}), 404
    
    data = request.json
    from_row = data.get('from_row')
    from_col = data.get('from_col')
    to_row = data.get('to_row')
    to_col = data.get('to_col')
    
    result = active_games[game_id].move(from_row, from_col, to_row, to_col)
    
    if 'error' in result:
        return jsonify(result), 400
    
    return jsonify({
        'game_id': game_id,
        'state': active_games[game_id].get_state(),
        'result': result
    })

@app.route('/api/game/<game_id>/valid_moves', methods=['GET'])
def get_valid_moves(game_id):
    """Get valid moves for a piece"""
    if game_id not in active_games:
        return jsonify({'error': 'Game not found'}), 404
    
    row = int(request.args.get('row'))
    col = int(request.args.get('col'))
    
    valid_moves = active_games[game_id].get_valid_moves(row, col)
    
    return jsonify({
        'valid_moves': valid_moves
    })

@app.route('/api/game/<game_id>/ai_move', methods=['POST'])
def ai_move(game_id):
    """Get an AI recommended move"""
    if game_id not in active_games:
        return jsonify({'error': 'Game not found'}), 404
    
    from ml.ai_player import get_ai_move
    
    # Get AI type from request
    data = request.json or {}
    ai_type = data.get('ai_type', 'heuristic')
    model_path = data.get('model_path', None)
    
    game = active_games[game_id]
    ai_move_result = get_ai_move(game, ai_type=ai_type, model_path=model_path)
    
    return jsonify({
        'move': ai_move_result
    })

@app.route('/api/ml/check_models', methods=['GET'])
def check_ml_models():
    """Check if ML models are available"""
    models = {
        'player1': os.path.exists('models/player1/skillego_model_final.h5'),
        'player2': os.path.exists('models/player2/skillego_model_final.h5')
    }
    
    models_available = models['player1'] or models['player2']
    
    return jsonify({
        'available': models_available,
        'models': models
    })

if __name__ == '__main__':
    app.run(debug=True, port=8080)
