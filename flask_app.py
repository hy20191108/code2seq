import sys
from pathlib import Path

# Add project root to sys.path for shared imports
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Setup logging
from shared.logger_manager import LoggerManager

base_dir = Path(__file__).resolve().parents[2]
LoggerManager.setup_logging("code2seq_logging_config.yaml", base_dir)
logger = LoggerManager("code2seq_flask_app")
logger.info("code2seq started")

import pickle
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request

import _code2seq
from config import Config
from model import Model

# Initialize Flask app
app = Flask(__name__)

def initialize_predictor():
    from interactive_predict import InteractivePredictor

    args = _code2seq.get_args()
    logger.info(f"Arguments: {args}")
    np.random.seed(args.seed)
    tf.compat.v1.set_random_seed(args.seed)
    config = Config.get_default_config(args)
    model = Model(config)
    return InteractivePredictor(config, model)

predictor = initialize_predictor()

@app.route("/api/code2seq", methods=["POST"])
def process_data():
    try:
        source_code = request.json.get("source_code")
        if not source_code:
            return jsonify({"error": "No source code provided"}), 400

        logger.info(f"Processing source code: {source_code}")
        result = predictor.get(source_code)
        data_binary = pickle.dumps(result)
        return data_binary
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001)
