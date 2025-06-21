import sys
from pathlib import Path
from typing import Union, Tuple

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
import traceback

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request, Response

import _code2seq  # type: ignore
from config import Config  # type: ignore
from model import Model  # type: ignore

# Initialize Flask app
app = Flask(__name__)


def initialize_predictor():
    from interactive_predict import InteractivePredictor  # type: ignore

    args = _code2seq.get_args()
    logger.info(f"Arguments: {args}")
    np.random.seed(args.seed)
    tf.compat.v1.set_random_seed(args.seed)
    config = Config.get_default_config(args)
    model = Model(config)
    return InteractivePredictor(config, model)


predictor = initialize_predictor()


@app.route("/", methods=["GET"])
def index():
    """Root endpoint - returns basic server information"""
    return jsonify({
        "service": "code2seq",
        "status": "running",
        "version": "1.0.0",
        "description": "Code2Seq sequence generation service",
        "endpoints": {
            "/": "Server information",
            "/health": "Health check endpoint",
            "/api/code2seq": "Main processing API (POST)"
        }
    })


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    try:
        # Simple test code for health verification
        test_code = "public class Test { int getValue() { return 42; } }"
        
        logger.info("Health check: testing predictor")
        result = predictor.get(test_code)
        logger.info(f"Health check: got result with {len(result) if result else 0} items")
        
        return jsonify({
            "status": "healthy",
            "service": "code2seq",
            "predictor": "working",
            "test_result_count": len(result) if result else 0,
            "message": "All systems operational"
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "service": "code2seq",
            "error": str(e),
            "message": "Service is experiencing issues"
        }), 500


@app.route("/api/code2seq", methods=["POST"])
def process_data() -> Union[bytes, Tuple[Response, int]]:
    """Main processing API"""
    try:
        # Request data validation and logging
        if not request.json:
            logger.error("No JSON data in request")
            return jsonify({"error": "No JSON data provided"}), 400

        source_code = request.json.get("source_code")
        if not source_code:
            logger.error("No source code in request data")
            return jsonify({"error": "No source code provided"}), 400

        logger.info(
            f"Processing source code (length: {len(source_code)}): {source_code[:100]}..."
        )

        # Call predictor.get() with logging
        logger.info("Calling predictor.get() method")
        result = predictor.get(source_code)
        logger.info(f"predictor.get() returned result type: {type(result)}")

        # Serialize with pickle and log
        logger.info("Serializing result with pickle")
        data_binary = pickle.dumps(result)
        logger.info(f"Successfully serialized data (size: {len(data_binary)} bytes)")

        return data_binary
    except Exception as e:
        # Detailed error information
        error_trace = traceback.format_exc()
        logger.error(f"Error processing data: {e}")
        logger.error(f"Full traceback:\n{error_trace}")

        # Identify which stage the error occurred
        if "predictor.get" in error_trace:
            logger.error("Error occurred in predictor.get() method")
        elif "pickle.dumps" in error_trace:
            logger.error("Error occurred in pickle serialization")
        else:
            logger.error("Error occurred in other part of the process")

        return jsonify({"error": "Internal server error", "details": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5001)
