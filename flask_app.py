import pickle

import numpy as np
import tensorflow as tf
from flask import Flask, request

from code2seq import _code2seq
from code2seq.config import Config
from code2seq.interactive_predict import InteractivePredictor
from code2seq.model import Model

app = Flask(__name__)

# generate predictor
args = _code2seq.get_args()
np.random.seed(args.seed)
tf.compat.v1.set_random_seed(args.seed)
config = Config.get_default_config(args)
model = Model(config)
predictor = InteractivePredictor(config, model)


@app.route("/api/code2seq", methods=["POST"])
def process_data():
    source_code = request.json["source_code"]
    print(source_code)
    result = predictor.get(source_code)
    data_binary = pickle.dumps(result)

    return data_binary


if __name__ == "__main__":
    app.run(debug=True)
