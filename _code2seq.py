from argparse import ArgumentParser

import numpy as np
import tensorflow as tf

try:
    from model_tf2 import ModelTF2
    from reader_tf2 import ReaderTF2
    TF2_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    TF2_AVAILABLE = False

from config import Config
from interactive_predict import InteractivePredictor
from model import Model


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--data",
        dest="data_path",
        help="path to preprocessed dataset",
        required=False,
    )
    parser.add_argument(
        "-te",
        "--test",
        dest="test_path",
        help="path to test file",
        metavar="FILE",
        required=False,
    )

    parser.add_argument(
        "-s",
        "--save_prefix",
        dest="save_path_prefix",
        help="path to save file",
        metavar="FILE",
        required=False,
    )
    parser.add_argument(
        "-l",
        "--load",
        dest="load_path",
        help="path to saved file",
        metavar="FILE",
        required=False,
    )
    parser.add_argument(
        "--release",
        action="store_true",
        help="if specified and loading a trained model, release the loaded model for a smaller model "
        "size.",
    )
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=239)
    parser.add_argument(
        "--tf2",
        action="store_true",
        help="use the experimental TensorFlow 2.x implementation",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    np.random.seed(args.seed)
    tf.compat.v1.set_random_seed(args.seed)

    if args.debug:
        config = Config.get_debug_config(args)
    else:
        config = Config.get_default_config(args)

    if args.tf2 and TF2_AVAILABLE:
        model = ModelTF2(config)
        print("Created TensorFlow 2.x model")
        train_dataset = None
        if config.TRAIN_PATH:
            reader = ReaderTF2(config.TRAIN_PATH + ".train.c2s", config.BATCH_SIZE)
            train_dataset = reader.get_dataset()
        if train_dataset is not None:
            model.train(train_dataset)
    else:
        model = Model(config)
        print("Created model")
        if config.TRAIN_PATH:
            model.train()
    if config.TEST_PATH and not args.data_path and not args.tf2:
        results, precision, recall, f1, rouge = model.evaluate()
        print("Accuracy: " + str(results))
        print(
            "Precision: "
            + str(precision)
            + ", recall: "
            + str(recall)
            + ", F1: "
            + str(f1)
        )
        print("Rouge: ", rouge)
    if args.predict and not args.tf2:
        predictor = InteractivePredictor(config, model)
        predictor.predict()
    if args.release and args.load_path and not args.tf2:
        model.evaluate(release=True)
    if not args.tf2:
        model.close_session()
