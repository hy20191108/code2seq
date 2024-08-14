import hashlib
from pathlib import Path

from common import Common
from config import Config
from extractor import Extractor
from model import Model

SHOW_TOP_CONTEXTS = 200
MAX_PATH_LENGTH = 8
MAX_PATH_WIDTH = 2
EXTRACTION_API = (
    "https://po3g2dx2qa.execute-api.us-east-1.amazonaws.com/production/extractmethods"
)
JAR_PATH = (
    Path(__file__).parent
    / "JavaExtractor/JPredict/target/JavaExtractor-0.0.2-SNAPSHOT.jar"
)


class InteractivePredictor:
    exit_keywords = ["exit", "quit", "q"]

    def __init__(self, config: Config, model: Model):
        model.predict([])
        self.model = model
        self.config = config
        self.path_extractor = Extractor(
            config,
            EXTRACTION_API,
            JAR_PATH,
            MAX_PATH_LENGTH,
            max_path_width=2,
        )

    @staticmethod
    def read_file(input_filename):
        with open(input_filename) as file:
            return file.readlines()

    def predict(self):
        input_filename = "Input.java"
        print("Serving")
        while True:
            print(
                'Modify the file: "'
                + input_filename
                + '" and press any key when ready, or "q" / "exit" to exit'
            )
            user_input = input()
            if user_input.lower() in self.exit_keywords:
                print("Exiting...")
                return
            user_input = " ".join(self.read_file(input_filename))
            try:
                predict_lines, pc_info_dict = self.path_extractor.extract_paths(
                    user_input
                )
            except ValueError:
                continue
            model_results = self.model.predict(predict_lines)

            prediction_results = Common.parse_results(
                model_results, pc_info_dict, topk=SHOW_TOP_CONTEXTS
            )
            for index, method_prediction in prediction_results.items():
                print("Original name:\t" + method_prediction.original_name)
                if self.config.BEAM_WIDTH == 0:
                    print(
                        "Predicted:\t%s"
                        % [step.prediction for step in method_prediction.predictions]
                    )
                    for timestep, single_timestep_prediction in enumerate(
                        method_prediction.predictions
                    ):
                        print("Attention:")
                        print(
                            "TIMESTEP: %d\t: %s"
                            % (timestep, single_timestep_prediction.prediction)
                        )
                        for attention_obj in single_timestep_prediction.attention_paths:
                            vector = attention_obj["vector"]
                            vector_hash = hashlib.md5(vector.tobytes()).hexdigest()[:5]
                            print(
                                "score:{:f}\tvecHash:{}\tcontext: {},{},{}".format(
                                    attention_obj["score"],
                                    vector_hash,
                                    attention_obj["token1"],
                                    attention_obj["path"],
                                    attention_obj["token2"],
                                )
                            )
                else:
                    print("Predicted:")
                    for predicted_seq in method_prediction.predictions:
                        print(f"\t{predicted_seq.prediction}")

    def get(self, code_path):
        code_string = " ".join(self.read_file(code_path))
        try:
            predict_lines, pc_info_dict = self.path_extractor.extract_paths(code_string)
        except ValueError:
            raise ValueError("Error in extracting paths")

        model_results = self.model.predict(predict_lines)

        prediction_results = Common.parse_results(
            model_results, pc_info_dict, topk=SHOW_TOP_CONTEXTS
        )
        for index, method_prediction in prediction_results.items():
            print("Original name:\t" + method_prediction.original_name)
            if self.config.BEAM_WIDTH == 0:
                one_method_astpaths = []
                print(
                    "Predicted:\t%s"
                    % [step.prediction for step in method_prediction.predictions]
                )
                for timestep, single_timestep_prediction in enumerate(
                    method_prediction.predictions
                ):
                    print("Attention:")
                    print(
                        "TIMESTEP: %d\t: %s"
                        % (timestep, single_timestep_prediction.prediction)
                    )
                    for attention_obj in single_timestep_prediction.attention_paths:
                        vector = attention_obj["vector"]
                        vector_hash = hashlib.md5(vector.tobytes()).hexdigest()[:5]
                        print(
                            "score:{:f}\tvecHash:{}\tcontext: {},{},{}".format(
                                attention_obj["score"],
                                vector_hash,
                                attention_obj["token1"],
                                attention_obj["path"],
                                attention_obj["token2"],
                            )
                        )
            else:
                print("Predicted:")
                for predicted_seq in method_prediction.predictions:
                    print(f"\t{predicted_seq.prediction}")
