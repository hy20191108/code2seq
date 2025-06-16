import hashlib
from pathlib import Path

from shared.logger_manager import LoggerManager

from common import Common, SingleTimeStepPrediction
from config import Config
from extractor import Extractor
from model import Model

SHOW_TOP_CONTEXTS = 3000
MAX_PATH_LENGTH = 8
MAX_PATH_WIDTH = 10000
EXTRACTION_API = (
    "https://po3g2dx2qa.execute-api.us-east-1.amazonaws.com/production/extractmethods"
)

JAR_PATH = (
    Path(__file__).parent
    / "JavaExtractor/JPredict/target/JavaExtractor-0.0.2-SNAPSHOT.jar"
).resolve()


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
            MAX_PATH_WIDTH,
        )
        self.logger = LoggerManager("interactive_predict").getlogger()
        self.logger.info("Configured interactive predictor")
        self.logger.info(f"JAR_PATH: {JAR_PATH}")
        self.logger.info(f"EXTRACTION_API: {EXTRACTION_API}")
        self.logger.info(f"SHOW_TOP_CONTEXTS: {SHOW_TOP_CONTEXTS}")
        self.logger.info(f"MAX_PATH_LENGTH: {MAX_PATH_LENGTH}")
        self.logger.info(f"MAX_PATH_WIDTH: {MAX_PATH_WIDTH}")

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
                predict_lines, pc_info_dict, pc_info_list = (
                    self.path_extractor.extract_paths(user_input)
                )
            except ValueError:
                continue
            model_results = self.model.predict(predict_lines)

            code_prediction = Common.parse_results(model_results, pc_info_dict)
            for method_prediction in code_prediction:
                print("Original name:\t" + method_prediction.original_name)
                if self.config.BEAM_WIDTH == 0:
                    print(
                        "Predicted:\t%s"
                        % [step.prediction for step in method_prediction.predictions]
                    )
                    for timestep, single_timestep_prediction in enumerate(
                        method_prediction.predictions
                    ):
                        single_timestep_prediction: SingleTimeStepPrediction
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
                                    attention_obj["source"],
                                    attention_obj["path"],
                                    attention_obj["target"],
                                )
                            )
                else:
                    print("Predicted:")
                    for predicted_seq in method_prediction.predictions:
                        print(f"\t{predicted_seq.prediction}")

    def get(self, code_string):
        # code_string = " ".join(self.read_file(code_path))
        try:
            predict_lines, pc_info_dict, code_info_list = (
                self.path_extractor.extract_paths(code_string)
            )
        except ValueError:
            import traceback

            traceback.print_exc()
            raise ValueError("Error in extracting paths")

        # モデルから直接ベクトル情報を取得
        model_results = self.model.predict(predict_lines)

        result_list = []

        code_prediction = Common.parse_results(model_results, pc_info_dict)
        assert len(code_prediction) == len(code_info_list)

        for method_prediction, method_info_list in zip(code_prediction, code_info_list):
            one_method_path_contexts = []

            if self.config.BEAM_WIDTH == 0:
                print(
                    "Predicted:\t%s"
                    % [step.prediction for step in method_prediction.predictions]
                )
                single_timestep_prediction = method_prediction.predictions[0]
                attention_paths = single_timestep_prediction.attention_paths
                print(len(attention_paths), len(method_info_list))

                assert len(attention_paths) == len(method_info_list)

                # attention_pathsには既に単語ベクトルとASTパスベクトルが含まれている
                for attention_obj, pc_info in zip(attention_paths, method_info_list):
                    # eye2vec/context_model.pyと整合性を取るため辞書形式で返す
                    import numpy as np

                    # numpy配列の型を保持して辞書作成
                    path_context_dict = {
                        "lineColumns": pc_info.lineColumns,
                        "source": attention_obj["source"],
                        "target": attention_obj["target"],
                        "path": attention_obj["path"],
                        "attention": attention_obj["score"],
                        "vector": np.asarray(attention_obj["vector"], dtype=np.float32),
                        "source_vector": np.asarray(
                            attention_obj["source_vector"], dtype=np.float32
                        )
                        if attention_obj["source_vector"] is not None
                        else None,
                        "target_vector": np.asarray(
                            attention_obj["target_vector"], dtype=np.float32
                        )
                        if attention_obj["target_vector"] is not None
                        else None,
                        "astpath_vector": np.asarray(
                            attention_obj["astpath_vector"], dtype=np.float32
                        )
                        if attention_obj["astpath_vector"] is not None
                        else None,
                    }
                    one_method_path_contexts.append(path_context_dict)

                result_list.append(
                    (
                        method_prediction.original_name,
                        -1,  # use -1 insted of raw_prediction.code_vector,
                        one_method_path_contexts,
                    )
                )
            else:
                print("Predicted:")
                for predicted_seq in method_prediction.predictions:
                    print(f"\t{predicted_seq.prediction}")
                raise ValueError("Error in extracting paths")

        return result_list
