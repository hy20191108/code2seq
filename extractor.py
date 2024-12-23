import argparse
import json
import tempfile
from typing import Dict, List, Tuple

import requests

from common import PathContextInformation
from JavaExtractor.extract import ExtractFeaturesForFile


class Extractor:
    def __init__(
        self, config, extractor_api_url, jar_path, max_path_length, max_path_width
    ):
        self.config = config
        self.max_path_length = max_path_length
        self.max_path_width = max_path_width
        self.extractor_api_url = extractor_api_url
        self.jar_path = jar_path
        self.bad_characters_table = str.maketrans("", "", "\t\r\n")

    @staticmethod
    def post_request(url, code_string):
        return requests.post(
            url,
            data=json.dumps(
                {"code": code_string, "decompose": True}, separators=(",", ":")
            ),
        )

    def emulated_post_request(self, jar_path, code_string):
        with tempfile.NamedTemporaryFile(
            "w+", encoding="utf8", delete=True
        ) as temp_file:
            temp_file.write(code_string)
            temp_file.flush()  # Ensure the written content is saved

            args = argparse.Namespace(
                jar=jar_path,
                max_path_length=self.max_path_length,
                max_path_width=self.max_path_width,
            )

            return ExtractFeaturesForFile(args, temp_file.name)

    def extract_paths(self, code_string):
        # response = self.post_request(self.extractor_api_url, code_string)
        # response_array = json.loads(response.text)
        # if "errorType" in response_array:
        #     raise ValueError(response.text)
        # if "errorMessage" in response_array:
        #     raise TimeoutError(response.text)

        response_array = self.emulated_post_request(self.jar_path, code_string)

        pc_info_dict: Dict[Tuple[str, str, str], PathContextInformation] = {}
        code_info_list: List[List[PathContextInformation]] = []

        result = []
        for single_method in response_array:
            method_info_list: List[PathContextInformation] = []

            method_name = single_method["target"]
            current_result_line_parts = [method_name]
            contexts = single_method["paths"]
            for context in contexts[: self.config.DATA_NUM_CONTEXTS]:
                pc_info = PathContextInformation(context)
                current_result_line_parts += [str(pc_info)]
                pc_info_dict[(pc_info.token1, pc_info.shortPath, pc_info.token2)] = (
                    pc_info
                )
                method_info_list.append(pc_info)
            space_padding = " " * (self.config.DATA_NUM_CONTEXTS - len(contexts))
            print("len of current_result_line_parts", len(current_result_line_parts))
            result_line = " ".join(current_result_line_parts) + space_padding
            result.append(result_line)
            code_info_list.append(method_info_list)

        return result, pc_info_dict, code_info_list
