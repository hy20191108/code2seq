import pickle

import requests


class FlaskWrapper:
    def __init__(self) -> None:
        pass

    def get(self, source_code):
        url = "http://127.0.0.1:5001/api/code2seq"
        data = {"source_code": source_code}
        response = requests.post(url, json=data)
        data_binary = response.content
        return pickle.loads(data_binary)


if __name__ == "__main__":
    wrapper = FlaskWrapper()

    with open("Input.java") as f:
        source_code = f.read()

    print(wrapper.get(source_code))
