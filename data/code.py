from typing import List

from data.method import Method


class Code:
    def __init__(self):
        self.method_list: List[Method] = []

    def append(self, method: Method):
        self.method_list.append(method)