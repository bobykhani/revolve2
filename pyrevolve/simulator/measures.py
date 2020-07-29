import string
from random import random
from typing import List, Dict


class Measures:

    def __init__(self):
        self.measures: Dict[string, List[float]] = {'distances': [],
                                                    'times': [],
                                                    'batteries': [],
                                                    'positions': []}

    def test(self):
        for key in self.measures.keys():
            self.measures[key] = [random.uniform(0.0, 1.0) for _ in range(10)]