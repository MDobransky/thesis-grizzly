from collections import defaultdict
from typing import List, Callable, Dict, Tuple

import numpy as np


class RankingMechanism:
    def __init__(self):
        pass

    @staticmethod
    def rank_func(ranking: List[Tuple[int, List[float]]],
                  func: Callable[[List[float]], float] = np.min) -> List[Tuple[int, float]]:
        return sorted(ranking, key=lambda item: func(item[1]))


    @staticmethod
    def borda_count(rankings):
        """Only names"""
        elements = defaultdict(int)
        for ranking in rankings:
            for item, weight in zip(ranking, np.linspace(1,0, num=len(ranking))):
                elements[item] += weight

        return [k for k, v in sorted(elements.items(), key=lambda item: -item[1])]

    @staticmethod
    def summing(rankings: List[str]):
        elements = defaultdict(int)
        for ranking in rankings:
            for rank, item in enumerate(ranking):
                elements[item] += rank

        sorted_ranking = sorted(elements.items(), key=lambda kv: (kv[1], kv[0]))
        return [key for key, value in sorted_ranking]
