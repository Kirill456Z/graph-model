from typing import List
import numpy as np


def flatten(dialgue_embaddings: List[np.array]):
    return np.concatenate(dialgue_embaddings, axis=0)
