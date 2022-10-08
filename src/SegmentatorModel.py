import onnxruntime
from abc import ABC
import numpy as np


class SegmentatorModel(ABC):

    def __init__(self, model: str) -> None:
        self.model_name = model

    def __call__(self, image: np.ndarray):
        session = onnxruntime.InferenceSession(self.model_name)
        masks = session.run(None, {'input': image})[0]
        masks = np.where(masks > 0.5, 1, 0)

        return masks.tolist()

