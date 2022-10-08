import onnxruntime
from abc import ABC
import numpy as np


class SegmentatorModel(ABC):

    def __init__(self, model: str) -> None:
        """
        Get's model names's to load it.
        Args:
            model: path to model's onnx.
        """
        self.model_name = model

    def __call__(self, image: np.ndarray):
        """
        Get path to image, segments and returns dict with labels and masks .
        Args:
            image: path to image.
        Returns: dict with label name and mask in Pillow Image.

        """
        session = onnxruntime.InferenceSession(self.model_name)
        masks = session.run(None, {'input': image})[0]
        masks = np.where(masks > 0.5, 1, 0)

        return masks.tolist()

