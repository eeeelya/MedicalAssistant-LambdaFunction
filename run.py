from src.SegmentatorModel import SegmentatorModel
from src.EndoModel import EndoModel


def model_predict_masks(model, image):
    model = SegmentatorModel(model)
    masks = model(image)
    return masks


def model_predict_classes(model, image):
    model = EndoModel(model)
    labels = model(image)
    return labels
