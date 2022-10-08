from src.SegmentatorModel import SegmentatorModel


def model_predict_masks(model, image):
    model = SegmentatorModel(model)
    masks = model(image)

    return masks
